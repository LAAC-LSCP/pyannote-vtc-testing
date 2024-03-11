import argparse
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml
import pandas as pd
from pyannote.audio import Model
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.pipelines import MultiLabelSegmentation as MultilabelSegmentationPipeline
from pyannote.audio.tasks.segmentation.multilabel import MultiLabelSegmentation
from pyannote.core import Annotation
from pyannote.audio.utils.preprocessors import DeriveMetaLabels
from pyannote.database import FileFinder, get_protocol, ProtocolFile
from pyannote.database.protocol.protocol import Preprocessor
from pyannote.database.util import load_rttm, LabelMapper
from pyannote.metrics.base import BaseMetric
from pyannote.pipeline import Optimizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from functools import partial
import scipy
from random import random

class ProcessorChain:
    """
    A class to chain multiple audio preprocessing steps together. Each preprocessor
    in the chain is applied sequentially to the input audio file. The result of one
    preprocessing step is passed as input to the next step in the chain.

    Attributes:
        procs (List[Preprocessor]): A list of preprocessing objects to be applied
            sequentially to the audio files.
        key (str): The key to use for storing the output of the last preprocessing
            step in the processed file's dictionary.

    Methods:
        __call__(file: ProtocolFile) -> Any:
            Applies the chain of preprocessors to the given audio file and returns
            the output of the last preprocessor in the chain.
    """


    def __init__(self, preprocessors: List[Preprocessor], key: str):
        """
        Initializes the ProcessorChain with a list of preprocessors and a key.

        Parameters:
            preprocessors (List[Preprocessor]): The list of preprocessors to apply.
            key (str): The key to use for storing the output in the processed file's dictionary.
        """
        self.procs = preprocessors
        self.key = key

    def __call__(self, file: ProtocolFile) -> Any:
        """
        Applies the preprocessors in the chain to the given file. Each preprocessor's
        output is passed as input to the next preprocessor. The final output is stored
        in the file's dictionary using the specified key.

        Parameters:
            file (ProtocolFile): The audio file to process.

        Returns:
            The output of the last preprocessor in the chain.
        """
        file_cp: Dict[str, Any] = abs(file)
        for proc in self.procs:
            out = proc(file_cp)
            file_cp[self.key] = out

        return out


# Configuration Variables

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
"""
Determines the computing device for model training and inference.
Automatically set to 'gpu' if a CUDA-compatible GPU is available, otherwise defaults to 'cpu'.
"""

CLASSES = {
    "babytrain": {
        'classes': ["MAL", "FEM", "CHI", "KCHI"],
        'unions': {"SPEECH": ["MAL", "FEM", "CHI", "KCHI"]},
        'intersections': {}
    }
}
"""
Configuration for audio class definitions used in the babytrain project.

- 'classes': Lists the individual classes to be recognized in the audio segmentation task.
- 'unions': Defines groups of classes that should be considered together under a new label for certain analyses or tasks.
  E.g., 'SPEECH' combines all speaker classes into one category.
- 'intersections': Defines combinations of classes that, when occurring together, form a new, meaningful category.
  E.g., "ADULT": ["MAL", "FEM"] is used to identify segments where both male and female voices are present simultaneously, and "FAMILY": ["FEM", "KCHI"] 
  might be used to highlight interactions between female adults and children, indicating family-like interactions.


This setup enables flexible handling of complex class relationships, facilitating experiments with different class groupings.
"""


def validate_helper_func(current_file, pipeline, metric, label):
    """
    Validates a single file's annotations against a given metric.

    This function extracts the reference annotations for a specific label, applies a processing pipeline to the file,
    and evaluates the hypothesis (pipeline output) against the reference using the provided metric. It's typically used
    in the context of validating audio segmentation or classification models.

    Parameters:
        current_file (dict): The file to be processed, containing at least 'annotation' and 'annotated' keys.
        pipeline (callable): A processing pipeline that takes a file as input and returns a hypothesis.
        metric (callable): A function that compares the reference and hypothesis annotations, returning a metric value.
        label (str): The label of interest to extract from the file's annotations.

    Returns:
        The result of the metric evaluation.
    """
    reference = current_file["annotation"].subset([label])
    hypothesis = pipeline(current_file)
    return metric(reference, hypothesis, current_file["annotated"])



class BaseCommand:
    """
    A base class for command-line commands within the script. It provides a common
    interface and shared attributes for all commands, ensuring consistency and facilitating
    the addition of new commands with standardized behavior and properties.

    Attributes:
        COMMAND (str): A unique identifier for the command, used in the command-line interface.
        DESCRIPTION (str): A brief description of what the command does, for display in help messages.
    """
    COMMAND = "command"
    DESCRIPTION = "Command description"


    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        pass

    @classmethod
    def run(cls, args: Namespace):
        pass

    @classmethod
    def get_protocol(cls, args: Namespace) -> Protocol:
        """
        Retrieves the protocol based on specified command-line arguments. This method
        configures the protocol with necessary preprocessors, which are essential
        for preparing the data according to the defined classes and their attributes.

        Parameters:
            args (Namespace): The command-line arguments provided by the user.
                              This includes 'classes' for class definitions and
                              'protocol' for specifying which protocol to use.

        Returns:
            Protocol: An instance of the protocol configured with the specified
                      preprocessors for audio and annotation processing.
        """
        classes_kwargs = CLASSES[args.classes]
        vtc_preprocessor = DeriveMetaLabels(**classes_kwargs)
        preprocessors = {
            "audio": FileFinder(),
            "annotation": vtc_preprocessor
        }
        if args.classes == "babytrain":
            with open(Path(__file__).parent / "data/babytrain_mapping.yml") as mapping_file:
                mapping_dict = yaml.safe_load(mapping_file)["mapping"]
            preprocessors["annotation"] = ProcessorChain([
                LabelMapper(mapping_dict, keep_missing=True),
                vtc_preprocessor
            ], key="annotation")
        return get_protocol(args.protocol, preprocessors=preprocessors)


    @classmethod
    def get_task(cls, args: Namespace) -> MultiLabelSegmentation:
        """
        Creates and configures a multi-label segmentation task based on the protocol
        obtained through the `get_protocol` method. This setup is crucial for defining
        the specific audio segmentation task, including its duration and other
        relevant configurations.

        Parameters:
            args (Namespace): The command-line arguments provided by the user.
                              It is used to retrieve the protocol with the necessary
                              configurations for the task.

        Returns:
            MultiLabelSegmentation: A multi-label segmentation task instance,
                                    configured and ready for setup and execution.
        """
        protocol = cls.get_protocol(args)
        task = MultiLabelSegmentation(protocol, duration=2.00)
        task.setup()
        return task


class TrainCommand(BaseCommand):
    """
    Command for training an audio segmentation model. This class handles the setup
    and execution of model training, including the selection of the model type,
    training protocol, and optimization criteria.

    Attributes are inherited from BaseCommand.
    """
    COMMAND = "train"
    DESCRIPTION = "Train the model on the specified dataset and protocol."

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        """
        Initializes the argument parser with options specific to the training command,
        such as model type, number of epochs, and protocol selection.
        """
        parser.add_argument("-p", "--protocol", type=str,
                            default="X.SpeakerDiarization.BBT2",
                            help="Pyannote database protocol to use for training")
        parser.add_argument("--model_type", choices=["simple", "pyannote"],
                            help="Type of model to train. 'simple' for a basic model, "
                                 "'pyannote' for models from the pyannote.audio library.")
        parser.add_argument("--epoch", type=int, required=True,
                            help="Number of epochs to train the model.")

        # Additional arguments as needed...

    @classmethod
    def run(cls, args: Namespace):
        """
        Executes the training command with the specified arguments. This includes
        setting up the model, data, and training loop according to the user's choices.
        """
        # Model setup based on args.model_type
        if args.model_type == "simple":
            # 'simple' might refer to a basic, less complex model for quick experiments.
            model = SimpleSegmentationModel(task=vtc)
        else:
            # 'pyannote' option utilizes the sophisticated models provided by pyannote.audio.
            model = PyanNet(pretrained=True)

        # Extracting the value to monitor during training for model checkpointing and early stopping.
        value_to_monitor, min_or_max = vtc.val_monitor  # Defined in task/model configuration.

        # Setting up PyTorch Lightning Trainer with dynamic device selection
        trainer_kwargs = {
            'accelerator': DEVICE if torch.cuda.is_available() else "cpu",
            'devices': 1 if DEVICE == "gpu" else None,
            'callbacks': [model_checkpoint, early_stopping],
            'logger': logger
        }

        # Initialize and run the PyTorch Lightning trainer with the above configurations.
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model, datamodule=...)

        # Post-training actions (e.g., saving the model) can be added here.

# Not documenting next section because we won't use it
class TuneOptunaCommand(BaseCommand):
    COMMAND = "tuneoptuna"
    DESCRIPTION = "tune the model hyperparameters using optuna"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="X.SpeakerDiarization.BBT2",
                            help="Pyannote database")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            default="babytrain",
                            type=str, help="Model model checkpoint")
        parser.add_argument("-m", "--model_path", type=Path, required=True,
                            help="Model checkpoint to tune pipeline with")
        parser.add_argument("-nit", "--n_iterations", type=int, default=50,
                            help="Number of tuning iterations, rule of thumb is 10^(number or parameters to optimize)")
        parser.add_argument("--metric", choices=["fscore", "ier"],
                            default="fscore")
        parser.add_argument("--params", type=Path, default=Path("best_params.yml"),
                            help="Filename for param yaml file")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        # Dirty fix for the non-serialization of the task params
        pipeline = MultilabelSegmentationPipeline(segmentation=model,
                                                  fscore=args.metric == "fscore", share_min_duration=True)
        # pipeline.instantiate(pipeline.default_parameters())
        validation_files = list(protocol.development())
        optimizer = Optimizer(pipeline)
        optimizer.tune(validation_files,
                       n_iterations=args.n_iterations,
                       show_progress=True)
        best_params = optimizer.best_params
        logging.info(f"Best params: \n{best_params}")
        params_filepath: Path = args.exp_dir / args.params
        logging.info(f"Saving params to {params_filepath}")
        pipeline.instantiate(best_params)
        pipeline.dump_params(params_filepath)


class TuneCommand(BaseCommand):
    """
    Command for tuning the hyperparameters of an audio segmentation model using
    scipy's optimization tools. This command supports selecting different class
    configurations and metrics for tuning, as well as specifying a YAML file for
    initial parameters.

    The tuning process optimizes threshold values for each label to maximize
    the specified performance metric on the development set.
    """

    COMMAND = "tune"
    DESCRIPTION = "tune the model hyperparameters using scipy"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="X.SpeakerDiarization.BBT2",
                            help="Pyannote database protocol to use for tuning (e.g. X.SpeakerDiarization.BBT2).")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            default="babytrain",
                            type=str, help="Class configuration for tuning (e.g., babytrain).")
        parser.add_argument("-m", "--model_path", type=Path, required=True,
                            help="Path to the model checkpoint for tuning.")
        parser.add_argument("--metric", choices=["fscore", "ier"],
                            default="fscore", help="Performance metric to optimize.")
        parser.add_argument("--params", type=Path, default=Path("best_params.yml"),
                            help="Path to a YAML file with initial parameters.")

    @classmethod
    def run(cls, args: Namespace):
        """
        Executes the tuning process with the specified arguments. This involves
        setting up the model and pipeline, loading initial parameters from a YAML
        file if provided, and optimizing threshold values for each label.

        Optimized parameters are saved back to a YAML file for future use.
        """
        # Set up the protocol, model, and initial pipeline configuration
        protocol = cls.get_protocol(args)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )

        # Initialize the segmentation pipeline with model and selected metric
        # Contains dirty fix for the non-serialization of the task params
        pipeline = MultilabelSegmentationPipeline(segmentation=model,
                                                  fscore=args.metric == "fscore", share_min_duration=True)
        validation_files = list(protocol.development())

        # Load and update parameters from a YAML file if specified
        params = {
            "min_duration_off": 0.1,
            "min_duration_on": 0.1,
        }  # Default parameters
        if args.params.exists() and args.params != Path("best_params.yml"):
            with args.params.open('r') as file:
                user_params = yaml.safe_load(file)
                params.update(user_params)  # Update default params with user-provided ones

        # Initialize parameters
        label_names = CLASSES[args.classes]["classes"] +\
                        list(CLASSES[args.classes]["unions"].keys()) +\
                        list(CLASSES[args.classes]["intersections"].keys())
        params["thresholds"] = {label: {"onset": random(), "offset": random()} for label in label_names}

        # Instantiate the pipeline with the initial parameters before starting the optimization process
        pipeline.instantiate(params)

        # Define the objective function for threshold optimization
        def fun(threshold, considered_label):
            """
            Objective function for optimizing onset and offset thresholds for a given label.

            This function updates the segmentation pipeline with the current threshold values for onset and offset,
            evaluates the pipeline's performance on the development set using a specified metric, and returns a value
            indicating the performance loss. It's used by the scipy.optimize.minimize_scalar method to find the threshold
            values that minimize the loss, effectively maximizing the performance of the segmentation model.

            Parameters:
                threshold (float): The current threshold value being tested for both onset and offset.
                considered_label (str): The label for which the thresholds are being optimized.

            Returns:
                float: The performance loss calculated as 1 minus the absolute value of the metric score. This value is
                       minimized by the optimizer to find the optimal threshold.
            """
            # Update the model's pipeline with the current threshold values for the considered label
            pipeline.instantiate({'thresholds': {considered_label: {
                "onset": threshold,
                "offset": threshold
            }}})

            # Initialize or retrieve the metric object to evaluate model performance
            metric = pipeline.get_metric()

            # Partial function setup for validating files with fixed pipeline, metric, and label
            validate = partial(validate_helper_func, pipeline=pipeline, metric=metric, label=considered_label)

            # Apply validation function to each file in the development set and update metric
            for file in validation_files:
                _ = validate(file)

            # Calculate and return the performance loss
            return 1. - abs(metric)

        # Optimize thresholds using scipy.optimize.minimize_scalar
        for label in label_names:
            res = scipy.optimize.minimize_scalar(
                fun, bounds=(0., 1.), method='bounded', options={'maxiter': 10}, args=label
            )

            threshold = res.x.item()
            params["thresholds"][label] = {'onset': threshold, 'offset': threshold}
            pipeline.instantiate(params)

        # Save the optimized parameters to a YAML file
        params_filepath: Path = args.exp_dir / args.params
        logging.info(f"Saving params to {params_filepath}")
        pipeline.instantiate(params)
        pipeline.dump_params(params_filepath)


class ApplyCommand(BaseCommand):
    COMMAND = "apply"
    DESCRIPTION = "apply the model on some data"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="X.SpeakerDiarization.BBT2",
                            help="Pyannote database")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            default="babytrain",
                            type=str, help="Model model checkpoint")
        parser.add_argument("-m", "--model_path", type=Path, required=True,
                            help="Model checkpoint to run pipeline with")
        parser.add_argument("--params", type=Path,
                            help="Path to best params. Default to EXP_DIR/best_params.yml")
        parser.add_argument("--apply_folder", type=Path,
                            help="Path to apply folder")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        pipeline = MultilabelSegmentationPipeline(segmentation=model, share_min_duration=True)
        params_path: Path = args.params if args.params is not None else args.exp_dir / "best_params.yml"
        pipeline.load_params(params_path)
        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder
        apply_folder.mkdir(parents=True, exist_ok=True)

        for file in tqdm(list(protocol.test())):
            logging.info(f"Inference for file {file['uri']}")
            annotation: Annotation = pipeline(file)
            with open(apply_folder / (file["uri"].replace("/", "_") + ".rttm"), "w") as rttm_file:
                annotation.write_rttm(rttm_file)


class ScoreCommand(BaseCommand):
    COMMAND = "score"
    DESCRIPTION = "score some inference"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="X.SpeakerDiarization.BBT2",
                            help="Pyannote database")
        parser.add_argument("--apply_folder", type=Path,
                            help="Path to the inference files")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            default="babytrain",
                            type=str, help="Model architecture")
        parser.add_argument("--metric", choices=["fscore", "ier"],
                            default="fscore")
        parser.add_argument("--model_path", type=Path, required=True,
                            help="Model model checkpoint")
        parser.add_argument("--report_path", type=Path, required=True,
                            help="Path to report csv")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder
        annotations: Dict[str, Annotation] = {}
        for filepath in apply_folder.glob("*.rttm"):
            rttm_annots = load_rttm(filepath)
            annotations.update(rttm_annots)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        pipeline = MultilabelSegmentationPipeline(segmentation=model,
                                                  fscore=args.metric == "fscore")
        metric: BaseMetric = pipeline.get_metric()

        for file in protocol.test():
            if file["uri"] not in annotations:
                continue
            inference = annotations[file["uri"]]
            metric(file["annotation"], inference, file["annotated"])

        df: pd.DataFrame = metric.report(display=True)
        if args.report_path is not None:
            args.report_path.parent.mkdir(parents=True, exist_ok=True) 
            df.to_csv(args.report_path)


commands = [TrainCommand, TuneCommand, TuneOptunaCommand, ApplyCommand, ScoreCommand]

argparser = argparse.ArgumentParser()
argparser.add_argument("-v", "--verbose", action="store_true",
                       help="Show debug information in the standard output")
argparser.add_argument("exp_dir", type=Path,
                       help="Experimental folder")
subparsers = argparser.add_subparsers()

for command in commands:
    subparser = subparsers.add_parser(command.COMMAND)
    subparser.set_defaults(func=command.run,
                           command_class=command,
                           subparser=subparser)
    command.init_parser(subparser)

if __name__ == '__main__':
    args = argparser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if hasattr(args, "func"):
        args.func(args)
    else:
        argparser.print_help()
