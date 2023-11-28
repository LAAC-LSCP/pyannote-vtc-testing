# Voice Type Classifier, with pyannote 2.0

This repo contains the scripts necessary to train, tune, apply and
score VTC2.0.

## Installation

Install pyannote.audio==2.0

```shell
pip install -r requirements.txt
```

Then, install this experimental repository :

```shell
git clone https://github.com/marianne-m/pyannote-vtc-testing.git
```

Make sure you have a `database.yml` file in `~/.pyannote`.

## Usage

The `main.py` script does all you need.
Run `python main.py -h` to get help or look at the launchers script to get an
idea of the arguments for each command.
You have some launchers for Jean Zay and Oberon in the `launchers` folder

With the `main.py` script, you can :

- train a model on a given dataset's train-set
- tune the pipeline's hyperparameters on the dataset's dev-set
- apply the tuned pipeline on a dataset's test-set
- score the test-set's inference files with either IER or average Fscore

### Training

To train the model :

```shell
python main.py runs/experiment/ train \
    -p X.SpeakerDiarization.BBT2 \
    --classes babytrain \
    --model_type pyannet \
    --epoch 100
```

### Tuning

After training, you need to tune the parameters :

```shell
python main.py runs/experiment/ tune \
    -p X.SpeakerDiarization.BBT2 \
    --model_path runs/experiment/checkpoints/best.ckpt \
    --classes babytrain \
    --metric fscore
```

### Apply

You can then apply the model with the best parameters found at the tuning step :

```shell
python main.py runs/experiment/ apply \
    -p X.SpeakerDiarization.BBT2 \
    --model_path runs/experiment/checkpoints/best.ckpt \
    --classes babytrain \
    --apply_folder runs/experiment/apply/ \
    --params runs/experiment/best_params.yml
```

### Score

Finally you can score a model :

```shell
python main.py runs/experiment/ score \
    -p X.SpeakerDiarization.BBT2 \
    --model_path runs/experiment/checkpoints/best.ckpt \
    --classes babytrain \
    --metric fscore \
    --apply_folder runs/experiment/apply/ \
    --report_path runs/experiment/results/fscore.csv
```
