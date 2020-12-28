# Self-Driving Car Sim Demo

<p align='center'>
<img src='thumbnails/jungle_track.png' width='256px'>
<img src='thumbnails/lake_track.png' width='256px'>
</p>

## Introduction

## Materials

- [Miniconda](https://conda.io/miniconda.html)
- [Anaconda](https://www.continuum.io/downloads)
- [Driving Simulator](https://github.com/udacity/self-driving-car-sim)

### Dependencies

You can install all dependencies by running one of the following commands

```
# Use TensorFlow without GPU
conda env create -f environment.yml 

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

Or you can manually install the required libraries using pip.

## Procedures

### Run the pretrained model

Start up the Udacity self-driving simulator, choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```
python drive.py model.h5
```

### To train the model

You'll need the data folder which contains the training images.

```
python model.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

## Authors

- **B. Bueno** - [bbueno5000](https://github.com/bbueno5000)

## Acknowledegements

- [naokishibuya](https://github.com/naokishibuya)
- [llSourcell](https://github.com/llSourcell)
