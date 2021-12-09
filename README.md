# Expressive DDSP (Internship)

Implementations of machine learning models converting midi files into expressive contours of fundamental frequency and loudness for [DDSP](https://magenta.tensorflow.org/ddsp)(Google Magenta).


## Audio samples generated


Audio samples generated with DDPMs (Denoising Diffusion Probabilistic Models) and the baseline are available here: https://gle-bellier.github.io/internship-expressive-DDSP/

## Installation

1. Clone this repo:

```bash
git clone https://github.com/gle-bellier/intership-expressive-DDSP.git

```

2. Install requirements:

```bash
cd internship-expressive-DDSP
pip install -r requirements.txt

```

## Project Structure

All code dealing with _Denoising Diffusion Probabilistic Models_ (DDPMs) is available in the `diffusion/` folder (models, training methods as well as data preprocessing dedicated to the diffusion models training process).

Approaches using U-Net architecture in a deterministic context and RNNs in probabilistic fashion are respectively located in the `unet-rnn/` and `lstms/`. A reimplementation of the baseline can be found in the `baseline/` directory.

Several tools for dataset making are available and fundamental frequency $f_0$ and loudness $l_o$ computation on the URMP dataset files (made with CREPE for $f_0$ and A-weighting for loudness) are available in `dataset/` and `f0-confidence-loudness-files/`. 