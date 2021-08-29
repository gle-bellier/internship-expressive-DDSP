# Expressive DDSP (Internship)

Implementations of machine learning models converting midi files into expressive contours of fundamental frequency and loudness for [DDSP](https://magenta.tensorflow.org/ddsp)(Google Magenta).

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

## Flute Samples

Diffusion model samples:

<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/diffusion-results-flute-midi1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>
<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/diffusion-results-flute-test1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>

U-Net model samples:

<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/unet-results-flute-midi1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>
<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/unet-results-flute-test1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>

GRU model samples:

<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/lstm-results-flute-midi1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>
<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/lstm-results-flute-test1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>

Baseline model samples:

<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/baseline-results-flute-midi1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>
<audio controls="controls">
<source src="https://raw.githubusercontent.com/gle-bellier/internship-expressive-DDSP/main/audio-samples/baseline-results-flute-test1-pred.wav"/>
<p>Your browser does not support the audio element.</p>
</audio>
