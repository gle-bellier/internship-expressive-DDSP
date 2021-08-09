# REDIF

Reimplementation of the diffusion model proposed during the internship.

## Dataset preprocessing

Use `preprocess_dataset.py` to read contours from `dataset/contours.csv` and save them as numpy arrays. The only real _preprocessing_ happens when saving the quantized pitch (mapped to log scale and converted to int).

## Training

At the _root_ of the project lies a little `train_redif.py`. A first glance might make it look like not much, but beware ! This is where the AI magic happens.

## Evaluation

```python
import torch.nn as nn
from redif.model import Model

model = Model.load_from_checkpoint("checkpoint.ckpt")

# PREPARE CONDITIONING SIGNALS
pitch = get_quantized_pitch() # shape: B x T
pitch = ftom(pitch)
pitch = nn.functional.one_hot(pitch, 127).permute(0, 2, 1)

loudness = get_loudness().unsqueeze(1) # shape: B x 1 x T

env = torch.cat([pitch, loudness], 1)

# PREPARE INPUT NOISE
x = torch.randn(env.shape[0], model.data_dim, env.shape[-1])

# SAMPLE FROM ESTIMATED DISTRIBUTION
y = model.sample(x, env)
f0, lo = model.transform.inverse(y)

# SYNTHESIS TIME !
sound = ddsp(f0, lo)
```
