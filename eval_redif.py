import torch
import torch.nn as nn

torch.set_grad_enabled(False)
from redif.model import Model, ftom
from effortless_config import Config


class args(Config):
    CKPT = None
    DDSP = None


args.parse_args()

# INSTANCIATE MODELS
model = Model.load_from_checkpoint(args.CKPT, strict=False).eval()
model.set_noise_schedule()
ddsp = torch.jit.load(args.DDSP).eval()

################################################
# TODO: replace dummy input with actual contours
pitch = torch.rand(1, 128) * 900 + 100
loudness = torch.randn(1, 128)
################################################

# FORMAT INPUT CONTOURS
pitch = torch.round(ftom(pitch)).long()
pitch = nn.functional.one_hot(pitch, 127).permute(0, 2, 1)
loudness = loudness.unsqueeze(1)

assert torch.all(pitch >= 0) and torch.all(pitch < 128), str(pitch)

env = torch.cat([pitch, loudness], 1)
x = torch.randn(env.shape[0], model.data_dim, env.shape[-1])

# SAMPLE FROM ESTIMATED DISTRIBUTION
y = model.sample(x, env)
f0, lo = model.transform.inverse(y)

# SYNTHESIS
sound = ddsp(f0.permute(0, 2, 1), lo.permute(0, 2, 1))
