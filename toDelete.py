import torch
import numpy as np
from sklearn.preprocessing import QuantileTransformer

v = torch.randn(1, 200000, 1)
m_in = torch.randn(16, 2000, 1)

v = v.squeeze(0)
print(v.shape)

m_in_f = m_in.flatten(end_dim=1)
print(m_in.shape)

q = QuantileTransformer()
f = q.fit(v)

m_norm = q.transform(m_in_f)

m_out = m_in_f.reshape(m_in.shape)

print(m_out.shape)
print(torch.equal(m_out, m_in))
