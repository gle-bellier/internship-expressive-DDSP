import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

with open("dataset-augmented.pickle", "rb") as dataset:
    dataset = pickle.load(dataset)

u_l0 = dataset["u_loudness"].reshape(-1, 1)
e_l0 = dataset["e_loudness"].reshape(-1, 1)

d_u_l0 = np.histogram(u_l0, bin=30)
d_e_l0 = np.histogram(e_l0, bin=30)

q_u = QuantileTransformer(n_quantiles=30)
q_e = QuantileTransformer(n_quantiles=30)

fit_u = q_u.fit(u_l0)
fit_e = q_e.fit(e_l0)

t_u_l0 = q_u.transform(u_l0, fit_u)
t_e_l0 = q_e.transform(e_l0, fit_e)

d_t_u_l0 = np.histogram(t_u_l0, bin=30)
d_t_e_l0 = np.histogram(t_e_l0, bin=30)

fig, (ax1, ax2, ax3, ax4) = plt.subplot(2, 2)

ax1.hist(d_u_l0)
ax1.subtitle("u_f0")
ax2.hist(d_t_u_l0)
ax2.subtitle("t_u_f0")
ax3.hist(d_e_l0)
ax3.subtitle("e_f0")
ax4.hist(d_t_e_l0)
ax4.subtitle("t_e_f0")

plt.show()