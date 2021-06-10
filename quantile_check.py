import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns

with open("dataset-augmented.pickle", "rb") as dataset:
    dataset = pickle.load(dataset)

u_l0 = dataset["u_loudness"].reshape(-1, 1)
e_l0 = dataset["e_loudness"].reshape(-1, 1)

# d_u_l0, _ = np.histogram(u_l0.reshape(-1), bins=30)
# d_e_l0, _ = np.histogram(e_l0.reshape(-1), bins=30)

q_u = QuantileTransformer(n_quantiles=30)
q_e = QuantileTransformer(n_quantiles=30)

fit_u = q_u.fit(u_l0)
fit_e = q_e.fit(e_l0)

t_u_l0 = q_u.transform(u_l0)
t_e_l0 = q_e.transform(e_l0)

r_u_l0 = q_u.inverse_transform(t_u_l0)
r_e_l0 = q_e.inverse_transform(t_e_l0)

u_l0, t_u_l0, r_u_l0 = u_l0.reshape(-1), t_u_l0.reshape(-1), r_u_l0.reshape(-1)
e_l0, t_e_l0, r_e_l0 = e_l0.reshape(-1), t_e_l0.reshape(-1), r_e_l0.reshape(-1)

sns.histplot(np.stack((u_l0, t_u_l0), axis=-1))
plt.show()

sns.histplot(np.stack((t_u_l0, r_u_l0), axis=-1))
plt.show()

sns.histplot(np.stack((e_l0, t_e_l0), axis=-1))
plt.show()

sns.histplot(np.stack((t_e_l0, r_e_l0), axis=-1))
plt.show()