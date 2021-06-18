import numpy as np
import matplotlib.pyplot as plt

number_samples = 20

with open("results.npy", "rb") as f:
    for i in range(number_samples):
        try:
            t = np.load(f)
            u_f0 = np.load(f)
            e_f0 = np.load(f)
            out_f0 = np.load(f)
            u_loudness = np.load(f)
            e_loudness = np.load(f)
            out_loudness = np.load(f)

            fig, (ax1, ax2) = plt.subplots(2, 1)

            ax1.plot(t, out_f0, label="Model")
            ax1.plot(t, u_f0, label="Midi")
            ax1.plot(t, e_f0, label="Performance")
            ax1.legend()

            # ax2.plot(t, out_loudness, label = "Model")
            # ax2.plot(t, u_loudness, label = "Midi")
            # ax2.plot(t, e_loudness, label = "Performance")
            # ax2.legend()

            plt.legend()
            plt.show()
        except:
            break
