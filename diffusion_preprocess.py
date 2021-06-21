import csv
import numpy as np
import pickle


def mtof(m):
    """
    converts midi note to frequency
    """
    return 440 * 2**((m - 69) / 12)


def ftom(f):
    """
    converts frequency to midi note
    """
    return 12 * (np.log(f) - np.log(440)) / np.log(2) + 69


def norm_array(x):
    """
    min max scaler
    """
    minimum = np.min(x)
    maximum = np.max(x)
    x = (x - minimum) / (maximum - minimum)
    return x, minimum, maximum


def ftopc(f):
    """
    converts frequency to pitch / cent
    """
    m_float = ftom(f)
    m_int = np.round(m_float).astype(int)
    c_float = m_float - m_int
    return m_int, c_float


def pctof(p, c):
    """
    convert pitch / cent to frequency
    """
    m = p + c
    return mtof(m)


if __name__ == "__main__":

    u_f0 = []
    u_loudness = []
    e_f0 = []
    e_loudness = []
    f0_conf = []
    events = []

    with open("dataset/contours.csv", "r") as contour:
        contour = csv.DictReader(contour)

        for row in contour:
            u_f0.append(row["u_f0"])
            u_loudness.append(row["u_loudness"])
            e_f0.append(row["e_f0"])
            e_loudness.append(row["e_loudness"])
            f0_conf.append(row["f0_conf"])
            events.append(row["events"])

    u_f0 = np.asarray(u_f0).astype(float)
    u_loudness = np.asarray(u_loudness).astype(float)
    e_f0 = np.asarray(e_f0).astype(float)
    e_loudness = np.asarray(e_loudness).astype(float)
    f0_conf = np.asarray(f0_conf).astype(float)
    events = np.asarray(events).astype(float)

    # # # data augmentation

    e_f0_pitch, e_cents = ftopc(e_f0)

    # Replicate arrays that stay the same

    u_loudness = np.tile(u_loudness, 3)
    e_loudness = np.tile(e_loudness, 3)
    f0_conf = np.tile(f0_conf, 3)
    events = np.tile(events, 3)

    # 1 step above
    e_f0_a = e_f0_pitch + 1
    u_f0_a = u_f0 + 1

    # 1 step below
    e_f0_b = e_f0_pitch - 1
    u_f0_b = u_f0 - 1

    # add cents to quantized contours :

    e_f0_a = pctof(e_f0_a, e_cents)
    e_f0_b = pctof(e_f0_b, e_cents)

    e_f0 = np.concatenate((e_f0, e_f0_a))
    u_f0 = np.concatenate((u_f0, u_f0_a))

    e_f0 = np.concatenate((e_f0, e_f0_b))
    u_f0 = np.concatenate((u_f0, u_f0_b))

    out = {
        "u_f0": u_f0,
        "u_loudness": u_loudness,
        "e_f0": e_f0,
        "e_loudness": e_loudness,
        "f0_conf": f0_conf,
        "events": events
    }

    with open("dataset/dataset-diffusion.pickle", "wb") as file_out:
        pickle.dump(out, file_out)
