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


def onsets_offsets(events):
    note_on = False
    onsets, offsets = np.zeros_like(events), np.zeros_like(events)
    for i in range(len(events)):
        if events[i] == -1:
            note_on = False
            offsets[i] = 1

        elif events[i] == 1:
            if note_on:
                offsets[i - 1] = 1
                onsets[i] = 1
                note_on = True
            else:
                onsets[i] = 1
                note_on = True

    return onsets, offsets


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

    e_f0, e_cents = ftopc(e_f0)
    u_f0, _ = ftopc(u_f0)

    onsets, offsets = onsets_offsets(events)

    # # # data augmentation

    # # 1 step above
    # e_f0_a = np.minimum(127 * np.ones_like(e_f0), e_f0 + 1)
    # u_f0_a = np.minimum(127 * np.ones_like(u_f0), u_f0 + 1)

    # # 1 step below
    # e_f0_b = np.maximum(np.zeros_like(e_f0), e_f0 - 1)
    # u_f0_b = np.maximum(np.zeros_like(u_f0), u_f0 - 1)

    # e_f0 = np.concatenate((e_f0, e_f0_a))
    # u_f0 = np.concatenate((u_f0, u_f0_a))

    # e_f0 = np.concatenate((e_f0, e_f0_b))
    # u_f0 = np.concatenate((u_f0, u_f0_b))

    # u_loudness = np.tile(u_loudness, 3)  # replicates 3x
    # e_loudness = np.tile(e_loudness, 3)
    # e_cents = np.tile(e_cents, 3)

    out = {
        "u_f0": u_f0,  # 0 - 127
        "u_loudness": u_loudness,
        "e_f0": e_f0,  # 0 - 127
        "e_cents": e_cents + .5,  # 0 - 1
        "e_loudness": e_loudness,
        "f0_conf": f0_conf,
        "onsets": onsets,
        "offsets": offsets
    }

    with open("dataset/dataset-article.pickle", "wb") as file_out:
        pickle.dump(out, file_out)
