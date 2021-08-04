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

    ratio = 0.05  # ratio between train/validation and test dataset

    u_f0 = []
    u_loudness = []
    e_f0 = []
    e_loudness = []
    f0_conf = []
    events = []
    print("Loading")
    with open("dataset/violin-contours-update.csv", "r") as contour:
        contour = csv.DictReader(contour)

        for row in contour:
            u_f0.append(row["u_f0"])
            u_loudness.append(row["u_loudness"])
            e_f0.append(row["e_f0"])
            e_loudness.append(row["e_loudness"])
            f0_conf.append(row["f0_conf"])
            events.append(row["events"])

    print("Dataset length : {}min {}s".format((len(u_f0) // 100) // 60,
                                              (len(u_f0) // 100) % 60))

    u_f0 = np.asarray(u_f0).astype(float)
    u_loudness = np.asarray(u_loudness).astype(float)
    e_f0 = np.asarray(e_f0).astype(float)
    e_loudness = np.asarray(e_loudness).astype(float)
    f0_conf = np.asarray(f0_conf).astype(float)
    events = np.asarray(events).astype(float)
    onsets, offsets = onsets_offsets(events)

    cut_idx = int(len(u_f0) * ratio)
    test = {
        "u_f0": u_f0[-cut_idx:],
        "u_loudness": u_loudness[-cut_idx:],
        "e_f0": e_f0[-cut_idx:],
        "e_loudness": e_loudness[-cut_idx:],
        "f0_conf": f0_conf[-cut_idx:],
        "onsets": onsets[-cut_idx:],
        "offsets": offsets[-cut_idx:]
    }

    with open("dataset/violin-test.pickle", "wb") as file_out:
        pickle.dump(test, file_out)

    # # # data augmentation on train dataset :

    u_f0 = u_f0[cut_idx:]
    u_loudness = u_loudness[cut_idx:]
    e_f0 = e_f0[cut_idx:]
    e_loudness = e_loudness[cut_idx:]
    f0_conf = f0_conf[cut_idx:]
    events = events[cut_idx:]
    onsets, offsets = onsets[cut_idx:], offsets[cut_idx:]

    e_f0_pitch, e_cents = ftopc(e_f0)
    u_f0_pitch, u_cents = ftopc(u_f0)

    # Replicate arrays that stay the same

    u_loudness = np.tile(u_loudness, 5)
    e_loudness = np.tile(e_loudness, 5)
    f0_conf = np.tile(f0_conf, 5)
    onsets = np.tile(onsets, 5)
    offsets = np.tile(offsets, 5)

    # 1 STEP ABOVE BELOW

    # 1 step above
    e_f0_1a = e_f0_pitch + 1
    u_f0_1a = u_f0_pitch + 1

    # 1 step below
    e_f0_1b = e_f0_pitch - 1
    u_f0_1b = u_f0_pitch - 1

    # add cents to quantized contours :

    e_f0_1a = pctof(e_f0_1a, e_cents)
    e_f0_1b = pctof(e_f0_1b, e_cents)
    u_f0_1a = pctof(u_f0_1a, u_cents)
    u_f0_1b = pctof(u_f0_1b, u_cents)

    # 2 STEP ABOVE BELOW

    # 2 step above
    e_f0_2a = e_f0_pitch + 2
    u_f0_2a = u_f0_pitch + 2

    # 2 step below
    e_f0_2b = e_f0_pitch - 2
    u_f0_2b = u_f0_pitch - 2

    # add cents to quantized contours :
    e_f0_2a = pctof(e_f0_2a, e_cents)
    e_f0_2b = pctof(e_f0_2b, e_cents)
    u_f0_2a = pctof(u_f0_2a, u_cents)
    u_f0_2b = pctof(u_f0_2b, u_cents)

    e_f0 = np.concatenate((e_f0, e_f0_1a))
    u_f0 = np.concatenate((u_f0, u_f0_1a))
    e_f0 = np.concatenate((e_f0, e_f0_1b))
    u_f0 = np.concatenate((u_f0, u_f0_1b))
    e_f0 = np.concatenate((e_f0, e_f0_2a))
    u_f0 = np.concatenate((u_f0, u_f0_2a))
    e_f0 = np.concatenate((e_f0, e_f0_2b))
    u_f0 = np.concatenate((u_f0, u_f0_2b))

    out = {
        "u_f0": u_f0,
        "u_loudness": u_loudness,
        "e_f0": e_f0,
        "e_loudness": e_loudness,
        "f0_conf": f0_conf,
        "onsets": onsets,
        "offsets": offsets
    }

    with open("dataset/violin-train.pickle", "wb") as file_out:
        pickle.dump(out, file_out)

    print(
        "Train dataset length : {}min {}s \n Test dataset length : {}min {}s".
        format((len(u_f0) // 100) // 60, (len(u_f0) // 100) % 60,
               (cut_idx // 100) // 60, (cut_idx // 100) % 60))
