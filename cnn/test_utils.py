import numpy as np


def detections_and_delay(tp_idx, fp_idx, seizures_idx):
    true_detections = 0
    detection_delay = []
    n_seizures = 0
    if len(seizures_idx) > 0:
        be = get_begin_end(seizures_idx)
        n_seizures = len(be)
        for i in range(n_seizures):
            s_idx = range(be[i, 0], be[i, 1] + 1)
            overlap = set(tp_idx) & set(s_idx)
            if overlap:
                true_detections += 1
                detection_delay.append(s_idx.index(min(overlap)))

    false_detections = len(fp_idx)
    i = 0
    while i < len(fp_idx)-1:
        if fp_idx[i + 1] - fp_idx[i] == 1:
            false_detections -= 1
            i += 2
        else:
            i += 1

    return {'true_detections:': true_detections, 'detection_delay:': detection_delay,
            'false_detections:': false_detections, 'n_seizures:': n_seizures}


def get_begin_end(idx):
    idx1 = np.hstack((idx, 0))
    idx2 = np.roll(idx1, 1)
    breaks = np.where(idx1 - idx2 > 1)[0][1:]
    be = np.zeros((len(breaks) + 1, 2), dtype='int32')
    be[:, 0] = np.insert(np.array([idx[i] for i in breaks]), 0, idx[0])
    be[:, 1] = np.hstack((np.array([idx[i] for i in breaks - 1]), idx[-1]))
    return be


if __name__=='__main__':
    tp_idx = [12,13,283]
    fp_idx = [1,2,3,5,6,8]
    seizures_idx = [9,10,11,12,13,14,282,283]
    print detections_and_delay(tp_idx, fp_idx, seizures_idx)