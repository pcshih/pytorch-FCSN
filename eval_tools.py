import torch

import numpy as np

from knapsack import knapsack

def eval_metrics(y_pred, y_true):
    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return [precision, recall, fscore]


def select_keyshots(video_info, pred_score):
    """
    input:
        video_info: specific video of *.h5 file
        pred_score: [320] key frame score in every frames
    """
    N = video_info['length'][()] # scalar, video original length
    cps = video_info['change_points'][()] # shape [n_segments,2], stores begin and end of each segment in original length index
    weight = video_info['n_frame_per_seg'][()] # shape [n_segments], number of frames in each segment
    pred_score = pred_score.to("cpu").detach().numpy() # GPU->CPU, requires_grad=False, to numpy
    pred_score = upsample(pred_score, N) # Use Nearest Neighbor to extend from 320 to N
    pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps]) # [n_segments]
    _, selected = knapsack(pred_value, weight, int(0.15 * N)) # selected -> [66, 64, 51, 50, 44, 41, 40, 38, 34, 33, 31, 25, 24, 23, 20, 10, 9]
    selected = selected[::-1] # inverse the selected list, which seg is selected
    key_labels = np.zeros(shape=(N, ))
    for i in selected:
        key_labels[cps[i][0]:cps[i][1]] = 1 # assign 1 to seg
    # pred_score: shape [n_segments]
    # selected: which seg is selected
    # key_labels: assign 1 to selected seg
    return pred_score.tolist(), selected, key_labels.tolist()


def upsample(down_arr, N):
    """
    Use Nearest Neighbor to extend from 320 to N
    input: 
        down_arr: shape [320], indicates key frame prob.
        N: scalar, video original length
    """
    up_arr = np.zeros(N) # get N zeros
    ratio = N // 320 # no rounding. i.e. 5//2 = 2
    l = (N - ratio * 320) // 2
    i = 0
    while i < 320:
        up_arr[l:l+ratio] = np.ones(ratio, dtype=int) * down_arr[i]
        l += ratio
        i += 1
    return up_arr


if __name__ == "__main__":
    device = torch.device("cuda:0")

    import h5py
    data_file = h5py.File("datasets/fcsn_tvsum.h5")
    video_info = data_file["video_1"]
    pred_score = torch.randn((320,), requires_grad=True)
    pred_score = pred_score.to(device)
    select_keyshots(video_info, pred_score)


