import torch

import numpy as np
from scipy import interpolate

from knapsack import knapsack

def eval_metrics(y_pred, y_true):
    overlap = np.sum(y_pred * y_true); #print(overlap, np.sum(y_pred))
    precision = overlap / (np.sum(y_pred) + 1e-8) # 預測為1且真正為1
    recall = overlap / (np.sum(y_true) + 1e-8) # 真正為1且model預測為1
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
    pred_score = upsample_self(pred_score, N) # Use Nearest Neighbor to extend from 320 to N
    pred_score_key_frames = (pred_score>=0.5).astype(np.float) # convert to key frames
    value = np.array([pred_score_key_frames[cp[0]:(cp[1]+1)].mean() for cp in cps]) # [n_segments]
    _, selected = knapsack(value, weight, int(0.15*N)) # selected -> [66, 64, 51, 50, 44, 41, 40, 38, 34, 33, 31, 25, 24, 23, 20, 10, 9]
    selected = selected[::-1] # inverse the selected list, which seg is selected
    key_shots = np.zeros(shape=(N, ))
    for i in selected:
        key_shots[cps[i][0]:(cps[i][1]+1)] = 1 # assign 1 to seg
        #print("assign!!!")
    # N: total video length
    # pred_score: shape [N]
    # selected: which seg is selected
    # key_shots: assign 1 to selected seg
    return N, pred_score.tolist(), selected, key_shots

def upsample_self(pred_score, N):
    """
    Use Nearest Neighbor to extend from 320 to N
    input: 
        pred_score: shape [320], indicates key frame prob.
        N: scalar, video original length
    output
        up_arr: shape [N]
    """
    x = np.linspace(0, len(pred_score)-1, len(pred_score))
    f = interpolate.interp1d(x, pred_score, kind='nearest')
    x_new = np.linspace(0, len(pred_score)-1, N); #print(x_new, N)
    up_arr = f(x_new)

    return up_arr


def upsample(down_arr, N):
    """
    Use Nearest Neighbor to extend from 320 to N
    input: 
        down_arr: shape [320], indicates key frame prob.
        N: scalar, video original length
    output
        up_arr: shape [N]
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


