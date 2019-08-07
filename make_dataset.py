from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image
from pathlib import Path
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import pdb

import eval_tools


parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, help='directory containing mp4 file of specified dataset.', default='/media/data/PTec131b/VideoSum/SumMe/video')
parser.add_argument('--h5_path', type=str, help='save path of the generated dataset, which should be a hdf5 file.', default='datasets/fcsn_summe.h5')
parser.add_argument('--vsumm_data', type=str, help='preprocessed dataset path from this repo: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce, which should be a hdf5 file. We copy cps and some other info from it.', default='/media/data/PTec131b/VideoSum/SumMe/eccv16_dataset_summe_google_pool5.h5')

args = parser.parse_args()
video_dir = args.video_dir # directory containing *.mp4 file
h5_path = args.h5_path # where to save the processed dataset
vsumm_data = h5py.File(args.vsumm_data) # copy cps and other info from it

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # HWC->CHW [0,255]->[0.0,1.0]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


net = models.googlenet(pretrained=True)
# we only want features with no grads
for param in net.parameters():
    param.requires_grad = False
net.to(device) # to GPU or CPU
fea_net = nn.Sequential(*list(net.children())[:-2]) # pool5 feature
fea_net.eval() # eval mode

def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8)
        recall = overlap / (true_sum + 1e-8)
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)

def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape    # [n_user,n_frame] i.e.[15,4494] for this video

    overlap_arr = np.zeros(n_user)
    oracle_summary = np.zeros(n_frame)
    
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1) # [n_user], accumulate each user key frame amount
    priority_idx = np.argsort(-user_summary.sum(axis=0)) # [n_frame], most select frame -> least select frame(because of "-" sign)
    best_fscore = 0
    for idx in priority_idx: # loop over the most select frames e.g. [5, 7, 11, 1 ......]
        oracle_sum += 1
        for usr_i in range(n_user): # loop1: 15個人對於最多人選的frame，統計有沒有選
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break
    # tqdm.write('Overlap: '+str(overlap_arr))
    # tqdm.write('True summary n_key: '+str(true_sum_arr))
    # tqdm.write('Oracle smmary n_key: '+str(oracle_sum))
    # tqdm.write('Final F-score: '+str(best_fscore))
    #print("weirme", best_fscore)
    return oracle_summary

# trial
def get_oracle_summary_self(user_summary): 
    """
    create a single ground-truth set of keyframes from multiple user-annotated ones for each video.
    Args:
        user_summary: [n_user,n_frame]
    """
    n_user, n_frame = user_summary.shape    # [n_user,n_frame] e.g.[15,4494] for video

    oracle_summary = np.zeros(n_frame)  # single ground-truth

    priority_idx = np.argsort(-(user_summary.sum(axis=0))) # [n_frame], the most select frame -> the least select frame(because of "-" sign)

    best_fscore = 0
    for idx in priority_idx:
        oracle_summary[idx] = 1 # assume fake sum
        n_user_fscore = np.zeros(n_user) # record of every user to fake sum
        for usr_i in range(n_user):
            y_pred = user_summary[usr_i]
            y_true = oracle_summary
            fscore = eval_tools.eval_metrics(y_pred,y_true)[2] # only get fscore
            n_user_fscore[usr_i] = fscore

        cur_fscore = n_user_fscore.() # avg. of each sum to fake sum
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
        else:
            oracle_summary[idx] = 0 # assume sum is not real

    #print("pcshih", best_fscore)
    
    return oracle_summary

def video2fea(video_path, h5_f):
    video = cv2.VideoCapture(video_path)
    idx = video_path.split('.')[0].split('/')[-1] # split('.') -> ['/media/data/PTec131b/VideoSum/SumMe/video/12', 'mp4']
    #tqdm.write('Processing video '+idx)
    #print("Processing video {}".format(idx))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = length//320 # no rounding, pick once every ratio frames
    fea = []
    label = []
    usr_sum_arr = vsumm_data['video_'+idx]['user_summary'][...]  # i.e. [15, 4494] binary vector
    cps = vsumm_data['video_'+idx]['change_points'][...] # [n_segs, 2]
    n_frame_per_seg = vsumm_data['video_'+idx]['n_frame_per_seg'][...] #[n_segs]
    usr_sum = get_oracle_summary(usr_sum_arr) # i.e. [15,4494] -> [4494] binary vector, get authentic key frame indication
    usr_sum = get_oracle_summary_self(usr_sum_arr) # i.e. [15,4494] -> [4494] binary vector, get authentic key frame indication
    i = 0
    success, frame = video.read()
    while success:
        if (i+1) % ratio == 0:
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_RGB_PIL = Image.fromarray(frame_RGB)
            frame_RGB_PIL_transform = transform(frame_RGB_PIL).to(device).view(1,3,224,224) # [3,224,224] -> [1,3,224,224]
            fea_out = fea_net(frame_RGB_PIL_transform).permute(0,3,2,1).contiguous().view(1024).detach().to("cpu")
            fea.append(fea_out)
            label.append(usr_sum[i])
            #try:
            #    label.append(usr_sum[i])
            #except:
            #    pdb.set_trace()
        i += 1
        success, frame = video.read()
    fea = torch.stack(fea, 0) # in order to change to tensor
    fea = fea[:320] # we only want 320 frames
    label = label[:320] # for training
    v_data = h5_f.create_group('video_'+idx)
    v_data['feature'] = fea.numpy() # convert to numpy 
    v_data['label'] = label
    v_data['length'] = len(usr_sum) # total video length
    v_data['change_points'] = cps
    v_data['n_frame_per_seg'] = n_frame_per_seg
    v_data['picks'] = [ratio*i for i in range(320)] # which index pick to subsample
    v_data['user_summary'] = usr_sum_arr
    #if fea.shape[0] != 320 or len(label) != 320:
    #    print('error in video ', idx, feashape[0], len(label))


def make_dataset(video_dir, h5_path):
    video_dir = Path(video_dir).resolve() # resolve: Make the path absolute
    video_list = list(video_dir.glob('*.mp4')) # glob: find the specific format files, need to add list
    video_list.sort() # sort the *.mp4 file name
    with h5py.File(h5_path, 'w') as h5_f: # saving dataset file pointer
        #for video_path in tqdm(video_list, desc='Video', ncols=80, leave=False):
        #for video_path in video_list:
        for video_path in tqdm(video_list):
            #print(video_path)  # i.e. /media/data/PTec131b/VideoSum/SumMe/video/1.mp4
            str_video_path = str(video_path)
            video2fea(str_video_path, h5_f)
    
    
if __name__ == '__main__':
    make_dataset(video_dir, h5_path)
    vsumm_data.close()