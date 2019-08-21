from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image
from pathlib import Path
import cv2
import h5py
import hdf5storage
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import argparse
import pdb

import eval_tools

# parser init and init
parser = argparse.ArgumentParser()
# tvsum
parser.add_argument('--tvsum_videos_path', type=str, help='directory containing mp4 file of specified dataset.', 
default='/media/data/PTec131b/VideoSum/TVsum/video')
parser.add_argument('--tvsum_eccv16_data', type=str, help='preprocessed dataset path from this repo: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce, which should be a hdf5 file. We copy cps and some other info from it.', 
default='/media/data/PTec131b/VideoSum/TVsum/eccv16_dataset_tvsum_google_pool5.h5')
parser.add_argument('--tvsum_GT_path', type=str, help='GT frame-level scores dir from summe or tvsum', 
default='/media/data/PTec131b/VideoSum/TVsum/GT')
parser.add_argument('--tvsum_h5_data', type=str, help='save path of the generated dataset, which should be a hdf5 file.', 
default='datasets/fcsn_tvsum.h5')
# summe
parser.add_argument('--summe_videos_path', type=str, help='directory containing mp4 file of specified dataset.', 
default='/media/data/PTec131b/VideoSum/SumMe/video')
parser.add_argument('--summe_eccv16_data', type=str, help='preprocessed dataset path from this repo: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce, which should be a hdf5 file. We copy cps and some other info from it.', 
default='/media/data/PTec131b/VideoSum/SumMe/eccv16_dataset_summe_google_pool5.h5')
parser.add_argument('--summe_GT_path', type=str, help='GT frame-level scores dir from summe or tvsum', 
default='/media/data/PTec131b/VideoSum/SumMe/GT')
parser.add_argument('--summe_h5_data', type=str, help='save path of the generated dataset, which should be a hdf5 file.', 
default='datasets/fcsn_summe.h5')

# parse argument
args = parser.parse_args()
# tvsum
tvsum_videos_path = args.tvsum_videos_path # directory containing *.mp4 file
tvsum_eccv16_data = args.tvsum_eccv16_data # copy cps and other info from it
tvsum_GT_path = args.tvsum_GT_path # GT frame-level scores dir from summe or tvsum
tvsum_h5_data = args.tvsum_h5_data # where to save the processed dataset
# summe
summe_videos_path = args.summe_videos_path # directory containing *.mp4 file
summe_eccv16_data = args.summe_eccv16_data # copy cps and other info from it
summe_GT_path = args.summe_GT_path # GT frame-level scores dir from summe or tvsum
summe_h5_data = args.summe_h5_data # where to save the processed dataset


# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # HWC->CHW [0,255]->[0.0,1.0]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# googlenet preparation
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

        cur_fscore = n_user_fscore.mean() # avg. of each sum to fake sum
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
    #usr_sum = get_oracle_summary(usr_sum_arr) # i.e. [15,4494] -> [4494] binary vector, get authentic key frame indication
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
            video2feature(str_video_path)



def video2feature(video_path, idx, eccv16, gtscore):
    # for feature and label
    video = cv2.VideoCapture(video_path)
    
    # tvsum: video16,39 one more frame from *.mat file
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = length//320 # no rounding, pick once every ratio frames

    
    label = []
    # handle labels
    cps = eccv16['video_'+idx]['change_points'][()] # [n_segs, 2]
    weight = eccv16['video_'+idx]['n_frame_per_seg'][()] # [n_segs]
    n_frame = eccv16['video_'+idx]['n_frames'][()] # scalar
    gtscore = np.ravel(gtscore.transpose())
    value = np.array([gtscore[cp[0]:(cp[1]+1)].mean() for cp in cps])
    _, selected = eval_tools.knapsack(value, weight, int(0.15*length))
    selected = selected[::-1] # inverse the selected list, which seg is selected
    key_shots = np.zeros(shape=(n_frame, ))
    key_frames = np.zeros(shape=(n_frame, ))
    for i in selected:
        # key shot processing
        key_shots[cps[i][0]:(cps[i][1]+1)] = 1 # assign 1 to seg
        # key frame processing
        max_idx = np.where(gtscore[cps[i][0]:(cps[i][1]+1)] == np.max(gtscore[cps[i][0]:(cps[i][1]+1)]) )
        max_idx = max_idx[0] # because (array[0,1,2])

        key_frames_slice = key_frames[cps[i][0]:(cps[i][1]+1)]
        key_frames_slice[max_idx] = 1 # also inference key_frames

        #print(idx,i, cps[i], key_frames_slice, key_frames[:100], gtscore[:100], sep='\n')

    # handle feature
    fea = []
    i = 0
    success, frame = video.read()
    while success:
        if (i+1) % ratio == 0:
            frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_RGB_PIL = Image.fromarray(frame_RGB)
            frame_RGB_PIL_transform = transform(frame_RGB_PIL).to(device).view(1,3,224,224) # [3,224,224] -> [1,3,224,224]
            fea_out = fea_net(frame_RGB_PIL_transform).permute(0,3,2,1).contiguous().view(1024).detach().to("cpu")
            fea.append(fea_out)
            label.append(key_frames[i])
        i += 1
        success, frame = video.read()

    fea = torch.stack(fea, 0) # in order to change to tensor
    fea = fea[:320] # we only want 320 frames
    label = label[:320] # for training

    return fea,label,cps,weight,n_frame


def make_dataset_tvsum(videos_path, eccv16_data, GT_path, h5_data):
    # load eccv16_data
    eccv16 = h5py.File(eccv16_data)

    with h5py.File(h5_data, 'w') as h5_f: # saving dataset file pointer

        GT_path = Path(GT_path).resolve() # resolve: Make the path absolute
        GT_list = list(GT_path.glob('*.mat')) # glob: find the specific format files, need to add list
        for GT in GT_list:
            GT = str(GT)

            GT_mat = hdf5storage.loadmat(GT)
            GT_tvsum50 = GT_mat['tvsum50'] # ['video', 'category', 'title', 'length', 'nframes', 'user_anno', 'gt_score']
            for idx, GT_video in enumerate(GT_tvsum50[0], 1): # [0] because every thing is store in 2d array
                idx = str(idx)
                #eccv16_nframes = eccv16['video_'+str(idx)]['n_frames'][()]
                #print(GT_attr[4][0][0]) # test for consistency [4] for nframes
                video_path = "{}/{}.mp4".format(videos_path, GT_video[0][0][0]) # [0] for video, [0][0] for getting value, each value store in 2d array
                fea,label,cps,n_frame_per_seg,length = video2feature(video_path, idx, eccv16, GT_video[6])
                # user_summary -> convert multiple frame-level score to keyshot [20, length]
                usr_sum_arr = []
                user_anno = GT_video[5].transpose()
                n_users, n_frame = user_anno.shape # 20,length
                for user_score in user_anno:
                    value = np.array([user_score[cp[0]:(cp[1]+1)].mean() for cp in cps]) # [n_segments]
                    _, selected = eval_tools.knapsack(value, n_frame_per_seg, int(0.15*length))
                    selected = selected[::-1] # inverse the selected list, which seg is selected
                    key_shots = np.zeros(shape=(length, ))
                    for i in selected:
                        # key shot processing
                        key_shots[cps[i][0]:(cps[i][1]+1)] = 1 # assign 1 to seg
                    usr_sum_arr.append(key_shots)
                usr_sum_arr = np.asarray(usr_sum_arr) # convert to numpy array


                # save to dataset
                dataset = h5_f.create_group('video_'+idx)
                dataset['feature'] = fea.numpy() # convert to numpy [320,1024]
                dataset['label'] = label # [320]
                dataset['length'] = length # total video length
                dataset['change_points'] = cps # [n_segs, 2]
                dataset['n_frame_per_seg'] = n_frame_per_seg # [n_segs]
                dataset['user_summary'] = usr_sum_arr # [20,length]

                # print status
                print("{} -> {} is processed...".format(idx, GT_video[2][0][0]))
   
    eccv16.close()
    h5_f.close()


def make_dataset_summe(videos_path, eccv16_data, GT_path, h5_data):
    # load eccv16_data
    eccv16 = h5py.File(eccv16_data); #print(eccv16['video_1'].keys())

    with h5py.File(h5_data, 'w') as h5_f: # saving dataset file pointer
        for video_idx in eccv16.keys():
            video_name = eccv16[video_idx]['video_name'][()].decode("utf-8") # utf-8 bytes to string
            #print(video_idx, video_name)
            GT_data = "{}/{}.mat".format(GT_path, video_name)
            GT_mat = hdf5storage.loadmat(GT_data) # [all_userIDs, FPS, gt_score, nFrames, segments, user_score, video_duration]
            
            video_path = "{}/{}.mp4".format(videos_path, video_name)
            idx = video_idx.split('_')[-1]
            fea,label,cps,n_frame_per_seg,length = video2feature(video_path, idx, eccv16, GT_mat['gt_score'])
            # user_summary
            usr_sum_arr = (GT_mat['user_score'].transpose()>0).astype(np.int) # [15~18,length]

            # save to dataset
            dataset = h5_f.create_group('video_'+idx)
            dataset['feature'] = fea.numpy() # convert to numpy [320,1024]
            dataset['label'] = label # [320]
            dataset['length'] = length # total video length
            dataset['change_points'] = cps # [n_segs, 2]
            dataset['n_frame_per_seg'] = n_frame_per_seg # [n_segs]
            dataset['user_summary'] = usr_sum_arr # [15~18,length]

            print("{} -> {} is processed...".format(idx, video_name))
   
    eccv16.close()
    h5_f.close()



if __name__ == '__main__':
    # make_dataset for tvsum
    #make_dataset_tvsum(tvsum_videos_path, tvsum_eccv16_data, tvsum_GT_path, tvsum_h5_data)
    # make dataset for summe
    make_dataset_summe(summe_videos_path, summe_eccv16_data, summe_GT_path, summe_h5_data)
    