import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data_loader import get_loader
from FCSN import *
import eval_tools

# load training and testing dataset
train_loader,test_dataset,data_file = get_loader("datasets/fcsn_tvsum.h5", "1D", 5)
# device use for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# number of epoch to train
EPOCHS = 50
# model declaration
model = FCSN_1D()
# optimizer declaration
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# switch to train mode
model.train()
# put model in to device
model.to(device)
# configure training record
writer = SummaryWriter()


for epoch in range(EPOCHS):
    for batch_i, (feature,label,_) in enumerate(train_loader):
        feature = feature.to(device) #[5,1024,320]
        label = label.to(device) #[5,320]

        # reshape
        label = label.view(-1) #[5*320] every element indicates non-key or key (0 or 1)

        # loss criterion
        label_0 = 0.5*label.shape[0]/(label.shape[0]-label.sum())
        label_1 = 0.5*label.shape[0]/label.sum()
        weights = torch.tensor([label_0,label_1], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(feature) # output shape [5,2,320]

        # reshape output
        outputs = outputs.permute(0,2,1).contiguous().view(-1,2) #[5*320,2] 2 choices prob., non-key or key

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

    # eval every 5 epoch
    if(epoch+1)%5 == 0:
        model.eval()
        eval_res_avg = [] # for all testing video results
        for feature,label,index in test_dataset: # index has been +1 in dataloader.py
            feature = feature.view(1,1024,-1).to(device) # [1024,320] -> [1,1024,320]
            pred_score = model(feature).view(-1,320) # [1,2,320] -> [2,320]
            # we only want key frame prob. -> [1]
            pred_score = torch.softmax(pred_score, dim=0)[1] # [320]
            

            video_name = "video_{}".format(index)
            video_info = data_file[video_name]
            # select key shots by video_info and pred_score
            # pred_summary: [N]
            _, _, pred_summary = eval_tools.select_keyshots(video_info, pred_score)
            true_summary_arr = video_info['user_summary'][()] # shape (20,N), summary from 20 users, each row is a binary vector
            eval_res = [eval_tools.eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr] # shape [20,3] 20 for users,3 for[precision, recall, fscore]
            eval_res = np.mean(eval_res, axis=0).tolist()
            eval_res_avg.append(eval_res) # [[precision1, recall1, fscore1], [precision2, recall2, fscore2]......]

        eval_res_avg = np.mean(eval_res_avg, axis=0).tolist()
        precision = eval_res_avg[0]
        recall = eval_res_avg[1]
        fscore = eval_res_avg[2]
        print("epoch:{:0>3d} precision:{:.1%} recall:{:.1%} fscore:{:.1%}".format(epoch, precision, recall, fscore))

        writer.add_scalar("eval/precision", precision, epoch, time.time())   # tag, Y, X -> 當Y只有一個時
        writer.add_scalar("eval/recall", recall, epoch, time.time())
        writer.add_scalar("eval/fscore", fscore, epoch, time.time())

        model.train()




        
























def peek_dataset():
    dataset_filename = "datasets/fcsn_tvsum.h5"
    dataset = h5py.File(dataset_filename, 'r')

    num_videos = len(dataset.keys()); print("num of videos in this dataset:", num_videos)
    video_name_list = list(dataset.keys()); print("video name list: (below)", video_name_list,sep='\n')
    
    video_want_to_peek = "video_1"; print("we want to peek", video_want_to_peek, "information")

    information_in_each_video = list(dataset[video_want_to_peek].keys()); print("information in each video: (below)", information_in_each_video, sep='\n')
    
    # # ['change_points', 'features', 'gtscore', 'gtsummary', 'n_frame_per_seg', 'n_frames', 'n_steps', 'picks', 'user_summary', 'video_name']
    # # video_name: 這個影片真實的的名稱
    # #video_name = dataset[video_want_to_peek]["video_name"][...]; print(video_want_to_peek, "true video_name:", video_name)
    # # n_frames: 這個影片總共有幾個frames
    # n_frames = dataset[video_want_to_peek]["n_frames"][...]; print(n_frames, "frames in", video_want_to_peek)
    # # picks: 從原始影片那些indices挑的frame，每15張挑第1張來subsample
    # picks = dataset[video_want_to_peek]["picks"][...]; print("selected indices: (below)", picks, sep="\n")
    # # n_step: 這個影片經過subsample後剩下多少frame
    # n_steps =  dataset[video_want_to_peek]["n_steps"][...]; print("#frames after subsampled", n_steps)
    # # features: 這個影片經過subsample後經過googlenet，pool5後的特徵，這裡顯示其shape
    # features = dataset[video_want_to_peek]["features"][...]; print("googlenet pool5 features' shape in", video_want_to_peek, features.shape)
    # # change_points: 將影片分成若干個區間，每個區間的範圍(原始影片的indices)
    # change_points = dataset[video_want_to_peek]["change_points"][...]; print("change_points: (below)", change_points, change_points.shape, sep="\n")
    # # n_frame_per_seg: 每個區間裡面有幾個frame
    # n_frame_per_seg = dataset[video_want_to_peek]["n_frame_per_seg"][...]; print("n_frame_per_seg: (below)", n_frame_per_seg, n_frame_per_seg.shape, sep='\n')
    # # user_summary: 若干個人對於原影片覺得那些是key那些不是，binary vector
    # user_summary = dataset[video_want_to_peek]["user_summary"][...]; print("user_summary: (below)", user_summary, user_summary.shape, sep="\n")
    # # gtsummary: subsampled的影片的每個frame那些是key那些不是，binary vector
    # gtsummary = dataset[video_want_to_peek]["gtsummary"][...]; print("gtsummary: (below)", gtsummary, gtsummary.shape, sep="\n")
    # # gtscore: subsampled的影片的每個frame每個frame是key的分數
    # gtscore = dataset[video_want_to_peek]["gtscore"][...]; print("gtscore: (below)", gtscore, gtscore.shape, sep="\n")
    

    

