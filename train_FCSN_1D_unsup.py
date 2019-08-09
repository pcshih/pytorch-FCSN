import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
#from tensorboardX import SummaryWriter

from data_loader import get_loader
from FCSN import *
import eval_tools

# configure training record
#writer = SummaryWriter()
# load training and testing dataset
train_loader_list,test_dataset_list,data_file = get_loader("datasets/fcsn_summe.h5", "2D", 5)
# device use for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# number of epoch to train
EPOCHS = 100
# array for calc. eval fscore
fscore_arr = np.zeros(len(train_loader_list))


for i in range(len(train_loader_list)):
    # model declaration
    model = FCSN_1D_unsup()
    # optimizer declaration
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # switch to train mode
    model.train()
    # put model in to device
    model.to(device)
    for epoch in range(EPOCHS):
        for batch_i, (feature,label,_) in enumerate(train_loader_list[i]):
            feature = feature.to(device) #[5,1024,320]


            # zero the parameter gradients
            optimizer.zero_grad()

            outputs_select_reconstruct,soft_argmax = model(feature) # output shape [5,1024,320],[5,1,320]

            # reconst. loss
            feature_select = feature*soft_argmax # [5,1024,320]
            reconstruct_loss = torch.sum((feature_select-outputs_select_reconstruct)**2) / torch.sum(soft_argmax)

            # diversity loss
            # S_K_summary_reshape = S_K_summary.view(S_K_summary.shape[1], S_K_summary.shape[3])
            # norm_div = torch.norm(S_K_summary_reshape, 2, 0, True); #print(norm_div)
            # S_K_summary_reshape = S_K_summary_reshape/norm_div; #print(select_data)
            # loss_matrix = S_K_summary_reshape.transpose(1, 0).mm(S_K_summary_reshape); #print(loss_matrix)
            # diversity_loss = loss_matrix.sum() - loss_matrix.trace()
            # diversity_loss = diversity_loss/len(column_mask)/(len(column_mask)-1)

            total_loss = reconstruct_loss + diversity_loss
            total_loss.backward()
            optimizer.step()

        # eval every 5 epoch
        if(epoch+1)%5 == 0:
            model.eval()
            eval_res_avg = [] # for all testing video results
            for feature,label,index in test_dataset_list[i]: # index has been +1 in dataloader.py
                feature = feature.view(1,1024,-1).to(device) # [1024,320] -> [1,1024,320]
                pred_score = model(feature).view(-1,320) # [1,2,320] -> [2,320]
                # we only want key frame prob. -> [1]
                pred_score = torch.softmax(pred_score, dim=0)[1] # [320]
                

                video_name = "video_{}".format(index)
                video_info = data_file[video_name]
                # select key shots by video_info and pred_score
                # pred_summary: [N]
                _, _, pred_summary = eval_tools.select_keyshots(video_info, pred_score)
                true_summary_arr = video_info['user_summary'][()] # shape (n_users,N), summary from some users, each row is a binary vector
                eval_res = [eval_tools.eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr] # shape [n_user,3],3 for[precision, recall, fscore]
                #eval_res = np.mean(eval_res, axis=0).tolist()  # for tvsum
                eval_res = np.max(eval_res, axis=0).tolist()    # for summe
                eval_res_avg.append(eval_res) # [[precision1, recall1, fscore1], [precision2, recall2, fscore2]......]
                
            eval_res_avg = np.mean(eval_res_avg, axis=0).tolist()
            precision = eval_res_avg[0]
            recall = eval_res_avg[1]
            fscore = eval_res_avg[2]
            #print("split:{} epoch:{:0>3d} precision:{:.1%} recall:{:.1%} fscore:{:.1%}".format(i, epoch, precision, recall, fscore))

            model.train()

            # store the last fscore for eval, and remove model from GPU
            if((epoch+1)==EPOCHS):
                # store fscore
                fscore_arr[i] = fscore
                print("split:{} epoch:{:0>3d} precision:{:.1%} recall:{:.1%} fscore:{:.1%}".format(i, epoch, precision, recall, fscore))
                # release model from GPU
                model = model.cpu()
                torch.cuda.empty_cache()

            #writer.add_scalar("eval_1D_X_epoch/precision", precision, epoch, time.time())   # tag, Y, X -> 當Y只有一個時
            #writer.add_scalar("eval_1D_X_epoch/recall", recall, epoch, time.time())
            #writer.add_scalar("eval_1D_X_epoch/fscore", fscore, epoch, time.time())

            

# print eval fscore
print("average fscore:{:.1%}".format(fscore_arr.mean()))