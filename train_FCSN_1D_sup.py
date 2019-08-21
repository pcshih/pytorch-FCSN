import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data_loader import get_loader
from FCSN import *
import eval_tools

# configure training record
#writer = SummaryWriter()
# batch_size
BATCH_SIZE = 5
# load training and testing dataset
train_loader_list,test_dataset_list,data_file = get_loader("datasets/fcsn_summe.h5", "1D", BATCH_SIZE)
# device use for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# number of epoch to train
EPOCHS = 100
# array for calc. eval fscore
fscore_arr = np.zeros(len(train_loader_list))



for i in range(len(train_loader_list)):
    # model declaration
    model = FCSN_1D_sup()
    # optimizer declaration
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # switch to train mode
    model.train()
    # put model in to device
    model.to(device)


    for epoch in range(EPOCHS):
        for batch_i, (feature,label,_) in enumerate(train_loader_list[i]):
            feature = feature.to(device) #[5,1024,320]
            label = label.to(device) #[5,320]
            outputs = model(feature) # output shape [5,2,320]

            
            # zero the parameter gradients
            optimizer.zero_grad()

            # cross entropy loss
            classification_loss_sum = 0
            # for one movie each
            for k in range(BATCH_SIZE):
                label_each = label[k,:] #[320]
                outputs_each = outputs[k,:,:].permute(1,0) #[2,320]->[320,2]

                # loss criterion
                label_each_float = label_each.type(torch.float)
                label_0 = 0.5*label_each_float.shape[0]/(label_each_float.shape[0]-label_each_float.sum())
                label_1 = 0.5*label_each_float.shape[0]/label_each_float.sum()
                weights = torch.tensor([label_0,label_1], dtype=torch.float, device=device)
                criterion = nn.CrossEntropyLoss(weight=weights)

                classification_loss_sum += criterion(outputs_each, label_each)

            classification_loss = classification_loss_sum/BATCH_SIZE
            
            # variance loss
            # score = torch.softmax(outputs, dim=1) #[5,2,320]
            # eps = 1e-8
            # med_score,_ = torch.median(score, dim=1, keepdim=True) #[5,1]
            # batch_variance = torch.mean((score-med_score)**2, dim=1) #[5]
            # batch_variance_loss = 1.0/(batch_variance+eps)
            # variance_loss = torch.mean(batch_variance_loss); #print(variance_loss)

            # total loss
            total_loss = classification_loss
            #total_loss = classification_loss+0.005*variance_loss; #print(0.005*variance_loss)

            # only plot split:0 loss
            #if i==0:
                # tvsum: 1 epoch has 8 iter.
                #iteration = len(train_loader_list[i])*epoch+batch_i; #print(len(train_loader_list[i]), epoch, batch_i,iteration)
                #writer.add_scalar("sup_tvsum_1D/split_0_loss_classification", classification_loss, iteration, time.time())   # tag, Y, X -> 當Y只有一個時
                #writer.add_scalar("sup_loss_1D_X_batch/unscaled_variance_loss", variance_loss, iteration, time.time())   # tag, Y, X -> 當Y只有一個時
                #writer.add_scalar("sup_loss_1D_X_batch/total_loss_scaled", total_loss, iteration, time.time())   # tag, Y, X -> 當Y只有一個時
            
            total_loss.backward()
            optimizer.step()

        # eval every 5 epoch
        if(epoch+1)%5 == 0:
            #print(i, epoch, total_loss)
            model.eval()
            eval_res_avg = [] # for all testing video results
            for j, (feature,label,index) in enumerate(test_dataset_list[i], 1): # index has been +1 in dataloader.py
                feature = feature.view(1,1024,-1).to(device) # [1024,320] -> [1,1024,320]
                pred_score = model(feature).view(-1,320) # [1,2,320] -> [2,320]
                # we only want key frame prob. -> [1]
                pred_score = torch.softmax(pred_score, dim=0)[1] # [320]
                
                #print("epoch:{:0>3d} video_index:{:0>2d} key_frame_score(first20):{}".format(epoch, index, pred_score[:20]))
                
                video_name = "video_{}".format(index)
                video_info = data_file[video_name]
                # select key shots by video_info and pred_score
                # pred_summary: [N] binary keyshot
                N, pred_score_upsample, _, pred_summary = eval_tools.select_keyshots(video_info, pred_score)
                true_summary_arr = video_info['user_summary'][()] # shape (n_users,N), summary from some users, each row is a binary vector
                eval_res = [eval_tools.eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr] # shape [n_user,3],3 for[precision, recall, fscore]
                #eval_res = np.mean(eval_res, axis=0).tolist()  # for tvsum
                eval_res = np.max(eval_res, axis=0).tolist()    # for summe
                eval_res_avg.append(eval_res) # [[precision1, recall1, fscore1], [precision2, recall2, fscore2]......]

                # plot split 0 and first test histogram only save epoch4 and epoch last
                # if (i==0 and j==1 and (epoch==4 or epoch==EPOCHS-1)):
                #     fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(100, 60))
                #     ax[0].bar(list(range(320)), pred_score.tolist(), alpha=1.0, width=0.6)
                #     ax[0].set_xlabel('frame_index', fontsize=50)  
                #     ax[0].set_ylabel('Probability', fontsize=50)
                #     ax[0].tick_params(axis='both', which='major', labelsize=50)
                #     ax[1].bar(list(range(N)), pred_score_upsample, alpha=0.5, width=0.6) 
                #     ax[1].bar(list(range(N)), pred_summary, alpha=0.8, width=0.6) 
                #     ax[1].set_xlabel('frame_index', fontsize=50)  
                #     ax[1].set_ylabel('Probability', fontsize=50)
                #     ax[1].tick_params(axis='both', which='major', labelsize=50)
                #     fig.savefig("imgs/summe_1D_split{}_epoch{:0>3d}_video{}.png".format(i, epoch, index))
                #     plt.close() # close a current window


                
            eval_res_avg = np.mean(eval_res_avg, axis=0).tolist()
            precision = eval_res_avg[0]
            recall = eval_res_avg[1]
            fscore = eval_res_avg[2]
            print("split:{} epoch:{:0>3d} precision:{:.1%} recall:{:.1%} fscore:{:.1%} loss:{}".format(i, epoch, precision, recall, fscore, total_loss))

            model.train()

            # store the last fscore for eval, and remove model from GPU
            if((epoch+1)==EPOCHS):
                # store fscore
                fscore_arr[i] = fscore
                #print("split:{} epoch:{:0>3d} precision:{:.1%} recall:{:.1%} fscore:{:.1%}".format(i, epoch, precision, recall, fscore))
                # release model from GPU
                model = model.cpu()
                torch.cuda.empty_cache()

            #writer.add_scalar("eval_1D_X_epoch/precision", precision, epoch, time.time())   # tag, Y, X -> 當Y只有一個時
            #writer.add_scalar("eval_1D_X_epoch/recall", recall, epoch, time.time())
            #writer.add_scalar("eval_1D_X_epoch/fscore", fscore, epoch, time.time())

            

# print eval fscore
print("summe average fscore:{:.1%}".format(fscore_arr.mean()))