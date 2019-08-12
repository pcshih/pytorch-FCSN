import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
#from tensorboardX import SummaryWriter

from data_loader import get_loader
from FCSN import *
import eval_tools

# configure training record
#writer = SummaryWriter()
# load training and testing dataset
train_loader_list,test_dataset_list,data_file = get_loader("datasets/fcsn_tvsum.h5", "1D", 5)
# device use for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# number of epoch to train
EPOCHS = 100
# array for calc. eval fscore
fscore_arr = np.zeros(len(train_loader_list))

# ref: https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        #init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        #init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


for i in range(len(train_loader_list)):
    # model declaration
    model = FCSN_1D_unsup()
    # optimizer declaration
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # apply init weights(bad performance)
    #model.apply(weights_init)
    # switch to train mode
    model.train()
    # put model in to device
    model.to(device)
    for epoch in range(EPOCHS):
        for batch_i, (feature,label,_) in enumerate(train_loader_list[i]):
            feature = feature.to(device) #[5,1024,320]


            # zero the parameter gradients
            optimizer.zero_grad()

            outputs_reconstruct,mask,_ = model(feature) # output shape [5,1024,320],[5,1,320]

            # reconst. loss改成分批再做平均
            feature_select = feature*mask # [5,1024,320]
            outputs_reconstruct_select = outputs_reconstruct*mask # [5,1024,320]
            feature_diff_1 = torch.sum((feature_select-outputs_reconstruct_select)**2, dim=1) # [5,320]
            feature_diff_1 = torch.sum(feature_diff_1, dim=1) # [5]

            mask_sum = torch.sum(mask, dim=2) # [5,1]
            mask_sum = torch.sum(mask_sum, dim=1) # [5]

            reconstruct_loss = torch.mean(feature_diff_1/mask_sum)
            

            # diversity loss
            batch_size, feat_size, frames = outputs_reconstruct.shape

            outputs_reconstruct_norm = torch.norm(outputs_reconstruct, p=2, dim=1, keepdim=True)

            normalized_outputs_reconstruct = outputs_reconstruct/outputs_reconstruct_norm # [5,1024,320]

            normalized_outputs_reconstruct_reshape = normalized_outputs_reconstruct.permute(0, 2, 1) # [5,320,1024]

            similarity_matrix = torch.bmm(normalized_outputs_reconstruct_reshape, normalized_outputs_reconstruct) # [5, 320, 320]

            mask_trans = mask.permute(0,2,1) # [5,320,1]
            mask_matrix = torch.bmm(mask_trans, mask) # [5,320,320] 
            # filter out non key
            similarity_matrix_filtered = similarity_matrix*mask_matrix # [5,320,320]

            diversity_loss = 0
            acc_batch_size = 0
            for j in range(batch_size):
                batch_similarity_matrix_filtered = similarity_matrix_filtered[j,:,:] # [320,320]
                batch_mask = mask[j,:,:] # [1, 320]
                if batch_mask.sum() < 2:
                    #print("select less than 2 frames", batch_mask.sum())
                    batch_diversity_loss = 0
                else:
                    batch_diversity_loss = (batch_similarity_matrix_filtered.sum()-batch_similarity_matrix_filtered.trace())/(batch_mask.sum()*(batch_mask.sum()-1))
                    acc_batch_size += 1

                diversity_loss += batch_diversity_loss

            if acc_batch_size>0:
                diversity_loss /= acc_batch_size
                #print(acc_batch_size)
            else:
                diversity_loss = 0

            total_loss = reconstruct_loss + diversity_loss
            total_loss.backward()
            optimizer.step()

        # eval every 5 epoch
        if(epoch+1)%5 == 0:
            model.eval()
            eval_res_avg = [] # for all testing video results
            for feature,label,index in test_dataset_list[i]: # index has been +1 in dataloader.py
                feature = feature.view(1,1024,-1).to(device) # [1024,320] -> [1,1024,320]
                # we only want key frame prob. -> [1]
                pred_score = model(feature)[1].view(320) # [1,1,320] -> [320]

                video_name = "video_{}".format(index)
                video_info = data_file[video_name]
                # select key shots by video_info and pred_score
                # pred_summary: [N]
                _, _, pred_summary = eval_tools.select_keyshots(video_info, pred_score)
                true_summary_arr = video_info['user_summary'][()] # shape (n_users,N), summary from some users, each row is a binary vector
                eval_res = [eval_tools.eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr] # shape [n_user,3],3 for[precision, recall, fscore]
                eval_res = np.mean(eval_res, axis=0).tolist()  # for tvsum
                #eval_res = np.max(eval_res, axis=0).tolist()    # for summe
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