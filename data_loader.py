# ref: https://github.com/weirme/Video_Summary_using_FCSN/blob/master/data_loader.py
import torch
import torch.utils.data

import h5py

# ref: https://blog.csdn.net/yuan_j_y/article/details/9317817
class VideoData(object):
    """Dataset class"""
    def __init__(self, data_path, dim):
        self.data_file = h5py.File(data_path) # load *.h5 file
        self.dim = dim # for FCSN use 1D or 2D conv

    def __len__(self):
        return len(self.data_file.keys()) # a number of keys in [video_1, video_11 ......]

    def __getitem__(self, index):
        index = index+1 # because the video is start from 1
        video_name = "video_{}".format(index)
        video = self.data_file[video_name] # get specific video

        if self.dim=="1D": # for FCSN_1D
            video_feature = torch.from_numpy(video["feature"][...]).transpose(1,0).view(1024,-1) # [320,1024] -> [1024,320]
        else:
            video_feature = torch.from_numpy(video["feature"][...]).transpose(1,0).view(1,1024,1,-1) #[320,1024] -> [1,1024,1,320]

        video_label = torch.from_numpy(video["label"][...]).type(torch.long) # [320,]
        
        return video_feature,video_label,index

def get_loader(path, dim, batch_size=5):
    """
    dim=1D prepare for FCSN_1D
    dim=2D prepare for FCSN_2D
    """
    dataset = VideoData(path, dim)

    # train_dataset: [(video_index1_feature,video_index1_label,index1)...]
    train_dataset,test_dataset = torch.utils.data.random_split(dataset, [int(dataset.__len__()*0.8), int(dataset.__len__()*0.2)])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)

    # train_loader: batch [5,1024,320], [5,320], [5,] -> feature, label, index
    # test_dataset: [(video_index1_feature,video_index1_label,index1), (video_index11_feature,video_index11_label,index11)...]
    # dataset.data_file is the whole *.h5 file
    return train_loader,test_dataset,dataset.data_file

if __name__ == "__main__":
    train_loader,test_dataset,data_file = get_loader("datasets/fcsn_tvsum.h5", "1D", 5)

