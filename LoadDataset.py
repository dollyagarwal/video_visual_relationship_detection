import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd
import os

class VideoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file,encoding='utf-8')
        self.transform = transform 

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video """
        video = self.dataframe.iloc[index].path.replace(u'\xa0', ' ')
        object1 = self.dataframe.iloc[index].object1
        relation = self.dataframe.iloc[index].relation
        object2 = self.dataframe.iloc[index].object2
        #label = list(object1,relation,object2)
        if self.transform:
            video = self.transform(video)
        return video,object1,relation,object2


class VideoFolderPathToTensor(object):
    """ load video at given folder path to torch.Tensor (C x L x H x W) 
        It can be composed with torchvision.transforms.Compose().
        
    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames. 
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, max_len=None, padding_mode=None):
        self.max_len = max_len
        assert padding_mode in (None, 'zero', 'last')
        self.padding_mode = padding_mode

    def __call__(self, path):
        """
        Args:
            path (str): path of video folder.
            
        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """
        # get video properity
        frames_path = sorted([os.path.join(path,f) for f in os.listdir(path) if (os.path.exists(os.path.join(path, f)) and os.path.isfile(os.path.join(path, f)))])
        frame = cv2.imread(frames_path[0])
        height, width, channels = frame.shape
        num_frames = len(frames_path)
        
        # init empty output frames (C x L x H x W)
        time_len = None
        if self.max_len:
            # time length has upper bound
            if self.padding_mode:
                # padding all video to the same time length
                time_len = self.max_len
            else:
                # video have variable time length
                time_len = min(num_frames, self.max_len)
        else:
            # time length is unlimited
            time_len = num_frames

 #       frames = torch.FloatTensor(channels, time_len, height, width)
        frames = torch.FloatTensor(time_len, channels, 224, 224)

        data_transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
      
              # load the video to tensor
        for index in range(time_len):
            if index < num_frames:
                # frame exists
                # read frame
                frame = cv2.imread(frames_path[index])
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # (H x W x C) to (C x H x W)
                frame = frame.permute(2, 0, 1).type('torch.DoubleTensor')
                frame=data_transform(frame)
                #frames[:, index, :, :] = frame.float()

                frames[index,:, :, :] = frame.float()
            else:
                # reach the end of the video
                if self.padding_mode == 'zero':
                    # fill the rest frames with 0.0
                    frames[index:,:,  :, :] = 0
                elif self.padding_mode == 'last':
                    # fill the rest frames with the last frame
                    assert(index > 0)
                    frames[index:,:,  :, :] = frames[index-1,:,  :, :].view(1,channels,  height, width)
                break

        frames /= 255
        return frames