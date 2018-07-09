import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""
    
    def __init__(self, crop, rgb=False):
        assert isinstance(crop, int)
        self.crop = crop
        self.rgb = rgb

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if not self.rgb:
            # convert image to grayscale
            image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        s = self.crop / 2
        key_pts_copy = (key_pts_copy - s)/s


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, random_flip=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top_max = min(max(key_pts[:,1].max() - new_h, 0), h - new_h - 1)
        left_max = min(max(key_pts[:,0].max() - new_w, 0), w - new_w - 1)
        top = np.random.randint(top_max, h - new_h)
        left = np.random.randint(left_max, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

# From: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
class RandomFlip(object):
    """Randomly flip image and keypoints to match"""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        if np.random.choice((True, False)):
            image = self.flip(image)
            key_pts = self.flip(key_pts, is_label=True)
            
        return {'image': image, 'keypoints': key_pts}   
    
    def shuffle_lr(self, parts, pairs=None):
        if pairs is None:
            pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10],
                     [7, 9], [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45],
                     [37, 44], [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34],
                     [50, 52], [49, 53], [48, 54], [61, 63], [60, 64], [67, 65], [59, 55], [58, 56]]
        for matched_p in pairs:
            idx1, idx2 = matched_p[0], matched_p[1]
            tmp = np.copy(parts[idx1])
            np.copyto(parts[idx1], parts[idx2])
            np.copyto(parts[idx2], tmp)
        return parts


    def flip(self, tensor, is_label=False):
        was_cuda = False
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        elif isinstance(tensor, torch.cuda.FloatTensor):
            tensor = tensor.cpu().numpy()
            was_cuda = True

        was_squeezed = False
        if tensor.ndim == 4:
            tensor = np.squeeze(tensor)
            was_squeezed = True
        if is_label:
            #tensor = tensor.swapaxes(0, 1).swapaxes(1, 2)
            #tensor = cv2.flip(self.shuffle_lr(tensor), 0).reshape(tensor.shape)
            tensor[:,0] = tensor[:,0] * -1
            tensor = self.shuffle_lr(tensor)
            #tensor = tensor.swapaxes(2, 1).swapaxes(1, 0)
        else:
            tensor = tensor.swapaxes(0, 1).swapaxes(1, 2)
            tensor = cv2.flip(tensor, 1).reshape(tensor.shape)
            tensor = tensor.swapaxes(2, 1).swapaxes(1, 0)
        if was_squeezed:
            tensor = np.expand_dims(tensor, axis=0)
        tensor = torch.from_numpy(tensor)
        if was_cuda:
            tensor = tensor.cuda()
        return tensor