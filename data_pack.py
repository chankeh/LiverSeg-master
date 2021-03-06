import cv2 as cv
import numpy as np
import PIL.Image as Image
import os
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
            self, data_file, data_dir, transform_trn=None, transform_val=None
    ):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform_{trn, val} (callable, optional): Optional transform to be applied
                on a sample.
        """
        # data_file = [data_dir+x for x in data_file]
        # with open(data_file, 'rb') as f:
        #     datalist = f.readlines()
        # self.datalist = [(k, v) for k, v in map(lambda x: x.decode('utf-8').strip('\n').split('\t'), datalist)]
        self.datalist = data_file
        self.root_dir = data_dir # mask
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = 'train'

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = self.datalist[idx]
        msk_name = self.root_dir[idx]

        def read_image(x):
            img_arr = np.array(Image.open(x).convert('L'), 'f')
            img_arr = np.expand_dims(img_arr, 0)
            img_arr = np.expand_dims(img_arr, 1)
            # img_arr = img_arr.reshape([-1,1])

            # if len(img_arr.shape) == 2:  # grayscale
            #     img_arr = np.tile(img_arr, [512, 3, 1, 1]).transpose(2, 3, 0, 1)
            # img_arr = img_arr
            return img_arr

        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        # if img_name != msk_name:
        #     assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample = {'image': image, 'mask': mask}
        return sample
