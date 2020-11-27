from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


def crop(im, width_split, height_split):
    #Returns evenly split tiles. No. tiles is equal to width_split * height_split
    M = int(im.shape[1]//width_split)
    N = int(im.shape[0]//height_split)
    tiles = np.array([np.array(im[y:y+N,x:x+M]) for y in range(0,im.shape[0],N) for x in range(0,im.shape[1],M)])
    return tiles


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        wh_ratio = newW / newH
        img_tiles = crop(img_nd, wh_ratio*2, 2)

        if len(img_tiles.shape) == 3:
            img_tiles = np.expand_dims(img_tiles, axis=1)
        # Should now be TCHW, if not then need to np.transpose()
        if img_tiles.max() > 1:
            img_tiles = img_tiles / 255

        return img_tiles

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + '//' + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + '//' + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


# class CarvanaDataset(BasicDataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1):
#         super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
