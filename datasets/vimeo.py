import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('vimeo-sep')
class VIMEO_SEP(Dataset):

    def __init__(self, root_path, stage='train', n_frames=7, first_k=None):
        with open(osp.join(root_path, f'sep_{stage}list.txt'), 'r') as f:
            paths = [l.rstrip('\n') for l in f.readlines()]
        self.paths = [osp.join(root_path, 'sequences', x) for x in paths]
        self.n_frames = n_frames
        self.transform = transforms.Compose([
            transforms.Resize((64, 112)),
            transforms.ToTensor(),
        ])

        if first_k is not None:
            self.paths = self.paths[:first_k]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = []
        for i in range(1, self.n_frames + 1):
            img = Image.open(osp.join(self.paths[idx], f'im{i}.png'))
            x.append(self.transform(img))
        x = torch.stack(x, dim=1)
        return x
