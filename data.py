from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
from collections import defaultdict as dd
import numpy as np
from torchvision.datasets.folder import ImageFolder

DS_SEED = 666


class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input1 = np.load(os.path.join(self.root, 'facescrub_actor.npy'))
        input2 = np.load(os.path.join(self.root, 'facescrub_actress.npy'))
        actor_images = input1[0]
        actor_labels = input1[1]
        actress_images = input2[0]
        actress_labels = input2[1]

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)

        # v_min = np.min(data, axis=0)
        # v_max = np.max(data, axis=0)
        # data = (data - v_min) / (v_max - v_min)

        np.random.seed(DS_SEED)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        else:
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target





class CelebA(ImageFolder):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        root = os.path.expanduser(root)
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, ''
            ))

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self._cleanup()
        self.ntest = 0  # Reserve these many examples per class for evaluation
        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def _cleanup(self):
        # Remove examples belonging to classes of FaceScrub
        classes_path = osp.join(self.root, 'class_list.txt')
        with open(classes_path, 'r') as f:
            for cl in f.readlines():
                cl = cl.strip()
                cl_idx = self.class_to_idx[cl]
                self.samples = [s for s in self.samples if s[1] != cl_idx]


    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Use this random seed to make partition consistent
        prev_state = np.random.get_state()
        np.random.seed(DS_SEED)

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]     # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]    # Train on remaining

        # Revert randomness to original state
        np.random.set_state(prev_state)

        return partition_to_idxs

