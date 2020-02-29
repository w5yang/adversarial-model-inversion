from torchvision import transforms
from torchvision.transforms import functional as F
import torch
import torch.nn as nn
import numbers
from model import Inversion, Classifier


class AssignedCenterCrop(object):
    """Crops the given PIL Image with a given left up corner and size.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        leftUpCorner (sequence): Desired left up corner of the crop.
    """

    def __init__(self, size, leftUpCorner):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.leftUpCorner = leftUpCorner

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        return F.crop(img, self.leftUpCorner[0], self.leftUpCorner[1], self.size[0], self.size[1])

    def __repr__(self):
        return self.__class__.__name__ + '(leftUp={0},size={0})'.format(self.leftUpCorner, self.size)


tf = transforms.Compose([
    AssignedCenterCrop(64, (35, 70)),
    transforms.ToTensor(),
])


class default_args(object):
    nc = 1
    ndf = 128
    ngf = 128
    nz = 530
    truncation = 530
    c = 50


def load_blackbox(path, args=default_args, device=torch.device('cuda')):
    inversion = nn.DataParallel(
        Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.truncation, c=args.c)).to(device)
    try:
        checkpoint = torch.load(path)
        inversion.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        epoch = checkpoint['epoch']
        best_recon_loss = checkpoint['best_recon_loss']
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_recon_loss))
        return inversion
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
        return None



def load_classifier(path, args=default_args, device=torch.device('cuda')):
    classifier = nn.DataParallel(Classifier(nc=args.nc, ndf=args.ndf, nz=args.nz)).to(device)
    try:
        checkpoint = torch.load(path)
        classifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        best_cl_acc = checkpoint['best_cl_acc']
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
        return classifier
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
        return None