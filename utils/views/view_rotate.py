from PIL import Image
import torch

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im, background=None, **kwargs):
        
        res = im.transpose(-1, -2)
        res = torch.flip(res, dims=[-1])

        return res

        # # TODO: Implement forward_mapping
        # raise NotImplementedError("forward_mapping is not implemented yet.")

    def inverse_view(self, noise, background=None, **kwargs):
        # # TODO: Implement inverse_mapping
        res = noise.transpose(-1, -2)
        res = torch.flip(res, dims=[-2])

        return res
