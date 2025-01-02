import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from .view_base import BaseView

def get_circle_mask(img_size: int, r: int):
    # TODO: Implement get_circle_mask
    # raise NotImplementedError(f"get_circle_mask not implemented")

    h, w = img_size, img_size
    c = (h // 2, w // 2)  # Automatically center the circle in the middle of the image
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dst = torch.sqrt((X - c[1]) ** 2 + (Y - c[0]) ** 2)
    mask = (dst <= r).float()  # 1 inside the circle, 0 outside
    # print(type(mask))
    return mask

def inner_rotate_func_with_mask(im: torch.Tensor, mask: torch.Tensor, angle, interpolate=False):
    # TODO: Implement inner_rotate_func_with_mask
    # raise NotImplementedError(f"inner_rotate_func_with_mask not implemented")
    
    C, W, H = im.shape

    center = H // 2
    
    mask_expanded = mask.unsqueeze(0).expand(C, W, H)

    masked_im = im * mask_expanded

    masked_pil = TF.to_pil_image(masked_im)
    
    # Rotate the masked inner circle region around the center
    rotated_masked_pil = TF.rotate(
        masked_pil, 
        angle=angle, 
        interpolation=InterpolationMode.BILINEAR, 
        center=(center, center)
    )

    # print('here')
    rotated_masked_tensor = TF.to_tensor(rotated_masked_pil).to(im.device)
    output_images = rotated_masked_tensor * mask_expanded + im * (1 - mask_expanded)
    # print(output_images.shape)

    output_images = output_images.to(im.device).half()
    # print(output_images.shape)
    
    return output_images
    

 
class InnerRotateView(BaseView):
    """
    Implements an "inner circle" view, where a circle inside the image spins
    but the border stays still. Inherits from `PermuteView`, which implements
    the `view` and `inverse_view` functions as permutations. We just make
    the correct permutation here, and implement the `make_frame` method
    for animation
    """

    def __init__(self, angle):
        """
        Make the correct "inner circle" permutations and pass it to the
        parent class constructor.
        """
        self.angle = angle
        self.stage_1_mask = get_circle_mask(64, 24)
        self.stage_2_mask = get_circle_mask(256, 96)

    def view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask

        inner_rotated = inner_rotate_func_with_mask(im, mask, -self.angle, interpolate=False)
        # print(inner_rotated.dtype)

        return inner_rotated

    def inverse_view(self, im, **kwargs):
        im_size = im.shape[-1]
        if im_size == 64:
            mask = self.stage_1_mask.to(im)
            self.stage_1_mask = mask
        elif im_size == 256:
            mask = self.stage_2_mask.to(im)
            self.stage_2_mask = mask
        

        inner_rotated = inner_rotate_func_with_mask(im, mask, self.angle, interpolate=False)
        # print(inner_rotated.dtype)

        return inner_rotated

