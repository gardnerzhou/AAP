"""
数据增强模块
包含翻转、旋转、平移、颜色变化、放大缩小等增强方法
注意：确保图像和mask的几何变换完全同步
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import threading
from torchvision.transforms.functional import InterpolationMode


_thread_local = threading.local()


class SynchronizedGeometricTransform:
    """
    同步的几何变换类，确保图像和mask使用完全相同的随机参数
    通过线程局部存储缓存随机参数，确保同一线程内的多次调用使用相同的参数
    """
    def __init__(self, flip_p_h=0.5, flip_p_v=0.2, degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)):
        self.flip_p_h = flip_p_h
        self.flip_p_v = flip_p_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
    
    def _get_or_generate_params(self):
        if not hasattr(_thread_local, 'geo_params') or _thread_local.geo_params is None:
            rand_vals = torch.rand(6).tolist()
            do_hflip = rand_vals[0] < self.flip_p_h
            do_vflip = rand_vals[1] < self.flip_p_v
            angle = rand_vals[2] * 2 * self.degrees - self.degrees
            if isinstance(self.translate, tuple):
                max_dx = self.translate[0]
                max_dy = self.translate[1]
            else:
                max_dx = max_dy = self.translate
            dx = rand_vals[3] * 2 * max_dx - max_dx
            dy = rand_vals[4] * 2 * max_dy - max_dy
            scale_val = self.scale[0] + rand_vals[5] * (self.scale[1] - self.scale[0])
            scale_val = max(0.1, min(scale_val, 2.0))
            _thread_local.geo_params = {'hflip': do_hflip,'vflip': do_vflip,'angle': angle,'translate': (dx, dy),'scale': scale_val}
        return _thread_local.geo_params
    
    def __call__(self, img):
        params = self._get_or_generate_params()
        if params['hflip']:
            img = TF.hflip(img)
        if params['vflip']:
            img = TF.vflip(img)
        w, h = img.size
        max_dx = params['translate'][0] * w
        max_dy = params['translate'][1] * h
        translations = (np.round(max_dx), np.round(max_dy))
        img = TF.affine(img, angle=params['angle'], translate=translations, scale=params['scale'], shear=0, fill=0, interpolation=InterpolationMode.NEAREST)
        return img
    
    @staticmethod
    def reset():
        if hasattr(_thread_local, 'geo_params'):
            _thread_local.geo_params = None


class RandomFlip:
    def __init__(self, p_horizontal=0.5, p_vertical=0.5):
        self.p_horizontal = p_horizontal
        self.p_vertical = p_vertical
    def __call__(self, img):
        rand_val = torch.rand(2).tolist()
        if rand_val[0] < self.p_horizontal:
            img = TF.hflip(img)
        if rand_val[1] < self.p_vertical:
            img = TF.vflip(img)
        return img


class RandomRotation:
    def __init__(self, degrees=15, interpolation=InterpolationMode.NEAREST):
        self.degrees = degrees
        self.interpolation = interpolation
    def __call__(self, img):
        angle = torch.rand(1).item() * 2 * self.degrees - self.degrees
        return TF.rotate(img, angle, interpolation=self.interpolation, fill=0)


class RandomTranslation:
    def __init__(self, translate=(0.1, 0.1), interpolation=InterpolationMode.NEAREST):
        self.translate = translate
        self.interpolation = interpolation
    def __call__(self, img):
        if isinstance(self.translate, tuple):
            max_dx = self.translate[0] * img.size[0]
            max_dy = self.translate[1] * img.size[1]
        else:
            max_dx = max_dy = self.translate * min(img.size)
        translations = (np.round(torch.rand(1).item() * 2 * max_dx - max_dx), np.round(torch.rand(1).item() * 2 * max_dy - max_dy))
        return TF.affine(img, angle=0, translate=translations, scale=1.0, shear=0, fill=0, interpolation=self.interpolation)


class RandomColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, img):
        if self.brightness > 0:
            brightness_factor = 1 + (torch.rand(1).item() * 2 - 1) * self.brightness
            img = TF.adjust_brightness(img, brightness_factor)
        if self.contrast > 0:
            contrast_factor = 1 + (torch.rand(1).item() * 2 - 1) * self.contrast
            img = TF.adjust_contrast(img, contrast_factor)
        if self.saturation > 0:
            saturation_factor = 1 + (torch.rand(1).item() * 2 - 1) * self.saturation
            img = TF.adjust_saturation(img, saturation_factor)
        if self.hue > 0:
            hue_factor = (torch.rand(1).item() * 2 - 1) * self.hue
            img = TF.adjust_hue(img, hue_factor)
        return img


class RandomScale:
    def __init__(self, scale_range=(0.9, 1.1), interpolation=InterpolationMode.NEAREST):
        self.scale_range = scale_range
        self.interpolation = interpolation
    def __call__(self, img):
        scale = self.scale_range[0] + torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
        scale = max(0.1, min(scale, 2.0))
        w, h = img.size
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = TF.resize(img, (new_h, new_w), interpolation=self.interpolation)
        if scale > 1.0:
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            img = TF.crop(img, top, left, h, w)
        else:
            pad_h = h - new_h
            pad_w = w - new_w
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = TF.pad(img, padding, fill=0)
        return img


class RandomAffine:
    def __init__(self, degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=InterpolationMode.NEAREST):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.interpolation = interpolation
    def __call__(self, img):
        rand_vals = torch.rand(4).tolist()
        angle = rand_vals[0] * 2 * self.degrees - self.degrees
        if isinstance(self.translate, tuple):
            max_dx = self.translate[0] * img.size[0]
            max_dy = self.translate[1] * img.size[1]
        else:
            max_dx = max_dy = self.translate * min(img.size)
        translations = (np.round(rand_vals[1] * 2 * max_dx - max_dx), np.round(rand_vals[2] * 2 * max_dy - max_dy))
        scale = self.scale[0] + rand_vals[3] * (self.scale[1] - self.scale[0])
        scale = max(0.1, min(scale, 2.0))
        return TF.affine(img, angle=angle, translate=translations, scale=scale, shear=0, fill=0, interpolation=self.interpolation)


def get_train_augmentation(image_size, use_augmentation=True):
    if use_augmentation:
        sync_geo_transform = SynchronizedGeometricTransform(flip_p_h=0.5, flip_p_v=0.5, degrees=0, translate=(0, 0), scale=(1.0, 1.0))
        transform = transforms.Compose([
            sync_geo_transform,
            RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_msk = transforms.Compose([
            sync_geo_transform,
            transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_msk = transforms.Compose([
            transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    return transform, transform_msk


def get_test_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_msk = transforms.Compose([
        transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return transform, transform_msk
