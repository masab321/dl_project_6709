import random
from typing import Tuple, Union

import torch
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


TensorLike = Union[Tensor]


class SegmentationAugmentations:
	"""Joint augmentations for segmentation (image + mask).

	Assumes:
	- image: float tensor in [0, 1], shape (C, H, W)
	- mask:  float tensor in {0, 1}, shape (1, H, W)

	Geometric transforms are applied to both image and mask.
	Intensity transforms are applied to the image only.
	"""

	def __init__(
		self,
		# Geometric probabilities
		p_hflip: float = 0.5,
		p_vflip: float = 0.5,
		p_rotate: float = 0.5,
		min_rotate: float = 15.0,
		max_rotate: float = 30.0,
		p_random_crop: float = 0.5,
		crop_scale_min: float = 0.8,
		crop_scale_max: float = 1.0,
		# Intensity probabilities (image only)
		p_brightness: float = 0.4,
		brightness_factor_min: float = 0.8,
		brightness_factor_max: float = 1.2,
		p_contrast: float = 0.4,
		contrast_factor_min: float = 0.8,
		contrast_factor_max: float = 1.2,
		p_gamma: float = 0.3,
		gamma_min: float = 0.8,
		gamma_max: float = 1.2,
		p_gaussian_noise: float = 0.3,
		noise_std: float = 0.03,
		p_gaussian_blur: float = 0.3,
		blur_kernel_size: int = 3,
		blur_sigma_min: float = 0.1,
		blur_sigma_max: float = 1.0,
	) -> None:
		# Geometric
		self.p_hflip = p_hflip
		self.p_vflip = p_vflip
		self.p_rotate = p_rotate
		self.min_rotate = float(min_rotate)
		self.max_rotate = float(max_rotate)
		self.p_random_crop = p_random_crop
		self.crop_scale_min = float(crop_scale_min)
		self.crop_scale_max = float(crop_scale_max)

		# Intensity
		self.p_brightness = p_brightness
		self.brightness_factor_min = float(brightness_factor_min)
		self.brightness_factor_max = float(brightness_factor_max)

		self.p_contrast = p_contrast
		self.contrast_factor_min = float(contrast_factor_min)
		self.contrast_factor_max = float(contrast_factor_max)

		self.p_gamma = p_gamma
		self.gamma_min = float(gamma_min)
		self.gamma_max = float(gamma_max)

		self.p_gaussian_noise = p_gaussian_noise
		self.noise_std = float(noise_std)

		self.p_gaussian_blur = p_gaussian_blur
		self.blur_kernel_size = int(blur_kernel_size)
		if self.blur_kernel_size % 2 == 0:
			self.blur_kernel_size += 1  # must be odd
		self.blur_sigma_min = float(blur_sigma_min)
		self.blur_sigma_max = float(blur_sigma_max)

	def __call__(self, image: TensorLike, mask: TensorLike) -> Tuple[Tensor, Tensor]:
		# Ensure tensor type
		if not torch.is_tensor(image):
			image = torch.as_tensor(image)
		if not torch.is_tensor(mask):
			mask = torch.as_tensor(mask)

		_, h, w = image.shape

		# ----- Geometric transforms (image + mask) -----

		# Horizontal flip
		if random.random() < self.p_hflip:
			image = torch.flip(image, dims=[2])
			mask = torch.flip(mask, dims=[2])

		# Vertical flip
		if random.random() < self.p_vflip:
			image = torch.flip(image, dims=[1])
			mask = torch.flip(mask, dims=[1])

		# Rotation (±min_rotate to ±max_rotate)
		if random.random() < self.p_rotate:
			angle_mag = random.uniform(self.min_rotate, self.max_rotate)
			angle = angle_mag * random.choice([-1.0, 1.0])
			image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
			mask = F.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

		# Random crop / zoom (random resized crop back to original size)
		if random.random() < self.p_random_crop:
			scale = random.uniform(self.crop_scale_min, self.crop_scale_max)
			crop_h = max(1, int(h * scale))
			crop_w = max(1, int(w * scale))

			if crop_h < h and crop_w < w:
				top = random.randint(0, h - crop_h)
				left = random.randint(0, w - crop_w)

				image = F.resized_crop(
					image,
					top,
					left,
					crop_h,
					crop_w,
					size=[h, w],
					interpolation=InterpolationMode.BILINEAR,
				)
				mask = F.resized_crop(
					mask,
					top,
					left,
					crop_h,
					crop_w,
					size=[h, w],
					interpolation=InterpolationMode.NEAREST,
				)

		# ----- Intensity transforms (image only) -----

		# Brightness shift
		if random.random() < self.p_brightness:
			factor = random.uniform(self.brightness_factor_min, self.brightness_factor_max)
			image = F.adjust_brightness(image, factor)

		# Contrast shift
		if random.random() < self.p_contrast:
			factor = random.uniform(self.contrast_factor_min, self.contrast_factor_max)
			image = F.adjust_contrast(image, factor)

		# Gamma correction
		if random.random() < self.p_gamma:
			gamma = random.uniform(self.gamma_min, self.gamma_max)
			image = F.adjust_gamma(image, gamma)

		# Gaussian noise (low)
		if random.random() < self.p_gaussian_noise:
			noise = torch.randn_like(image) * self.noise_std
			image = torch.clamp(image + noise, 0.0, 1.0)

		# Gaussian blur (very mild)
		if random.random() < self.p_gaussian_blur:
			sigma = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
			image = F.gaussian_blur(
				image,
				kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
				sigma=sigma,
			)

		return image, mask


__all__ = ["SegmentationAugmentations"]
