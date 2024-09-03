import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torchvision import transforms

class FundusImageAugmentor:
    def __init__(self):
        self.augmentations = [
            self.random_flip,
            self.random_rotate,
            self.random_brightness,
            self.random_contrast,
            self.random_noise,
            self.random_horizontal_flip,
            self.random_vertical_flip,
            self.random_flip_and_color_jitter,
            self.random_resized_crop
        ]
        
        # self.train_transform = transforms.Compose([
        #     # transforms.RandomRotation(degrees=(-20, 20)),
        #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.RandomHorizontalFlip(p=1.0),
        #     transforms.RandomVerticalFlip(p=1.0),
        #     transforms.ColorJitter(brightness=0.05),
        #     # AddGaussianNoise(mean=0.0, std=10.0),
            
        #     transforms.RandomResizedCrop(size=(224, 224)),
            
        #     # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        #     # transforms.GaussianBlur(kernel_size=3),
        #     # transforms.RandomAffine(degrees=10, translate=(0.0, 0.0), shear=0),
        #     transforms.ToTensor(), 
        #     # transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        #     # transforms.ToPILImage,
        #     transforms.Resize(224),
        #     transforms.CenterCrop(224),
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.Normalize(mean=[0.5], std=[0.5])
        # ])
        
        
        
        
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.ColorJitter(brightness=0.05, contrast=0.02, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            # transforms.ColorJitter(brightness=0.05),
            # ConditionalGaussianNoise(mean=0.0, std=5.0, probability=0.5),
            # AddGaussianNoise(),
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1),
            # transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            # transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.test_transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    
    
    
    
    def augment_image(self, image, augmentation_idx):
        augmentation = self.augmentations[augmentation_idx % len(self.augmentations)]
        return augmentation(image)
    
    def random_flip(self, image):
        return ImageOps.mirror(image) if random.choice([True, False]) else image
    
    def random_rotate(self, image):
        angle = np.random.uniform(-30, 30)
        return image.rotate(angle)
    
    def random_brightness(self, image):
        enhancer = ImageEnhance.Brightness(image)
        factor = np.random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    def random_contrast(self, image):
        enhancer = ImageEnhance.Contrast(image)
        factor = np.random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    
    def random_noise(self, image):
        img_array = np.array(image)
        noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
        noisy_img = Image.fromarray(np.clip(img_array + noise, 0, 255).astype(np.uint8))
        return noisy_img
    
    def random_horizontal_flip(self, image, p=0.5):
        return ImageOps.mirror(image) if random.random() < p else image

    def random_vertical_flip(self, image, p=0.5):
        return ImageOps.flip(image) if random.random() < p else image

    def random_flip_and_color_jitter(self, image, brightness=0.2, contrast=0.2, saturation=0.2):
        if random.choice([True, False]):
            image = ImageOps.mirror(image)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(1 - brightness, 1 + brightness))
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(1 - contrast, 1 + contrast))
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(1 - saturation, 1 + saturation))
        
        return image

    def random_resized_crop(self, image, size=(244, 244), scale=(0.8, 1.0)):
        width, height = image.size
        area = width * height
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(3. / 4., 4. / 3.)

        new_width = int(round(np.sqrt(target_area * aspect_ratio)))
        new_height = int(round(np.sqrt(target_area / aspect_ratio)))

        if new_width <= width and new_height <= height:
            x = random.randint(0, width - new_width)
            y = random.randint(0, height - new_height)
            image = image.crop((x, y, x + new_width, y + new_height))
        image = image.resize(size, Image.BILINEAR)
        
        return image


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image but got {type(img)}")
        
        # Convert PIL image to tensor
        tensor = transforms.ToTensor()(img)
        
        # Add Gaussian noise
        noisy_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        
        # Convert tensor back to PIL image
        noisy_img = transforms.ToPILImage()(noisy_tensor)
        
        return noisy_img

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    
class ConditionalGaussianNoise:
    def __init__(self, mean=0.0, std=1.0, probability=0.5):
        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, tensor):
        if random.random() < self.probability:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor