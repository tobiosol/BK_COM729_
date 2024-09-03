import itertools
import os
import random
import sys
from math import log10, sqrt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)

import os
import cupy as cp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import proj_util
from fundus_v2 import image_perf_metric
import cv2.ximgproc as ximgproc
from skimage.restoration import denoise_tv_chambolle
import torch.nn.functional as F
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy



class FundusImagePreprocessor:
    
    
    def __init__(self):
        pass

    
    
    def extract_green_layer(self, image):
        if len(image.shape) == 3:
            image = image[:, :, 1]
        return image
    
    def resize_image(self, image):
        # Resize the image by dividing both sides by 2
        new_size = (image.shape[1] // 2, image.shape[0] // 2)
        resized_image = cv2.resize(cp.asnumpy(image), new_size, interpolation=cv2.INTER_AREA)
        image = cp.asarray(resized_image)
        return image
    
    
    def apply_median_filter(self, image):
        image = cp.asarray(cv2.medianBlur(cp.asnumpy(image), 5))
        return image
    
    
    def remove_blood_vessels(self, image):
        kernel = cp.ones((5, 5), cp.uint8)
        kernel_np = cp.asnumpy(kernel)  # Convert CuPy array to NumPy array
        blood_vessels = cp.asarray(cv2.morphologyEx(cp.asnumpy(image), cv2.MORPH_TOPHAT, kernel_np))
        image = cp.subtract(image, blood_vessels)
        return image
    
        
    
    
    
    def remove_optic_disk(self, image):
        # Ensure the image is in grayscale and of type CV_8UC1
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(cp.asnumpy(image), cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cp.asnumpy(image)
    
        gray_image = gray_image.astype(np.uint8)  # Ensure the image is a NumPy array of type uint8
    
        # Apply HoughCircles to detect circles
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=100)
    
        mask = cp.zeros(image.shape[:2], dtype=cp.uint8)
    
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                x, y, r = int(x), int(y), int(r)  # Ensure coordinates and radius are integers
                cv2.circle(cp.asnumpy(mask), (x, y), r, (255, 255, 255), -1)
    
        # Convert mask to CuPy array and invert it
        mask = cp.asarray(mask)
        inverted_mask = cp.bitwise_not(mask)
    
        # Ensure the mask has the same number of dimensions as the image
        inverted_mask = inverted_mask[:, :, None]
    
        # Remove the extra dimension from the mask
        inverted_mask = inverted_mask[:, :, 0]
    
        # Apply the mask using CuPy's array operations
        image = image * (inverted_mask / 255)
    
        return image
    
    
    

    
    

    def detect_microaneurysms(self, image):
        if image.dtype != cp.uint8:
            image = cp.clip(image * (255.0 / cp.max(image)), 0, 255).astype(cp.uint8)

        # Convert the image to a NumPy array for OpenCV processing
        image_np = cp.asnumpy(image)

        # Ensure the kernel is a NumPy array
        kernel = np.ones((3, 3), np.uint8)

        # Apply morphological opening to remove small objects
        small_objects = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)

        # Convert the result back to a CuPy array
        small_objects_gpu = cp.asarray(small_objects)

        return small_objects_gpu

    def compute_cii(self, original_image, processed_image):
        original_image = cp.asarray(original_image, dtype=cp.float32)
        processed_image = cp.asarray(processed_image, dtype=cp.float32)
        
        if len(original_image.shape) == 3:
            original_gray = cp.asarray(cv2.cvtColor(cp.asnumpy(original_image), cv2.COLOR_BGR2GRAY))
        else:
            original_gray = original_image
        
        if len(processed_image.shape) == 3:
            processed_gray = cp.asarray(cv2.cvtColor(cp.asnumpy(processed_image), cv2.COLOR_BGR2GRAY))
        else:
            processed_gray = processed_image
        
        cii = cp.std(processed_gray) / cp.std(original_gray)
        return cii
    
    def compute_entropy(self, image):
        image = cp.asarray(image, dtype=cp.float32)
        
        if len(image.shape) == 3:
            gray_image = cp.asarray(cv2.cvtColor(cp.asnumpy(image), cv2.COLOR_BGR2GRAY))
        else:
            gray_image = image
        
        hist, _ = cp.histogram(gray_image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -cp.sum(hist * cp.log2(hist + 1e-7))
        return entropy
    
    
    def preprocess_image(self, image):
        image = self.resize_image(image)
        image = self.extract_green_layer(image)
        image = self.apply_median_filter(image)
        image = self.remove_blood_vessels(image)
        image = self.remove_optic_disk(image)        
        image = self.detect_microaneurysms(image)
        return image
    
    
    
    def apply_non_local_means(self, image, h=8, template_window_size=7, search_window_size=21):
        """Applies non-local means denoising using CuPy and OpenCV.

        Args:
            image: Input image as a CuPy array.
            h: Parameter controlling filter strength.
            template_window_size: Size of the local patch to be compared.
            search_window_size: Size of the search window for similar patches.

        Returns:
            Denoised image as a CuPy array.
        """

        # Ensure image is in BGR format for OpenCV compatibility
        if image.ndim == 2:
            image = cp.stack((image, image, image), axis=-1)

        # Convert to numpy array for OpenCV
        image_np = cp.asnumpy(image)

        # Apply non-local means denoising using OpenCV
        denoised_np = cv2.fastNlMeansDenoisingColored(image_np, None, h, h, template_window_size, search_window_size)

        # Convert back to CuPy array
        denoised = cp.array(denoised_np)

        return denoised

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0, device='cpu'):
        """Applies unsharp masking using CuPy.
    
        Args:
            image: Input image as a CuPy array.
            kernel_size: Size of the Gaussian blur kernel.
            sigma: Standard deviation of the Gaussian kernel.
            amount: Strength of the sharpening effect.
            threshold: Threshold for clipping values.
    
        Returns:
            Sharpened image as a CuPy array.
        """
    
        blurred = cp.array(cv2.GaussianBlur(cp.asnumpy(image), kernel_size, sigma))
        sharpened = (amount + 1) * image - amount * blurred
        sharpened = cp.clip(sharpened, 0, 255).astype(cp.uint8)
        return sharpened
    
    
    
    def save_processed_image(self, image, dest_image_dir, original_filename):
        new_filename = f"{os.path.splitext(original_filename)[0]}{os.path.splitext(original_filename)[1]}"
        save_path = os.path.join(dest_image_dir, new_filename)
        image_np = cp.asnumpy(image)
        cv2.imwrite(save_path, image_np)

    def preprocess_and_save_dataset(self, src_image_dir, dest_image_dir):
        image_paths = self.load_images(src_image_dir)
        for image_path in image_paths:
            original_filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            
            
            
            # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)            
            processed_image = self.preprocess_image(image=image)
            if processed_image is not None:
                print("preprocess_and_save_dataset", processed_image.shape)
                self.save_processed_image(processed_image, dest_image_dir, original_filename)
            else:
                print(f"Skipping {original_filename} due to preprocessing error.")
    
    def load_images(self, image_folder):
        return [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".jpg") or file.endswith(".png")]


    
    def display_data_distribution(csv_file):
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Plot the distribution of each column
        for column in data.columns:
            plt.figure(figsize=(10, 6))
            data[column].hist(bins=30, edgecolor='black')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(False)
            plt.show()
    
    
if __name__ == "__main__":
    preprocessor = FundusImagePreprocessor()
    image = cv2.imread('107.png')
    processed_image = preprocessor.preprocess_image(image)
    cii = preprocessor.compute_cii(image, processed_image)
    entropy = preprocessor.compute_entropy(processed_image)

    print(f'CII: {cii}')
    print(f'Entropy: {entropy}')
    # print(preprocessed_images)
    # preprocessor.display_transformations("107.png")
    
    # preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_TRAINING_DIR, dest_image_dir=proj_util.TRAINING_DIR)
    # preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_VALIDATION_DIR, dest_image_dir=proj_util.VALIDATION_DIR)
    
    # preprocessor.read_and_process_images("test")
    
    # preprocessor.preprocess_and_save_dataset(src_image_dir="cws_training_png", dest_image_dir=proj_util.TRAINING_DIR)
    # preprocessor.preprocess_and_save_dataset(src_image_dir="cws_validation_png", dest_image_dir=proj_util.VALIDATION_DIR)
    


# if __name__ == "__main__":
    
    # train_image_dir = proj_util.train_image_dir
    # test_image_dir = proj_util.test_image_dir
    # eval_image_dir = proj_util.eval_image_dir
    
    # orig_train_image_dir = proj_util.orig_train_image_dir
    # orig_eval_image_dir = proj_util.orig_eval_image_dir
    # orig_test_image_dir = proj_util.orig_test_image_dir
    
    # mtraining_dir = proj_util.mtraining_dir
    # mvalidation_dir = proj_util.mvalidation_dir
    
    # image = Image.open('augment.png')
    # preprocessor = FundusImagePreprocessor()
    # preprocessed_images = preprocessor.preprocess(image)
    
    # preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_TRAINING_DIR, dest_image_dir=proj_util.TRAINING_DIR)
    # preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_VALIDATION_DIR, dest_image_dir=proj_util.VALIDATION_DIR)
    
    # preprocessor.preprocess_and_save_dataset(src_image_dir="cws_training_png", dest_image_dir=proj_util.TRAINING_DIR)
    # preprocessor.preprocess_and_save_dataset(src_image_dir="cws_validation_png", dest_image_dir=proj_util.VALIDATION_DIR)
    

# Example usage:
# image = Image.open('path_to_fundus_image.jpg')
# preprocessor = FundusImagePreprocessor()
# preprocessed_images = preprocessor.preprocess(image)
# Example usage:
# display_transformations('path_to_fundus_image.jpg')
