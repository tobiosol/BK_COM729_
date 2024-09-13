import os
import sys
import cv2
import numpy as np
from skimage.filters import hessian, threshold_local
from skimage.morphology import opening, closing, disk
from skimage.segmentation import random_walker
import cupy as cp
from sklearn.decomposition import PCA
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.restoration import denoise_wavelet
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_local
from skimage.morphology import opening, closing, disk
from skimage.segmentation import random_walker
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = os.path.join(current_dir, 'fundus_v2')
sys.path.append(subdirectory_path)
import proj_util
from PIL import Image
import cv2
import numpy as np
import pywt
from skimage.restoration import denoise_nl_means, estimate_sigma

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from scipy.signal import wiener

import cv2
import numpy as np
from skimage import exposure, feature
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel
from skimage.morphology import erosion, dilation, opening, closing, disk
from skimage.segmentation import watershed
from skimage import exposure
class FundusImagePreprocessorV23:
    
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    def calculate_metrics(self, original_image, enhanced_image):
        
        print(type(original_image))
        print(type(enhanced_image))
        
        enhanced_image = cv2.resize(enhanced_image, (original_image.shape[1], original_image.shape[0]))
        # Contrast Improvement Index (CII)
        cii = np.std(enhanced_image) / np.std(original_image)
        # Structural Similarity Index (SSIM)
        ssim_value = ssim(original_image, enhanced_image)
        # Entropy
        hist, _ = np.histogram(enhanced_image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy_value = entropy(hist)
        
        # Mean-Standard Deviation Ratio
        mean_std_ratio_value = np.mean(enhanced_image) / np.std(enhanced_image)
        
        psnr_value = cv2.PSNR(original_image, enhanced_image)
        
        # print(f"CII: {cii}")
        # print(f"SSIM: {ssim_value}")
        # print(f"Entropy: {entropy_value}")
        # print(f"Mean-Standard Deviation Ratio: {mean_std_ratio_value}")
        
        print(f"SSIM: {ssim_value}, Entropy: {entropy_value}, PSNR: {psnr_value}, Mean/Std: {mean_std_ratio_value}, CII: {cii}")
        return cii, ssim_value, entropy_value, mean_std_ratio_value
    
    
    def adaptive_histogram_equalization(self, image):
        # Convert to uint8 if necessary
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_image = clahe.apply(image)
        return equalized_image

    def gaussian_blur(self, image):
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred_image

    def sobel_edge_detection(self, image):
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = np.uint8(sobel_combined)
        return sobel_combined

    def normalize(self, image):
        # Normalize the image
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_image

    # def preprocess(self, image_path):
    #     # Load the original image in grayscale
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    #     # Apply preprocessing techniques
    #     image = self.adaptive_histogram_equalization(image)
    #     image = self.gaussian_blur(image)
    #     # image = self.sobel_edge_detection(image)
        

        # return image
    
    
    
    
    # def preprocess(self, image_path):
    #     # Load the image as grayscale
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    #     # Resize the image to 256 x 256
    #     image = cv2.resize(image, (256, 256))
        
    #     # Contrast enhancement using histogram equalization
    #     enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
        
    #     # Convert the enhanced image to 8-bit format
    #     enhanced = (enhanced * 255).astype(np.uint8)
        
    #     # Noise reduction using Gaussian filter
    #     blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
    #     # Thresholding to segment bright lesions
    #     _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    #     # Morphological operations to remove small noise
    #     kernel = np.ones((3, 3), np.uint8)
    #     cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
        
    #     # Overlay the mask on the original image
    #     overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     overlay[cleaned == 255] = [0, 0, 255]  # Red color for the mask
        
    #     self.calculate_metrics(original_image=image, enhanced_image=cleaned)
        
    #     return overlay
    
    
    
    
    def preprocess(self, image_path):
        # Load the image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize the image to 256 x 256
        # resized = cv2.resize(image, (256, 256))
        
        # Contrast enhancement using histogram equalization
        enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
        
        # Convert the enhanced image to 8-bit format
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Noise reduction using Gaussian filter
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Thresholding to segment bright lesions
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to remove small noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Overlay the mask on the original image
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay[cleaned == 255] = [0, 0, 255]  # Red color for the mask
        
        # Plot the image progression
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        
        # axes[1].imshow(resized, cmap='gray')
        # axes[1].set_title('Resized Image')
        
        axes[1].imshow(enhanced, cmap='gray')
        axes[1].set_title('Enhanced Image')
        
        axes[2].imshow(blurred, cmap='gray')
        axes[2].set_title('Blurred Image')
        
        # axes[4].imshow(thresholded, cmap='gray')
        # axes[4].set_title('Thresholded Image')
        
        # axes[5].imshow(overlay)
        # axes[5].set_title('Overlay Image')
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        self.calculate_metrics(original_image=image, enhanced_image=cleaned)
    
    # def preprocess(self, image_path):
    #     # Load the image as grayscale
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    #     # Resize the image to 256 x 256
    #     image = cv2.resize(image, (256, 256))
        
    #     # Contrast enhancement using histogram equalization
    #     enhanced = exposure.equalize_adapthist(image, clip_limit=0.02)
        
    #     # Convert the enhanced image to 8-bit format
    #     enhanced = (enhanced * 255).astype(np.uint8)
        
    #     # Noise reduction using Gaussian filter
    #     # blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
    #     # # Thresholding to segment bright lesions
    #     # _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    #     # # Morphological operations to remove small noise
    #     # kernel = np.ones((3, 3), np.uint8)
    #     # cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
        
    #     # # Overlay the inverse mask on the original image
    #     # overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #     # overlay[cleaned == 0] = [0, 0, 255]  # Red color for the inverse mask
        
    #     self.calculate_metrics(original_image=image, enhanced_image=enhanced)
        
    #     return enhanced
    
    
    
    # def enhance_image(self, image):
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     enhanced_image = clahe.apply(image)
    #     return enhanced_image

    # def remove_noise(self, image):
    #     gaussian_filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    #     return gaussian_filtered_image

    # def localize_lesions(self, image):
    #     img = cv2.resize(image, (256, 256))
    #     img = img[np.newaxis, np.newaxis, :, :]  # Add batch and channel dimensions
    #     img = torch.from_numpy(img).float().div(255.0).to(self.device)

    #     with torch.no_grad():
    #         output = self.model(img)
    #         output = torch.sigmoid(output).cpu().numpy()[0, 0]

    #     # Thresholding to get binary mask
    #     mask = (output > 0.5).astype(np.uint8)

    #     # Morphological operations to refine the mask
    #     kernel = np.ones((5, 5), np.uint8)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #     return mask

    # def preprocess(self, image_path):
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     enhanced_image = self.enhance_image(image)
    #     noise_removed_image = self.remove_noise(enhanced_image)
    #     mask = self.localize_lesions(noise_removed_image)

    #     # Overlay mask on the original image
    #     result = cv2.bitwise_and(image, image, mask=mask)
    #     self.calculate_metrics(original_image=image, enhanced_image=result)
    #     return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    def save_processed_image(self, image, dest_image_dir, original_filename):
            new_filename = f"{os.path.splitext(original_filename)[0]}{os.path.splitext(original_filename)[1]}"
            save_path = os.path.join(dest_image_dir, new_filename)
            image_np = cp.asnumpy(image)
            cv2.imwrite(save_path, image_np)

    def preprocess_and_save_dataset(self, src_image_dir, dest_image_dir):
            image_paths = self.load_images(src_image_dir)
            for image_path in image_paths:
                original_filename = os.path.basename(image_path)
                processed_image = self.preprocess(image_path)
                if processed_image is not None:
                    # print("preprocess_and_save_dataset", processed_image.shape)
                    self.save_processed_image(processed_image, dest_image_dir, original_filename)
                else:
                    print(f"Skipping {original_filename} due to preprocessing error.")
    
    def load_images(self, image_folder):
                return [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(".jpg") or file.endswith(".png")]
        
    
    def predictor_preprocess(self, pil_image):
        # Convert PIL image to numpy array
        image = np.array(pil_image)

        # Convert to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Contrast enhancement using histogram equalization
        enhanced = exposure.equalize_adapthist(image_gray, clip_limit=0.03)

        # Convert the enhanced image to 8-bit format
        enhanced = (enhanced * 255).astype(np.uint8)

        # Noise reduction using Gaussian filter
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Thresholding to segment bright lesions
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to remove small noise
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

        # Convert cleaned image back to PIL format
        cleaned_pil = Image.fromarray(cleaned)

        return cleaned_pil
    
    
    

preprocessor = FundusImagePreprocessorV23()
image_path = 'timg/IMG0413 (8).png'
# image_path = 'timg/IMG0052.png'
# image_path = 'timg/A (117).png'
# processed_image = preprocessor.preprocess(image_path)
# Display the combined image
# cv2.imshow('Combined Image', processed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_TRAINING_DIR, dest_image_dir=proj_util.TRAINING_DIR)
# preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_VALIDATION_DIR, dest_image_dir=proj_util.VALIDATION_DIR)




# enhanced_image,coeffs= preprocessor.preprocess_and_plot('timg/55.png')
# enhanced_image = preprocessor.preprocess('timg/A (117).png')
# enhanced_image = preprocessor.preprocess('timg/IM000275.png')
# enhanced_image = preprocessor.preprocess('timg/55.png')
# # # print("coeffs", coeffs)
# cv2.imshow('Detections', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# enhanced_image, mask = preprocessor.preprocess('55.png')
# cv2.imwrite('processed_image.png', enhanced_image)
# cv2.imwrite('mask_image.png', mask)
# preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_TRAINING_DIR, dest_image_dir=proj_util.TRAINING_DIR)
# preprocessor.preprocess_and_save_dataset(src_image_dir=proj_util.ORIGINAL_VALIDATION_DIR, dest_image_dir=proj_util.VALIDATION_DIR)
    
# preprocessor.preprocess_and_save_dataset(src_image_dir="cws_training_png", dest_image_dir=proj_util.TRAINING_DIR)
# preprocessor.preprocess_and_save_dataset(src_image_dir="cws_validation_png", dest_image_dir=proj_util.VALIDATION_DIR)

# display_data_distribution('your_file.csv')
