import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.filters.rank import entropy
from skimage.morphology import disk
from math import log10, sqrt
from skimage.morphology import disk
from skimage.util import img_as_ubyte

class ImagePerfMetrics:
    def __init__(self):
        pass
    
    
    def contrast_improvement_index(self, original, enhanced):
        
        # print("enhanced.shape", type(enhanced))
        # print("original.shape", type(original))
        
        """
        Calculate the Contrast Improvement Index (CII).
        
        Parameters:
        original (numpy.ndarray): Original image.
        enhanced (numpy.ndarray): Enhanced image.
        
        Returns:
        float: Contrast Improvement Index.
        """
        original_contrast = np.std(original)
        enhanced_contrast = np.std(enhanced)
        cii = (enhanced_contrast - original_contrast) / original_contrast
        return cii

    def psnr(self, original, enhanced):
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR).
        
        Parameters:
        original (numpy.ndarray): Original image.
        enhanced (numpy.ndarray): Enhanced image.
        
        Returns:
        float: PSNR value in dB.
        """
        mse = np.mean((original - enhanced) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr_value = 20 * log10(max_pixel / sqrt(mse))
        return psnr_value

    def calculate_ssim(self, original, enhanced):
        """
        Calculate the Structural Similarity Index (SSIM).
    
        Parameters:
        original (numpy.ndarray): Original image.
        enhanced (numpy.ndarray): Enhanced image.
    
        Returns:
        float: SSIM value.
        """
        
        # print("original.shape", original.shape)
        # print("enhanced.shape", enhanced.shape)
        
        # Ensure both images have the same dimensions
        if original.shape != enhanced.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
        if len(original.shape) == 3:
            return ssim(original, enhanced, channel_axis=2)
        else:
            return ssim(original, enhanced)

    def calculate_entropy(self, image):
        """
        Calculate the entropy of an image.

        Parameters:
        image (numpy.ndarray): Input image.

        Returns:
        float: Entropy value.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure the image is in the correct range for img_as_ubyte
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
            image = img_as_ubyte(image)  # Convert to uint8

        entr_img = entropy(image, disk(5))
        return np.mean(entr_img)

    def mean_squared_error(self, original, enhanced):
        """
        Calculate the Mean Squared Error (MSE).
        
        Parameters:
        original (numpy.ndarray): Original image.
        enhanced (numpy.ndarray): Enhanced image.
        
        Returns:
        float: MSE value.
        """
        mse = np.mean((original - enhanced) ** 2)
        return mse

    def mean_absolute_error(self, original, enhanced):
        """
        Calculate the Mean Absolute Error (MAE).
        
        Parameters:
        original (numpy.ndarray): Original image.
        enhanced (numpy.ndarray): Enhanced image.
        
        Returns:
        float: MAE value.
        """
        mae = np.mean(np.abs(original - enhanced))
        return mae

    def plot_metrics(self, metrics):
        """
        Plot various image performance metrics.
        
        Parameters:
        metrics (dict): Dictionary containing metric names and their values.
        """
        plt.figure(figsize=(12, 6))
        for metric, values in metrics.items():
            plt.plot(values, marker='o', label=metric)
        plt.title('Image Performance Metrics')
        plt.xlabel('Image Index')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
# # Example usage
# # Initialize the class
# ppm = PreprocessPerfMetric()

# # Load a grayscale or colored fundus image
# image = cv2.imread('path_to_fundus_image.jpg', cv2.IMREAD_UNCHANGED)
# enhanced_image = ppm.apply_bpdfhe_gpu(image)

# # Calculate metrics
# cii_value = ppm.contrast_improvement_index(image, enhanced_image)
# psnr_value = ppm.psnr(image, enhanced_image)
# ssim_value = ppm.calculate_ssim(image, enhanced_image)
# entropy_value = ppm.calculate_entropy(enhanced_image)

# print(f"Contrast Improvement Index (CII): {cii_value}")
# print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB")
# print(f"Structural Similarity Index (SSIM): {ssim_value}")
# print(f"Entropy: {entropy_value}")

# # Example lists of metric values for plotting
# ciis = [cii_value]
# psnrs = [psnr_value]
# ssims = [ssim_value]
# entropies = [entropy_value]

# # Plot the results
# ppm.plot_cii(ciis)
# ppm.plot_psnr(psnrs)
# ppm.plot_ssim(ssims)
# ppm.plot_entropy(entropies)


# # Calculate metrics for a set of images
# ciis = []
# psnrs = []
# ssims = []
# entropies = []

# for i in range(num_images):
#     original_image = cv2.imread(f'path_to_original_image_{i}.jpg', cv2.IMREAD_UNCHANGED)
#     enhanced_image = apply_bpdfhe_gpu(original_image)
    
#     cii_value = contrast_improvement_index(original_image, enhanced_image)
#     psnr_value = psnr(original_image, enhanced_image)
#     ssim_value = calculate_ssim(original_image, enhanced_image)
#     entropy_value = calculate_entropy(enhanced_image)
    
#     ciis.append(cii_value)
#     psnrs.append(psnr_value)
#     ssims.append(ssim_value)
#     entropies.append(entropy_value)

# # Plot the results
# plot_cii(ciis)
# plot_psnr(psnrs)
# plot_ssim(ssims)
# plot_entropy(entropies)

# # Assuming you have a list of original and enhanced images
# original_images = [cv2.imread('path_to_original_image1.jpg'), cv2.imread('path_to_original_image2.jpg')]
# enhanced_images = [cv2.imread('path_to_enhanced_image1.jpg'), cv2.imread('path_to_enhanced_image2.jpg')]

# metrics_calculator = ImagePerfMetrics()

# ciis = []
# psnrs = []
# entropies = []
# mses = []
# maes = []

# for original, enhanced in zip(original_images, enhanced_images):
#     ciis.append(metrics_calculator.contrast_improvement_index(original, enhanced))
#     psnrs.append(metrics_calculator.psnr(original, enhanced))
#     entropies.append(metrics_calculator.calculate_entropy(enhanced))
#     mses.append(metrics_calculator.mean_squared_error(original, enhanced))
#     maes.append(metrics_calculator.mean_absolute_error(original, enhanced))

# metrics = {
#     'CII': ciis,
#     'PSNR': psnrs,
#     'Entropy': entropies,
#     'MSE': mses,
#     'MAE': maes
# }

# metrics_calculator.plot_metrics(metrics)
