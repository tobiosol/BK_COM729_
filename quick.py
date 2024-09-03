# import tensorflow as tf
# import torch

# is_built_with_cuda = tf.test.is_built_with_cuda()

# gpu_available = tf.test.is_gpu_available()
# is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
# is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))

# print(is_built_with_cuda)
# print(tf.config.list_physical_devices('GPU'))
# print(gpu_available)
# print(is_cuda_gpu_available)
# print(is_cuda_gpu_min_3)

# # tf.test.is_gpu_available()
# cuda_device = torch.cuda.is_available()
# print(cuda_device)


# import pandas as pd

# def filter_and_save_csv(file_name, output_file, dr_filter=1, brvo_filter=1, disease_risk_filter=0):
#     # Load the original CSV file
#     df = pd.read_csv(file_name)

#     # Filter rows based on conditions
#     df['Disease_Risk'] = 0  # Initialize Disease Risk to 0 for all rows

#     dr_df = df[df['DR'] == dr_filter]
#     brvo_df = df[df['CWS'] == brvo_filter]

#     # Combine the filtered dataframes, keeping only rows where both DR and CWS are not 0
#     combined_df = pd.concat([dr_df, brvo_df]).drop_duplicates()

#     # Find the maximum length of the columns
#     max_length = max(len(dr_df), len(brvo_df))

#     # Reindex to ensure the same height, filling missing values with the previous value
#     combined_df = combined_df.reindex(range(max_length)).fillna(method='ffill')

#     # Set Disease Risk to 1 where both DR and CWS are 0
#     combined_df.loc[(combined_df['DR'] == 0) & (combined_df['CWS'] == 0), 'Disease_Risk'] = 1

#     # Select only the required columns
#     combined_df = combined_df[['ID', 'Disease_Risk', 'DR', 'CWS']]

#     # Convert relevant columns to integers
#     combined_df = combined_df.astype({'ID': 'int', 'Disease_Risk': 'int', 'DR': 'int', 'CWS': 'int'})

#     # Save to a new CSV file
#     combined_df.to_csv(output_file, index=False)

# # Example usage
# # filter_and_save_csv('RFMiD_Training_Labels.csv', 'filtered.csv')


# def print_rows_with_cws_1(file_name):
#     # Read the CSV file
#     df = pd.read_csv(file_name)

#     # Print rows where CWS is 1
#     print(df[df['CWS'] == 1])

# # Example usage
# print_rows_with_cws_1('filtered.csv')



from PIL import Image
import os

from sklearn.decomposition import PCA

from unet_model import UNet



ROOT_DIR = os.path.join("dataset", "v2")

LABEL_DIR = os.path.join(ROOT_DIR, "label")
TRAIN_LABEL_PATH = os.path.join(LABEL_DIR, "train_label.csv")
VALIDATION_LABEL_PATH = os.path.join(LABEL_DIR, "validation_label.csv")
TEST_LABEL_PATH = os.path.join(LABEL_DIR, "test_label.csv")

IMAGES_DIR = os.path.join(ROOT_DIR, "images")
TRAINING_DIR = os.path.join(IMAGES_DIR, "training")
VALIDATION_DIR = os.path.join(IMAGES_DIR, "validation")
TESTING_DIR = os.path.join(IMAGES_DIR, "testing")

ORIGINAL_DIR = os.path.join(ROOT_DIR, "original")
ORIGINAL_TRAINING_DIR = os.path.join(ORIGINAL_DIR, "training")
ORIGINAL_TESTING_DIR = os.path.join(ORIGINAL_DIR, "testing")
ORIGINAL_VALIDATION_DIR = os.path.join(ORIGINAL_DIR, "validation")



import csv
import os
from shutil import copyfile


def load_images_from_folder(image_folder):
    image_paths = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image_paths.append(image_path)
            
            base_name = os.path.splitext(filename)[0]
            print(base_name)
            
    return image_paths

load_images_from_folder("ncws")
# load_images_from_folder("nvcws")

def copy_image(file_name, source_folder, target_folder):
    # Read the CSV file
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get the header row
        id_index = headers.index('ID')

        # Copy images for each row in the CSV file
        for row in reader:
            image_file_name = str(row[id_index]) + '.png'
            source_image_path = os.path.join(source_folder, image_file_name)
            target_image_path = os.path.join(target_folder, image_file_name)

            if os.path.isfile(source_image_path):
                copyfile(source_image_path, target_image_path)
                print(f"Image copied: {image_file_name}")
            else:
                print(f"No image found for ID {row[id_index]}")

# Example usage
# copy_image(TRAIN_LABEL_PATH, 'rfmid_training', ORIGINAL_TRAINING_DIR)
# copy_image(VALIDATION_LABEL_PATH, 'rfmid_validation', ORIGINAL_VALIDATION_DIR)
# copy_image("more_cws.csv", 'cws_png', "cotton_wool_spots")



# from PIL import Image

# # Open the PGM file
# pgm_image = Image.open('IM001082.pgm')

# # Save the image as PNG
# pgm_image.save('IM001082.png')

# print("Image has been converted from PGM to PNG.")





# # Function to convert PGM to PNG and save in a different directory
# def convert_pgm_to_png(source_dir, dest_dir):
#     # Create the destination directory if it doesn't exist
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
    
#     # Iterate over all files in the source directory
#     for filename in os.listdir(source_dir):
#         if filename.endswith('.pgm'):
#             # Open the PGM file
#             pgm_file_path = os.path.join(source_dir, filename)
#             pgm_image = Image.open(pgm_file_path)
            
#             # Get the base name without extension and remove characters after a space
#             base_name = os.path.splitext(filename)[0].split(' ')[0]
            
#             # Save the image as PNG in the specified destination directory
#             png_file_path = os.path.join(dest_dir, base_name + '.png')
#             pgm_image.save(png_file_path)
            
#             print(f"Image has been converted from {pgm_file_path} to {png_file_path}.")



# def convert_pgm_to_png(source_dir, dest_dir):
#     # Create the destination directory if it doesn't exist
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)
    
#     # Iterate over all files in the source directory
#     for filename in os.listdir(source_dir):
#         if filename.endswith('.pgm'):
#             # Open the PGM file
#             pgm_file_path = os.path.join(source_dir, filename)
#             pgm_image = Image.open(pgm_file_path)
            
#             # Get the base name without extension and remove characters after a space
#             base_name = os.path.splitext(filename)[0].split(' ')[0]
            
#             # print(f"Converting {filename} to PNG...")
            
#             # Save the image as PNG in the specified destination directory
#             png_file_path = os.path.join(dest_dir, base_name + '.png')
#             pgm_image.save(png_file_path)
            
#             print(base_name)

# # Example usage
# convert_pgm_to_png('cws_validation_pgm', 'cws_validation_png')



def convert_pgm_to_png(source_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.pgm'):
            # Open the PGM file
            pgm_file_path = os.path.join(source_dir, filename)
            pgm_image = Image.open(pgm_file_path)
            
            # Get the base name without extension
            base_name = os.path.splitext(filename)[0]
            
            # Save the image as PNG in the specified destination directory
            png_file_path = os.path.join(dest_dir, base_name + '.png')
            pgm_image.save(png_file_path)
            
            # Print the base name without extension
            print(base_name)

# Example usage
# convert_pgm_to_png('Deep Hemorrhages', 'deep_hemorrhages')
# convert_pgm_to_png('Hard Exudates', 'hard_exudates')
# convert_pgm_to_png('Cotton Wool Spots', 'cotton_wool_spots')
# convert_pgm_to_png('val/normal', 'validation_png')
# convert_pgm_to_png('cws_pgm', 'cws_png')


import os
from PIL import Image

def convert_ppm_to_png(source_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.ppm'):
            # Open the PPM file
            ppm_file_path = os.path.join(source_dir, filename)
            ppm_image = Image.open(ppm_file_path)
            
            # Get the base name without extension
            base_name = os.path.splitext(filename)[0]
            
            # Save the image as PNG in the specified destination directory
            png_file_path = os.path.join(dest_dir, base_name + '.png')
            ppm_image.save(png_file_path)
            
            # Print the base name without extension
            print(base_name)

# convert_ppm_to_png('cws_pgm', 'cws_png')




import os
import pandas as pd

def update_csv_with_images(csv_file, image_dir):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Get the list of image names without extensions
    image_names = [os.path.splitext(filename)[0] for filename in os.listdir(image_dir) if filename.endswith(('.png', '.jpg', '.jpeg', '.ppm'))]
    
    # Iterate over the DataFrame and update the 3rd column
    for index, row in df.iterrows():
        if row['ID'] in image_names:
            df.at[index, df.columns[2]] = 1
        else:
            df.at[index, df.columns[2]] = 0
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

# Example usage
# update_csv_with_images(TRAIN_LABEL_PATH, 'cotton_wool_spots')
















# import pycuda.driver as cuda
#     cuda.init()
#     print("CUDA device count:", cuda.Device.count())


# import cv2
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the original image
# img = cv2.imread('56.png', 0)

# # Apply Otsu's thresholding
# _, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Apply binary thresholding
# _, binary_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# # Convert the original image to BGR for overlay
# img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # Create colored overlays
# otsu_overlay = img_color.copy()
# binary_overlay = img_color.copy()

# # Apply red color to the Otsu thresholded regions
# otsu_overlay[otsu_thresh == 255] = [0, 0, 255]

# # Apply green color to the binary thresholded regions
# binary_overlay[binary_thresh == 255] = [0, 255, 0]

# # Combine the overlays
# combined_overlay = cv2.addWeighted(otsu_overlay, 0.5, binary_overlay, 0.5, 0)

# gray_comb = cv2.cvtColor(combined_overlay, cv2.COLOR_RGB2GRAY)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# gc = clahe.apply(gray_comb)
# cv2.imwrite('gray_comb.png', gc)

# # Display the images
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1), plt.title('Original Image'), plt.imshow(img, cmap='gray')
# plt.subplot(2, 2, 2), plt.title('Otsu Thresholding'), plt.imshow(otsu_thresh, cmap='gray')
# plt.subplot(2, 2, 3), plt.title('Binary Thresholding'), plt.imshow(binary_thresh, cmap='gray')
# plt.subplot(2, 2, 4), plt.title('Combined Overlay'), plt.imshow(combined_overlay)
# plt.show()



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def apply_clahe(img):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     return clahe.apply(img)

# def apply_unsharp_mask(img, sigma=1.0, strength=1.5):
#     blurred = cv2.GaussianBlur(img, (0, 0), sigma)
#     sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
#     return sharpened

# # Load the original image
# img = cv2.imread('processed_image.png', 0)

# # Apply CLAHE
# clahe_img = apply_clahe(img)

# # Apply Unsharp Masking
# sharpened_img = apply_unsharp_mask(clahe_img)

# # Display the images
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 3, 1), plt.title('Original Image'), plt.imshow(img, cmap='gray')
# plt.subplot(1, 3, 2), plt.title('CLAHE Image'), plt.imshow(clahe_img, cmap='gray')
# plt.subplot(1, 3, 3), plt.title('Sharpened Image'), plt.imshow(sharpened_img, cmap='gray')
# plt.show()


# from sklearn.cluster import DBSCAN

# def eadbsc_segmentation(img):
#     # Convert image to binary
#     _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
#     # Flatten the image
#     flat_img = binary_img.flatten().reshape(-1, 1)
    
#     # Apply DBSCAN (as a placeholder for EADBSC)
#     clustering = DBSCAN(eps=3, min_samples=2).fit(flat_img)
    
#     # Reshape the labels to the original image shape
#     segmented_img = clustering.labels_.reshape(img.shape)
    
#     # Convert labels to binary image
#     segmented_img = (segmented_img == 1).astype(np.uint8) * 255
    
#     return segmented_img

# # Apply EADBSC segmentation
# img = cv2.imread('processed_image.png', 0)
# segmented_img = eadbsc_segmentation(img)

# # Overlay the segmented image on the original image
# overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.7, cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR), 0.3, 0)

# # Display the segmented image and overlay
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1), plt.title('Segmented Image'), plt.imshow(segmented_img, cmap='gray')
# plt.subplot(1, 2, 2), plt.title('Overlay on Original Image'), plt.imshow(overlay)
# plt.show()


# import cv2
# import numpy as np
# from sklearn.decomposition import PCA

# class RetinalImageEnhancer:
    
#     def __init__(self):
#         pass
    
#     def enhance_image(self, image):
#         # Convert image to LAB color space
#         lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab_image)
        
#         # Apply PCA to the L channel
#         pca = PCA(n_components=1)
#         l_flat = l.flatten().reshape(-1, 1)
#         l_pca = pca.fit_transform(l_flat)
#         l_pca = pca.inverse_transform(l_pca).reshape(l.shape)
        
#         # Normalize the enhanced L channel
#         l_pca = cv2.normalize(l_pca, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
#         # Merge the enhanced L channel back with A and B channels
#         enhanced_lab_image = cv2.merge((l_pca, a, b))
#         enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
        
#         return enhanced_image

# # Example usage
# if __name__ == "__main__":
#     image = cv2.imread('processed_image.png')
#     enhancer = RetinalImageEnhancer()
#     enhanced_image = enhancer.enhance_image(image)
#     cv2.imwrite('enhanced_fundus_image.png', enhanced_image)



import cv2
import numpy as np
from skimage import morphology
from skimage.segmentation import active_contour
from skimage.filters import gaussian

# class OpticDiscSegmenter:
    
#     def __init__(self):
#         pass
    
#     def segment_optic_disc(self, image):
#         # Convert image to grayscale
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Apply morphological operations
#         selem = morphology.disk(5)
#         morph_image = morphology.opening(gray_image, selem)
        
#         # Apply Gaussian filter
#         blurred_image = gaussian(morph_image, sigma=2)
        
#         # Initialize snake (active contour)
#         s = np.linspace(0, 2 * np.pi, 400)
#         x = 100 + 50 * np.cos(s)
#         y = 100 + 50 * np.sin(s)
#         init = np.array([x, y]).T
        
#         # Apply active contour model
#         snake = active_contour(blurred_image, init, alpha=0.015, beta=10, gamma=0.001)
        
#         return snake

# # Example usage
# if __name__ == "__main__":
#     image = cv2.imread('processed_image.png')
#     segmenter = OpticDiscSegmenter()
#     optic_disc_contour = segmenter.segment_optic_disc(image)
    
#     # Draw the contour on the image
#     for point in optic_disc_contour:
#         cv2.circle(image, (int(point[1]), int(point[0])), 1, (0, 255, 0), -1)
    
#     cv2.imwrite('segmented_optic_disc.png', image)

# import cv2
# import numpy as np

# class LocalNormalizer:
    
#     def __init__(self):
#         pass
    
#     def local_normalize(self, image):
#         # Convert to grayscale
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Apply local normalization
#         kernel = np.ones((15, 15), np.float32) / 225
#         local_mean = cv2.filter2D(gray_image, -1, kernel)
#         local_sqr_mean = cv2.filter2D(gray_image**2, -1, kernel)
#         local_variance = local_sqr_mean - local_mean**2
#         local_stddev = np.sqrt(local_variance)
        
#         # Avoid division by zero and handle small stddev values
#         local_stddev[local_stddev < 1e-5] = 1e-5
        
#         # Suppress overflow warnings
#         np.seterr(over='ignore')
        
#         normalized_image = (gray_image - local_mean) / local_stddev
#         if not np.any(normalized_image):
#             print("Warning: Empty image after local normalization. Skipping normalization.")
#             return image  # Return original image if empty
        
#         normalized_image = cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         return normalized_image
        

# def bpdfhe(image):
#     # Convert image to grayscale if it is not already
#     if len(image.shape) == 3:
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray_image = image
    
#     # Compute the histogram
#     hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    
#     # Compute the cumulative distribution function (CDF)
#     cdf = hist.cumsum()
#     cdf_normalized = cdf * hist.max() / cdf.max()
    
#     # Apply histogram equalization
#     cdf_m = np.ma.masked_equal(cdf, 0)
#     cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
#     cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
#     # Apply the CDF to the grayscale image
#     equalized_image = cdf[gray_image]
    
#     # Convert back to BGR
#     enhanced_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    
#     return enhanced_image

# image = cv2.imread('timg/IM000275.png')
# edges = bpdfhe(image)
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def segment_image(image_path):
#     # Load the grayscale image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Edge-Based Segmentation using Canny Edge Detector
#     edges = cv2.Canny(image, 100, 200)
    
#     # Region-Based Segmentation using Watershed Algorithm
#     # Thresholding to create binary image
#     _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Noise removal using morphological operations
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # Sure background area
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
    
#     # Marker labelling
#     _, markers = cv2.connectedComponents(sure_fg)
    
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers + 1
    
#     # Mark the region of unknown with zero
#     markers[unknown == 0] = 0
    
#     # Apply watershed
#     markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
#     image[markers == -1] = [255]
    
#     # Thresholding Segmentation
#     _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
#     # Display results
#     titles = ['Original Image', 'Canny Edges', 'Watershed', 'Thresholding']
#     images = [image, edges, markers, thresh]
    
#     for i in range(4):
#         plt.subplot(2, 2, i+1)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
    
#     plt.show()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def segment_image(image_path):
#     # Load the grayscale image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Edge-Based Segmentation using Canny Edge Detector
#     edges = cv2.Canny(image, 100, 200)
    
#     # Region-Based Segmentation using Watershed Algorithm
#     # Thresholding to create binary image
#     _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Noise removal using morphological operations
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # Sure background area
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
    
#     # Marker labelling
#     _, markers = cv2.connectedComponents(sure_fg)
    
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers + 1
    
#     # Mark the region of unknown with zero
#     markers[unknown == 0] = 0
    
#     # Apply watershed
#     markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
#     image[markers == -1] = [255]
    
#     # Thresholding Segmentation
#     _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
#     # Blood Vessel Segmentation using Image Enhancement
#     enhanced_image = cv2.equalizeHist(image)
    
#     # Tyler Coye Algorithm
#     # Convert to grayscale using PCA (simulated here as direct grayscale conversion)
#     pca_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
    
#     # Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(pca_image)
    
#     # Morphological Operations
#     morph_kernel = np.ones((5, 5), np.uint8)
#     morph_image = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, morph_kernel)
    
#     # Display results
#     titles = ['Original Image', 'Canny Edges', 'Watershed', 'Thresholding', 'Enhanced Image', 'Tyler Coye Algorithm', 'Morphological Operations']
#     images = [image, edges, markers, thresh, enhanced_image, cl1, morph_image]
    
#     for i in range(7):
#         plt.subplot(3, 3, i+1)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
    
#     plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
# from skimage.feature import greycomatrix, greycoprops, local_binary_pattern

# def extract_glcm_features(image):
#     # Compute GLCM
#     glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
#     # Extract GLCM properties
#     contrast = greycoprops(glcm, 'contrast')[0, 0]
#     dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
#     homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
#     energy = greycoprops(glcm, 'energy')[0, 0]
#     correlation = greycoprops(glcm, 'correlation')[0, 0]
    
#     return contrast, dissimilarity, homogeneity, energy, correlation

# def extract_lbp_features(image):
#     # Parameters for LBP
#     radius = 1
#     n_points = 8 * radius
    
#     # Compute LBP
#     lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
#     # Calculate the histogram of LBP
#     (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
#     # Normalize the histogram
#     hist = hist.astype("float")
#     hist /= (hist.sum() + 1e-6)
    
#     return hist

# def fpcm(image, n_clusters=2, m=2, eta=2, max_iter=100, error=1e-5):
#     # Initialize cluster centers
#     np.random.seed(0)
#     centers = np.random.choice(image.flatten(), n_clusters)
    
#     # Initialize membership and typicality matrices
#     U = np.random.dirichlet(np.ones(n_clusters), size=image.size).T
#     T = np.random.dirichlet(np.ones(n_clusters), size=image.size).T
    
#     for iteration in range(max_iter):
#         U_old = U.copy()
        
#         # Update cluster centers
#         for j in range(n_clusters):
#             numerator = np.sum((U[j] ** m + T[j] ** eta) * image.flatten())
#             denominator = np.sum(U[j] ** m + T[j] ** eta)
#             centers[j] = numerator / denominator
        
#         # Update membership and typicality
#         for i in range(image.size):
#             for j in range(n_clusters):
#                 dist = np.abs(image.flatten()[i] - centers[j])
#                 U[j, i] = 1 / np.sum([(dist / np.abs(image.flatten()[i] - centers[k])) ** (2 / (m - 1)) for k in range(n_clusters)])
#                 T[j, i] = 1 / (1 + (dist / np.abs(image.flatten()[i] - centers[j])) ** (2 / (eta - 1)))
        
#         # Check for convergence
#         if np.linalg.norm(U - U_old) < error:
#             break
    
#     # Assign pixels to clusters
#     segmented_image = np.argmax(U, axis=0).reshape(image.shape)
    
#     return segmented_image

# def segment_image(image_path):
#     # Load the grayscale image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Identify and eliminate the fovea
#     # Assuming the fovea is the darkest region in the center of the image
#     mask = np.zeros_like(image)
#     center = (image.shape[1] // 2, image.shape[0] // 2)
#     radius = 50  # Adjust the radius as needed
#     cv2.circle(mask, center, radius, (255, 255, 255), -1)
#     image_no_fovea = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
#     # Thresholding Segmentation to make the segmented lesion obvious
#     _, thresh = cv2.threshold(image_no_fovea, 127, 255, cv2.THRESH_BINARY)
    
#     # Blood Vessel Segmentation using FPCM
#     fpcm_segmented = fpcm(image_no_fovea)
    
#     # Tyler Coye Algorithm
#     # Convert to grayscale using PCA
#     pca = PCA(n_components=1)
#     pca_image = pca.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
    
#     # Normalize PCA image to 8-bit
#     pca_image = cv2.normalize(pca_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     # Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(pca_image)
    
#     # Morphological Operations
#     morph_kernel = np.ones((5, 5), np.uint8)
#     morph_image = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, morph_kernel)
    
#     # Extract GLCM features
#     # glcm_features = extract_glcm_features(morph_image)
#     # print("GLCM Features:", glcm_features)
    
#     # # Extract LBP features
#     # lbp_features = extract_lbp_features(morph_image)
#     # print("LBP Features:", lbp_features)
    
#     # Display results
#     titles = ['Original Image', 'Image without Fovea', 'Thresholding', 'FPCM Segmentation', 'Tyler Coye Algorithm', 'Morphological Operations']
#     images = [image, image_no_fovea, thresh, fpcm_segmented, cl1, morph_image]
    
#     for i in range(6):
#         plt.subplot(3, 2, i+1)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
    
#     plt.show()



import cv2
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
# from keras.models import load_model
import torch

# Load pre-trained U-Net model for fovea and lesion detection
# Load the PyTorch model
# unet_model = torch.load('trained_model/R2U-Net.pth')
# unet_model.eval()  # Set the model to evaluation mode


# Instantiate the model
unet_model = UNet()

# Load the state dictionary
state_dict = torch.load('trained_model/U-Net.pth')
# unet_model.load_state_dict(state_dict)

# Set the model to evaluation mode
unet_model.eval()

def extract_glcm_features(image):
    # Compute GLCM
    glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract GLCM properties
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    
    return contrast, dissimilarity, homogeneity, energy, correlation

def extract_lbp_features(image):
    # Parameters for LBP
    radius = 1
    n_points = 8 * radius
    
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Calculate the histogram of LBP
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    return hist


def fpcm(image_cp, centers, m, n_clusters):
    U = cp.zeros((image_cp.shape[0], n_clusters))
    for i in range(image_cp.shape[0]):
        for j in range(n_clusters):
            dist = cp.abs(image_cp[i] - centers[j])
            U[j, i] = 1 / cp.sum(cp.array([(dist / cp.abs(image_cp[i] - centers[k])) ** (2 / (m - 1)) for k in range(n_clusters)]))
    return U

# def segment_image(image_path):
    
    
#     # Load the grayscale image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Detect and eliminate the fovea using U-Net
#     input_image = cv2.resize(image, (256, 256)) / 255.0
#     input_image = np.expand_dims(input_image, axis=0)
#     input_image = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0)
    
#     # Predict using the U-Net model
#     unet_model.eval()
#     with torch.no_grad():
#         prediction = unet_model(input_image)[0].numpy()
#         print("Prediction shape:", prediction.shape)
#         print("input_image shape:", input_image.shape)
#         print("image shape:", image.shape)
#         print("prediction.shape[0]:", prediction.shape[0])
#         print("prediction.shape[1]:", prediction.shape[1])
        
        
#     if prediction.shape[0] > 1:
#         lesion_mask = cv2.resize(prediction[1], (image.shape[1], image.shape[0]))
#     else:
#         print("Error: Prediction does not have enough elements.")
#         return

    
#     fovea_mask = cv2.resize(prediction[0], (image.shape[1], image.shape[0]))
#     lesion_mask = cv2.resize(prediction[1], (image.shape[1], image.shape[0]))
    
    
#     fovea_coords = np.where(fovea_mask > 0.5)
#     if len(fovea_coords[0]) > 0:
#         fovea_x, fovea_y = int(np.mean(fovea_coords[1])), int(np.mean(fovea_coords[0]))
#         mask = np.zeros_like(image)
#         radius = 50  # Adjust the radius as needed
#         cv2.circle(mask, (fovea_x, fovea_y), radius, (255, 255, 255), -1)
#         image_no_fovea = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
#     else:
#         image_no_fovea = image
    
#     # Apply lesion mask to the original image
#     lesion_mask = (lesion_mask > 0.5).astype(np.uint8) * 255
#     image_with_lesions = cv2.bitwise_and(image, image, mask=lesion_mask)
    
#     # Blood Vessel Segmentation using FPCM
#     # fpcm_segmented = fpcm(image_no_fovea)
    
#     # Tyler Coye Algorithm
#     # Convert to grayscale using PCA
#     pca = PCA(n_components=1)
#     pca_image = pca.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
    
#     # Normalize PCA image to 8-bit
#     pca_image = cv2.normalize(pca_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
#     # Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(pca_image)
    
#     # Morphological Operations
#     morph_kernel = np.ones((5, 5), np.uint8)
#     morph_image = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, morph_kernel)
    
#     # Extract GLCM features
#     glcm_features = extract_glcm_features(morph_image)
#     print("GLCM Features:", glcm_features)
    
#     # Extract LBP features
#     lbp_features = extract_lbp_features(morph_image)
#     print("LBP Features:", lbp_features)
    
#     # Display results
#     titles = ['Original Image', 'Image without Fovea', 'Lesion Detection', 'FPCM Segmentation', 'Tyler Coye Algorithm', 'Morphological Operations']
#     images = [image, image_no_fovea, image_with_lesions, image_with_lesions, cl1, morph_image]
    
#     for i in range(6):
#         plt.subplot(3, 2, i+1)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
    
#     plt.show()

# # Example usage
# segment_image('timg/IMG0052.png')
# ------------------------------------------











# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def segment_image(image_path):
#     # Load the grayscale image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Edge-Based Segmentation using Canny Edge Detector
#     edges = cv2.Canny(image, 100, 200)
    
#     # Region-Based Segmentation using Watershed Algorithm
#     # Thresholding to create binary image
#     _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Noise removal using morphological operations
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # Sure background area
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
    
#     # Marker labelling
#     _, markers = cv2.connectedComponents(sure_fg)
    
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers + 1
    
#     # Mark the region of unknown with zero
#     markers[unknown == 0] = 0
    
#     # Apply watershed
#     markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
#     image[markers == -1] = [255]
    
#     # Thresholding Segmentation
#     _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
#     # Blood Vessel Segmentation using Image Enhancement
#     enhanced_image = cv2.equalizeHist(image)
    
#     # Tyler Coye Algorithm
#     # Convert to grayscale using PCA (simulated here as direct grayscale conversion)
#     pca_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
    
#     # Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl1 = clahe.apply(pca_image)
    
#     # Morphological Operations
#     morph_kernel = np.ones((5, 5), np.uint8)
#     morph_image = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, morph_kernel)
    
#     # Display results
#     titles = ['Original Image', 'Canny Edges', 'Watershed', 'Thresholding', 'Enhanced Image', 'Tyler Coye Algorithm', 'Morphological Operations']
#     images = [image, edges, markers, thresh, enhanced_image, cl1, morph_image]
    
#     for i in range(7):
#         plt.subplot(3, 3, i+1)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
    
#     plt.show()

# # Example usage
# segment_image('timg/IMG0052.png')



import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# def segment_image(image_path):
#     # Load the grayscale image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Edge-Based Segmentation using Canny Edge Detector
#     edges = cv2.Canny(image, 50, 100)
    
#     # Enhance the edges by increasing the contrast
#     enhanced_edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    
#     # Overlay enhanced edges on the original image
#     overlay = cv2.addWeighted(image, 0.7, enhanced_edges, 0.3, 0)
    
#     # Apply PCA on the overlay image
#     overlay_flat = overlay.reshape(-1, 1)  # Flatten the image
#     pca = PCA(n_components=1)
#     overlay_pca = pca.fit_transform(overlay_flat)
#     overlay_pca = pca.inverse_transform(overlay_pca)
#     overlay_pca = overlay_pca.reshape(overlay.shape).astype(np.uint8)
    
#     # Stretch the contrast
#     enhanced_overlay = cv2.normalize(overlay_pca, None, 0, 255, cv2.NORM_MINMAX)
    
#     # Region-Based Segmentation using Watershed Algorithm
#     # Thresholding to create binary image
#     _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Noise removal using morphological operations
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
#     # Sure background area
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
#     # Finding sure foreground area
#     dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#     _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
#     # Finding unknown region
#     sure_fg = np.uint8(sure_fg)
#     unknown = cv2.subtract(sure_bg, sure_fg)
    
#     # Marker labelling
#     _, markers = cv2.connectedComponents(sure_fg)
    
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers + 1
    
#     # Mark the region of unknown with zero
#     markers[unknown == 0] = 0
    
#     # Apply watershed
#     markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
#     image[markers == -1] = [255]
    
#     # Thresholding Segmentation
#     _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
#     # Display results
#     titles = ['Original Image', 'Canny Edges', 'Enhanced Edges', 'Overlay', 'PCA Overlay', 'Enhanced Overlay', 'Watershed', 'Thresholding']
#     images = [image, edges, enhanced_edges, overlay, overlay_pca, enhanced_overlay, markers, thresh]
    
#     for i in range(8):
#         plt.subplot(4, 2, i+1)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
    
#     plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

# def segment_image(image_path):
#     # Load the grayscale image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Edge-Based Segmentation using Canny Edge Detector
#     edges = cv2.Canny(image, 50, 150)
    
#     # Enhance the edges using adaptive histogram equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_edges = clahe.apply(edges)
    
#     # Overlay enhanced edges on the original image with increased intensity
#     overlay = cv2.addWeighted(image, 0.5, enhanced_edges, 0.5, 0)
    
#     # Increase the contrast of the overlay
#     contrast_overlay = cv2.convertScaleAbs(overlay, alpha=1.5, beta=0)
    
#     # Apply PCA on the contrast-enhanced overlay image
#     overlay_flat = contrast_overlay.reshape(-1, 1)  # Flatten the image
#     pca = PCA(n_components=1)
#     overlay_pca = pca.fit_transform(overlay_flat)
#     overlay_pca = pca.inverse_transform(overlay_pca)
#     overlay_pca = overlay_pca.reshape(contrast_overlay.shape).astype(np.uint8)
    
#     # Stretch the contrast
#     enhanced_overlay = cv2.normalize(overlay_pca, None, 0, 255, cv2.NORM_MINMAX)
    
#     # Display results
#     titles = ['Original Image', 'Canny Edges', 'Enhanced Edges', 'Overlay', 'Contrast Overlay', 'Enhanced Overlay']
#     images = [image, edges, enhanced_edges, overlay, contrast_overlay, enhanced_overlay]
    
#     for i in range(6):
#         plt.subplot(3, 2, i+1)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([]), plt.yticks([])
    
#     plt.show()

# # Example usage
# segment_image('timg/IMG0052.png')



def segment_image(image_path):
    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Identify and eliminate the fovea
    # Assuming the fovea is the darkest region in the center of the image
    mask = np.zeros_like(image)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = 50  # Adjust the radius as needed
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    image_no_fovea = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    
    # Thresholding Segmentation to make the segmented lesion obvious
    _, thresh = cv2.threshold(image_no_fovea, 127, 255, cv2.THRESH_BINARY)
    
    # Blood Vessel Segmentation using FPCM
    # fpcm_segmented = fpcm(image_no_fovea)
    
    # Tyler Coye Algorithm
    # Convert to grayscale using PCA
    pca = PCA(n_components=1)
    pca_image = pca.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
    
    # Normalize PCA image to 8-bit
    pca_image = cv2.normalize(pca_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(pca_image)
    
    # Morphological Operations
    morph_kernel = np.ones((5, 5), np.uint8)
    morph_image = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, morph_kernel)
    
    
    print(type(morph_image))
    
    # Extract GLCM features
    glcm_features = extract_glcm_features(morph_image)
    print("GLCM Features:", glcm_features)
    
    # Extract LBP features
    lbp_features = extract_lbp_features(morph_image)
    print("LBP Features:", lbp_features)
    
    # Display results
    titles = ['Original Image', 'Image without Fovea', 'Thresholding', 'FPCM Segmentation', 'Tyler Coye Algorithm', 'Morphological Operations']
    images = [image, image_no_fovea, thresh, thresh, cl1, morph_image]
    
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.show()


# Example usage
# segment_image('timg/IMG0052.png')



import os
import pandas as pd

def check_csv_filenames(csv_file, image_dir):
    """
    Checks if filenames in the CSV correspond to actual files in the image directory.
    Prints lines from the CSV where the filename is missing.

    Args:
        csv_file (str): Path to the CSV file.
        image_dir (str): Path to the image directory.

    Returns:
        None
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    # Get the filenames from the CSV (assuming 'ID' is the column name)
    csv_filenames = df['ID'].tolist()

    # Check if each filename exists in the image directory
    for filename in csv_filenames:
        filename = filename +'.png'
        print(filename)
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"Missing file: {filename} (line {csv_filenames.index(filename) + 1})")
            
            
# csv_file_path = "dataset/v2/label/train_label.csv"
# image_directory_path = "dataset/v2/images/training"
# check_csv_filenames(csv_file_path, image_directory_path)


# def find_duplicate_ids(csv_filename):
#     """
#     Finds and returns rows with duplicate IDs from a CSV file.

#     Args:
#         csv_filename (str): Path to the CSV file.

#     Returns:
#         pd.DataFrame: DataFrame containing duplicate rows based on the "ID" column.
#     """
#     try:
#         df = pd.read_csv(csv_filename)
#     except FileNotFoundError:
#         print(f"Error: CSV file '{csv_filename}' not found.")
#         return None

#     # Check for duplicate IDs
#     duplicate_rows = df[df.duplicated(subset='ID', keep=False)]

#     return duplicate_rows

# # Example usage:
# csv_file_path = "dataset/v2/label/train_label.csv"
# duplicate_rows = find_duplicate_ids(csv_file_path)

# if duplicate_rows is not None:
#     print("Duplicate rows based on 'ID':")
#     print(duplicate_rows)
# else:
#     print("No duplicate rows found.")
    
    
import proj_util
def remove_duplicates_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Remove duplicate rows based on the 'ID' column
    df_cleaned = df.drop_duplicates(subset='ID')
    
    # Save the cleaned DataFrame back to the CSV file
    df_cleaned.to_csv(file_path, index=False)
    
    print(f"Removed duplicates based on 'ID' from {file_path}")

# Example usage
remove_duplicates_from_csv(proj_util.TRAIN_LABEL_PATH)



# import matplotlib.pyplot as plt
# from PIL import Image
# import torchvision.transforms as transforms

# # Load an example image
# image = Image.open('timg/10.png')

# # Define the transformations
# transform1 = transforms.ColorJitter(brightness=0.05, contrast=0.02, saturation=0.2, hue=0.1)
# transform2 = transforms.ColorJitter(brightness=0.05)

# # Apply the transformations
# image1 = transform1(image)
# image2 = transform2(image)

# # Display the original and transformed images
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].imshow(image)
# axs[0].set_title('Original Image')
# axs[1].imshow(image1)
# axs[1].set_title('ColorJitter (All)')
# axs[2].imshow(image2)
# axs[2].set_title('ColorJitter (Brightness)')
# plt.show()