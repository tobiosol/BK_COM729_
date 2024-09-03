import cupy as cp
import cv2
import numpy as np
from cupyx.scipy.ndimage import median_filter

class ImageProcessor:
    def __init__(self, image):
        self.image = cp.asarray(image, dtype=cp.float32)
        self.processed_image = None

    def extract_green_layer(self):
        if len(self.image.shape) == 3:
            self.image = self.image[:, :, 1]
        return self.image
    
    def resize_image(self):
        # Resize the image by dividing both sides by 2
        new_size = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        resized_image = cv2.resize(cp.asnumpy(self.image), new_size, interpolation=cv2.INTER_AREA)
        self.image = cp.asarray(resized_image)
        return self.image

    # def apply_clahe(self):
    #     # Step II: Green layer selected
    #     self.extract_green_layer()
    #     print("complete extract_green_layer")

    #     # Ensure the image is in the correct format
    #     if self.image.dtype != cp.uint8:
    #         self.image = cp.clip(self.image * (255.0 / cp.max(self.image)), 0, 255).astype(cp.uint8)

    #     # Step III: Decimate the image into 2x2 sized contextual regions (Tiles)
    #     tile_size = (2, 2)
    #     tiles = self.image.reshape(-1, tile_size[0], tile_size[1])

    #     # Convert tiles to CuPy arrays before processing
    #     tiles_gpu = [cp.asarray(tile) for tile in tiles]

    #     # Step IV: Apply image processing to each tile in parallel
    #     def process_tile(tile):
            
    #         tile_np = cp.asnumpy(tile)
    #         tile_clahe = cv2.equalizeHist(tile_np)
                        
    #         # Contrast enhancement (CLAHE)
    #         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #         tile_clahe = clahe.apply(cp.asnumpy(tile))

    #         # Noise reduction (Gaussian blur)
    #         tile_blurred = cv2.GaussianBlur(tile_clahe, (3, 3), 0)

    #         # Sharpening (unsharp masking)
    #         kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #         tile_sharpened = cv2.filter2D(tile_blurred, -1, kernel)

    #         return cp.array(tile_sharpened)

    #     # Process each tile individually
    #     processed_tiles = [process_tile(tile) for tile in tiles_gpu]

    #     # Reshape and concatenate tiles back to original image size
    #     self.processed_image = cp.concatenate(processed_tiles).reshape(self.image.shape[0], self.image.shape[1])

    #     return self.processed_image

        
        
        
    def apply_clahe(self):
        # Step II: Green layer selected
        self.extract_green_layer()
        print("complete extract_green_layer")
    
        # Ensure the image is in the correct format
        if self.image.dtype != cp.uint8:
            self.image = cp.clip(self.image * (255.0 / cp.max(self.image)), 0, 255).astype(cp.uint8)
    
        # Get the dimensions of the image
        height, width = self.image.shape
    
        # Ensure the dimensions are divisible by 2
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Image dimensions must be divisible by 2 for 2x2 tiling.")
    
        # Step III: Decimate the image into 2x2 sized contextual regions (Tiles)
        tiles = []
        for i in range(0, height, 2):
            for j in range(0, width, 2):
                tile = self.image[i:i+2, j:j+2]
                tiles.append(tile)
    
        # Convert tiles to CuPy arrays before processing
        tiles_gpu = [cp.asarray(tile) for tile in tiles]
    
        # Step IV: Apply image processing to each tile in parallel
        def process_tile(tile):
            # Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            tile_clahe = clahe.apply(cp.asnumpy(tile))
            
            # Noise reduction (Gaussian blur)
            tile_blurred = cv2.GaussianBlur(tile_clahe, (3, 3), 0)
    
            # Sharpening (unsharp masking)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Ensure kernel is a NumPy array
            tile_sharpened = cv2.filter2D(tile_blurred, -1, kernel)
    
            return cp.array(tile_sharpened)
    
        # Process each tile individually
        processed_tiles = [process_tile(tile) for tile in tiles_gpu]
    
        # Reshape and concatenate tiles back to original image size
        processed_image = cp.zeros_like(self.image)
        idx = 0
        for i in range(0, height, 2):
            for j in range(0, width, 2):
                processed_image[i:i+2, j:j+2] = processed_tiles[idx]
                idx += 1
    
        self.processed_image = processed_image
        return self.processed_image
        
        
        
        
        
        

    def apply_median_filter(self):
        self.image = cp.asarray(cv2.medianBlur(cp.asnumpy(self.image), 5))
        return self.image

    
    def eliminate_large_objects(self):
        if self.image is None:
            print("Error: self.image is None")
            return None
    
        if self.image.dtype != cp.uint8:
            self.image = cp.clip(self.image * (255.0 / cp.max(self.image)), 0, 255).astype(cp.uint8)
    
        # Convert the image to a NumPy array for OpenCV processing
        image_np = cp.asnumpy(self.image)
    
        # Ensure the image is in the correct format
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
    
        # Apply Canny edge detection
        edges = cv2.Canny(image_np, 100, 200)
    
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Draw contours to eliminate large objects
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                cv2.drawContours(image_np, [contour], -1, (0, 0, 0), -1)
    
        # Convert the result back to CuPy array
        self.processed_image = cp.asarray(image_np)
    
        return self.processed_image
    
    

    # def eliminate_large_objects(self):
    #     # Ensure the image is in the correct format
    #     if self.processed_image.dtype != cp.uint8:
    #         self.processed_image = cp.clip(self.processed_image * (255.0 / cp.max(self.processed_image)), 0, 255).astype(cp.uint8)
    
    #     # Convert the image to a NumPy array for OpenCV processing
    #     image_np = cp.asnumpy(self.processed_image)
    
    #     # Apply Canny edge detection
    #     edges = cv2.Canny(image_np, 100, 200)
    
    #     # Convert the result back to a CuPy array
    #     edges_gpu = cp.asarray(edges)
    
    #     return edges_gpu

    
    
    def remove_blood_vessels(self):
        kernel = cp.ones((5, 5), cp.uint8)
        kernel_np = cp.asnumpy(kernel)  # Convert CuPy array to NumPy array
        blood_vessels = cp.asarray(cv2.morphologyEx(cp.asnumpy(self.image), cv2.MORPH_TOPHAT, kernel_np))
        self.image = cp.subtract(self.image, blood_vessels)
        return self.image
    
        
    
    
    
    def remove_optic_disk(self):
        # Ensure the image is in grayscale and of type CV_8UC1
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(cp.asnumpy(self.image), cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cp.asnumpy(self.image)
    
        gray_image = gray_image.astype(np.uint8)  # Ensure the image is a NumPy array of type uint8
    
        # Apply HoughCircles to detect circles
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=100)
    
        mask = cp.zeros(self.image.shape[:2], dtype=cp.uint8)
    
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
        self.image = self.image * (inverted_mask / 255)
    
        return self.image
    
    def enhance_image(self):
        
        # Ensure the image is a CuPy array
        if not isinstance(self.image, cp.ndarray):
            raise TypeError("Image is not a valid CuPy array")

        # Convert CuPy array to NumPy array
        image_np = cp.asnumpy(self.image)

        # Define the sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        # Apply the sharpening filter
        enhanced_image_np = cv2.filter2D(image_np, -1, kernel)

        # Convert the enhanced image back to CuPy array
        enhanced_image_cp = cp.asarray(enhanced_image_np)

        return enhanced_image_cp

    
    

    def detect_microaneurysms(self):
        print(type(self.image))
        if self.image.dtype != cp.uint8:
            self.image = cp.clip(self.image * (255.0 / cp.max(self.image)), 0, 255).astype(cp.uint8)

        # Convert the image to a NumPy array for OpenCV processing
        image_np = cp.asnumpy(self.image)

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
    

    def preprocess_image(self):
        self.image = self.resize_image()
        self.image = self.extract_green_layer()
        # self.image = self.eliminate_large_objects()
        # self.image = self.apply_clahe()
        self.image = self.apply_median_filter()
        self.image = self.remove_blood_vessels()
        self.image = self.remove_optic_disk()        
        
        self.image = self.enhance_image()
        self.processed_image = self.detect_microaneurysms()
        
        return self.processed_image



# Load image
image = cv2.imread('IM000275.png')
# Create an instance of ImageProcessor
processor = ImageProcessor(image)
# Preprocess image
processed_image = processor.preprocess_image()
# Compute CII and Entropy
cii = processor.compute_cii(image, processed_image)
entropy = processor.compute_entropy(processed_image)

print(f'CII: {cii}')
print(f'Entropy: {entropy}')



# CII: 0.39811673760414124
# Entropy: 2.2644196274091755







"""
    def apply_clahe(self):
        # Step II: Green layer selected
        self.extract_green_layer()
        print("complete extract_green_layer")
        
        # Ensure the image is in the correct format
        if self.image.dtype != cp.uint8:
            self.image = cp.clip(self.image * (255.0 / cp.max(self.image)), 0, 255).astype(cp.uint8)
        
        # Step III: Decimate the image into 2x2 sized contextual regions (Tiles)
        tile_size = (2, 2)
        tiles = [self.image[i:i+tile_size[0], j:j+tile_size[1]] for i in range(0, self.image.shape[0], tile_size[0]) for j in range(0, self.image.shape[1], tile_size[1])]
        print("complete Decimate the image into 2x2 sized contextual regions (Tiles)")
        
        # Step IV: Apply CLAHE in each tile separately
        tiles_cpu = [tile.get() for tile in tiles]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tiles_clahe = [clahe.apply(tile) for tile in tiles_cpu]
        print("complete Apply CLAHE in each tile separately")
        
        # # Step V: Apply median filter in each tile separately
        # tiles_filtered = [median_filter(tile, size=5) for tile in tiles_clahe]
        # print("complete Apply median filter in each tile separately")
        
        tiles_clahe_gpu = [cp.asarray(tile) for tile in tiles_clahe]
        rows = [cp.concatenate(tiles_clahe_gpu[i:i+int(self.image.shape[1]/tile_size[1])], axis=1) for i in range(0, len(tiles_clahe_gpu), int(self.image.shape[1]/tile_size[1]))]

        print("complete Apply image concatenation method on four Tiles")
        self.processed_image = cp.concatenate(rows, axis=0)
        
        return self.processed_image
"""

"""
def apply_clahe(self):
        # Step II: Green layer selected
        self.extract_green_layer()
        print("complete extract_green_layer")
        
        # Ensure the image is in the correct format
        if self.image.dtype != cp.uint8:
            self.image = cp.clip(self.image * (255.0 / cp.max(self.image)), 0, 255).astype(cp.uint8)
        
        # Step III: Decimate the image into 2x2 sized contextual regions (Tiles)
        tile_size = (2, 2)
        tiles = [self.image[i:i+tile_size[0], j:j+tile_size[1]] for i in range(0, self.image.shape[0], tile_size[0]) for j in range(0, self.image.shape[1], tile_size[1])]
        print("complete Decimate the image into 2x2 sized contextual regions (Tiles)")
        
        processed_tiles = []
        for tile in tiles:
            # Convert to NumPy for OpenCV compatibility
            tile_np = cp.asnumpy(tile)
    
            # Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            tile_clahe = clahe.apply(tile_np)
    
            # Noise reduction (Gaussian blur)
            tile_blurred = cv2.GaussianBlur(tile_clahe, (3, 3), 0)
    
            # Sharpening (unsharp masking)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            tile_sharpened = cv2.filter2D(tile_blurred, -1, kernel)
    
            # Convert back to CuPy array
            processed_tiles.append(cp.array(tile_sharpened))
        
        
        
        
        # # Step IV: Apply CLAHE in each tile separately
        # tiles_cpu = [tile.get() for tile in tiles]
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # tiles_clahe = [clahe.apply(tile) for tile in tiles_cpu]
        # print("complete Apply CLAHE in each tile separately")
        
        # # Step V: Apply median filter in each tile separately
        # tiles_filtered = [median_filter(tile, size=5) for tile in tiles_clahe]
        # print("complete Apply median filter in each tile separately")
        
        tiles_clahe_gpu = [cp.asarray(tile) for tile in processed_tiles]
        rows = [cp.concatenate(tiles_clahe_gpu[i:i+int(self.image.shape[1]/tile_size[1])], axis=1) for i in range(0, len(tiles_clahe_gpu), int(self.image.shape[1]/tile_size[1]))]

        print("complete Apply image concatenation method on four Tiles")
        self.processed_image = cp.concatenate(rows, axis=0)
        
        return self.processed_image
"""
