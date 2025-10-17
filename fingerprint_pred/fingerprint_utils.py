import cv2
import numpy as np
from PIL import Image
import logging
import os

logger = logging.getLogger(__name__)

class FingerprintProcessor:
    """Class to handle fingerprint image processing and feature extraction"""
    
    def __init__(self):
        self.min_ridge_count = 5
        self.max_ridge_count = 200
    
    def preprocess_image(self, image_path):
        """
        Preprocess fingerprint image for better feature extraction
        """
        try:
            # Read image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            # Resize image if too large
            height, width = img.shape
            if width > 800 or height > 600:
                scale = min(800/width, 600/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            # Apply Gaussian blur to reduce noise
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
            # Apply histogram equalization
            img = cv2.equalizeHist(img)
            
            return img
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def detect_fingerprint_pattern(self, img):
        """
        Detect fingerprint pattern type (Loop, Whorl, Arch)
        This is a simplified implementation - in production, you'd use more sophisticated algorithms
        """
        try:
            if img is None:
                return "Loop"  # Default
            
            # Apply Sobel edge detection
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude and direction
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(sobely, sobelx)
            
            # Simple pattern classification based on gradient analysis
            # This is a basic heuristic - real fingerprint classification is much more complex
            
            # Calculate some basic statistics
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            center_region = img[center_y-50:center_y+50, center_x-50:center_x+50]
            
            if center_region.size == 0:
                return "Loop"
            
            # Analyze variance in center region
            variance = np.var(center_region)
            mean_magnitude = np.mean(magnitude)
            
            # Simple classification rules (this would be much more sophisticated in reality)
            if variance > 800 and mean_magnitude > 30:
                return "Whorl"
            elif variance < 300:
                return "Arch"
            else:
                return "Loop"
                
        except Exception as e:
            logger.error(f"Error detecting fingerprint pattern: {str(e)}")
            return "Loop"  # Default fallback
    
    def estimate_ridge_count(self, img):
        """
        Estimate ridge count using image processing techniques
        This is a simplified implementation
        """
        try:
            if img is None:
                return 50  # Default value
            
            # Apply binary threshold
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Get image center
            height, width = img.shape
            center_x, center_y = width // 2, height // 2
            
            # Define multiple scanning lines from center
            ridge_counts = []
            angles = np.linspace(0, 2*np.pi, 8)  # 8 directions
            
            for angle in angles:
                # Calculate end point of scanning line
                radius = min(width, height) // 4
                end_x = int(center_x + radius * np.cos(angle))
                end_y = int(center_y + radius * np.sin(angle))
                
                # Ensure end point is within image bounds
                end_x = max(0, min(width-1, end_x))
                end_y = max(0, min(height-1, end_y))
                
                # Extract pixel values along the line
                line_length = int(np.sqrt((end_x - center_x)**2 + (end_y - center_y)**2))
                if line_length > 0:
                    x_coords = np.linspace(center_x, end_x, line_length).astype(int)
                    y_coords = np.linspace(center_y, end_y, line_length).astype(int)
                    
                    # Get pixel values along the line
                    line_pixels = binary[y_coords, x_coords]
                    
                    # Count transitions (ridges)
                    transitions = np.sum(np.diff(line_pixels.astype(int)) != 0)
                    ridge_count = transitions // 2  # Each ridge has 2 transitions
                    ridge_counts.append(ridge_count)
            
            # Calculate average ridge count
            if ridge_counts:
                avg_ridge_count = int(np.mean(ridge_counts))
                # Ensure it's within reasonable bounds
                return max(self.min_ridge_count, min(self.max_ridge_count, avg_ridge_count))
            else:
                return 50  # Default
                
        except Exception as e:
            logger.error(f"Error estimating ridge count: {str(e)}")
            return 50  # Default fallback
    
    def extract_features(self, image_path):
        """
        Main method to extract fingerprint features
        Returns dict with extracted features
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            if processed_img is None:
                logger.warning(f"Could not process image: {image_path}")
                return {
                    'fingerprint_type': 'Loop',
                    'ridge_count': 50,
                    'success': False,
                    'error': 'Image preprocessing failed'
                }
            
            # Extract features
            fingerprint_type = self.detect_fingerprint_pattern(processed_img)
            ridge_count = self.estimate_ridge_count(processed_img)
            
            logger.info(f"Extracted features - Type: {fingerprint_type}, Ridge Count: {ridge_count}")
            
            return {
                'fingerprint_type': fingerprint_type,
                'ridge_count': ridge_count,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {
                'fingerprint_type': 'Loop',
                'ridge_count': 50,
                'success': False,
                'error': str(e)
            }
    
    def validate_fingerprint_image(self, image_path):
        """
        Validate if the uploaded image contains a fingerprint
        Basic validation - checks for image quality and patterns
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "Invalid image file"
            
            # Check image size
            height, width = img.shape
            if width < 100 or height < 100:
                return False, "Image too small (minimum 100x100 pixels)"
            
            # Check if image has enough contrast/variation
            variance = np.var(img)
            if variance < 50:
                return False, "Image lacks sufficient detail or contrast"
            
            # Check for completely black or white images
            mean_intensity = np.mean(img)
            if mean_intensity < 10 or mean_intensity > 245:
                return False, "Image is too dark or too bright"
            
            return True, "Valid fingerprint image"
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
