import cv2
import numpy as np

def detect_blur(image_path, threshold=100):
    """
    Detect if an image is blurry using the Laplacian variance method.
    
    Args:
        image_path (str): Path to the input image
        threshold (float): Threshold value to determine blur (lower means more sensitive)
    
    Returns:
        tuple: (is_blurry, laplacian_variance)
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if image exists
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the Laplacian of the image and compute the variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = laplacian.var()
    
    # Determine if the image is blurry based on the variance
    is_blurry = laplacian_variance < threshold
    
    return is_blurry, laplacian_variance

def preprocess_overexposed_image(image_path):
    """
    Preprocess an overexposed image to balance colors and improve readability.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the processed image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Convert to LAB color space (better for color adjustments)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with the a and b channels
    enhanced_lab = cv2.merge((l_clahe, a, b))
    
    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_img


# Example usage
if __name__ == "__main__":
    # Path to your image
    image_path = "blur_image.png"
    
    # Detect blur
    is_blurry, score = detect_blur(image_path)
    
    # Print results
    print(f"Image: {image_path}")
    print(f"Laplacian variance: {score:.2f}")
    print(f"Blur detection: {'Blurry' if is_blurry else 'Not blurry'}")