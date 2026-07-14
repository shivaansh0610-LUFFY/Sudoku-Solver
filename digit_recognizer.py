import cv2
import numpy as np

def clear_cell_borders(gray_image, border_width=20):
    """
    Clears the outer border of a grayscale image by setting it to white background (255).
    This is extremely effective for removing grid line artifacts.
    Using border_width=20 is the optimal sweet spot to clear grid borders without clipping shifted digits.
    """
    img = gray_image.copy()
    h, w = img.shape[:2]
    img[0:border_width, :] = 255
    img[h-border_width:h, :] = 255
    img[:, 0:border_width] = 255
    img[:, w-border_width:w] = 255
    return img

def keep_largest_component(binary_image):
    """
    Finds the largest connected component in the binary image and discards all others.
    Assumes background is 255 (white) and foreground (digit) is 0 (black).
    This completely cleans up any isolated noise, split serifs, or grid remnants.
    """
    # Invert binary image so foreground is 1 and background is 0
    fg = (binary_image == 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    
    if num_labels <= 1:
        # Only background found
        return binary_image.copy()
        
    # Stats format: [left, top, width, height, area]
    # Label 0 is background, so we slice from index 1 to ignore it
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1 # +1 because of slicing
    
    # Create white canvas and draw only the largest component
    cleaned = np.ones_like(binary_image) * 255
    cleaned[labels == largest_label] = 0
    
    return cleaned

def center_digit(binary_image, target_size=32):
    """
    Crops a binary digit to its bounding box, scales it to fit within a 24x24 box
    preserving aspect ratio, and centers it on a 32x32 white canvas.
    Assumes background is 255 (white) and digit is 0 (black).
    """
    # Find black pixels
    y_indices, x_indices = np.where(binary_image == 0)
    
    if len(y_indices) == 0:
        # No digit found, return a pure white canvas
        return np.ones((target_size, target_size), dtype=np.uint8) * 255
        
    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()
    
    # Crop to bounding box
    digit_crop = binary_image[ymin:ymax+1, xmin:xmax+1]
    
    # Determine target scale to fit inside 24x24 (leaves 4px margin on all sides)
    box_size = int(target_size * 0.75)
    h_crop, w_crop = digit_crop.shape
    
    scale = box_size / max(h_crop, w_crop)
    new_h = max(1, int(h_crop * scale))
    new_w = max(1, int(w_crop * scale))
    
    # Resize keeping aspect ratio
    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create white canvas and center the digit
    canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
    dy = (target_size - new_h) // 2
    dx = (target_size - new_w) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized_digit
    
    return canvas
