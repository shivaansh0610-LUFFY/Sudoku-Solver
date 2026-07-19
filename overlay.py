import os
import cv2
import numpy as np

def detect_corners(original_img):
    """
    Detects the corners of the Sudoku grid on the original image.
    
    ===========================================================================
    IMPORTANT WARNING / CODE DUPLICATION NOTICE:
    This function duplicates the preprocessing and contour detection logic from 
    grid_extractor.py because extract_cells() only returns the cells and not the
    detected corner points. 
    
    If the detection logic in grid_extractor.py is modified or tuned in the 
    future, those changes MUST be manually mirrored here.
    ===========================================================================
    """
    h, w = original_img.shape[:2]
    max_dim = max(h, w)
    
    # 1. Resize to max_dim 1000 for detection (matching grid_extractor.py behavior)
    scale = 1.0
    img_detect = original_img.copy()
    if max_dim > 1000:
        scale = 1000.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img_detect = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    # Convert to grayscale
    gray = cv2.cvtColor(img_detect, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to handle uneven lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological dilation to close small gaps in the grid lines
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area_threshold = 0.025 * (img_detect.shape[0] * img_detect.shape[1])
    
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_threshold:
            continue
            
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            candidates.append((area, approx))
            
    # Sort candidates by area descending
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    grid_contour = None
    
    if candidates:
        best_contour = candidates[0][1]
        best_area = candidates[0][0]
        
        for area, approx in reversed(candidates):
            if area >= 0.6 * best_area:
                grid_contour = approx
                break
                
    if grid_contour is None:
        raise ValueError(
            f"No sudoku grid detected. Could not find a 4-sided contour with area >= {min_area_threshold:.1f} pixels."
        )
        
    # Reshape from (4, 1, 2) to (4, 2)
    pts = grid_contour.reshape(4, 2)
    
    # Order corners using order_corners from grid_extractor
    from grid_extractor import order_corners
    ordered_pts = order_corners(pts)
    
    # Scale corners back to original image space
    if max_dim > 1000:
        ordered_pts = ordered_pts / scale
        
    return ordered_pts
