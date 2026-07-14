import os
import cv2
import numpy as np

def order_corners(pts):
    """
    Orders a set of 4 coordinates in the order:
    [top-left, top-right, bottom-right, bottom-left].
    
    Coordinates are formatted as [x, y].
    - top-left has the smallest sum (x + y)
    - bottom-right has the largest sum (x + y)
    - top-right has the smallest difference (y - x)
    - bottom-left has the largest difference (y - x)
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of coordinates
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    
    # Difference of coordinates (y - x)
    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    
    return rect

def extract_cells(image_path):
    """
    Extracts the 9x9 sudoku grid from an image, warps it to a 900x900px square,
    and returns a list of 81 cells (100x100px) in row-major order.
    
    Saves intermediate debug images to output/ directory.
    """
    # 1. Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image path does not exist: {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at path: {image_path}. Ensure it is a valid image format.")
        
    # Resize if larger than 1000px on the longest side (keep aspect ratio)
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1000:
        scale = 1000.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h} to optimize processing.")
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # 2. Preprocessing
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to handle uneven lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological dilation to close small gaps in the grid lines
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Save the thresholded/dilated image
    cv2.imwrite(os.path.join("output", "01_threshold.jpg"), thresh)
    
    # 3. Contour detection
    try:
        # Find all contours including internal ones
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as e:
        raise RuntimeError(f"Failed to find contours in preprocessed image: {str(e)}")
        
    grid_contour = None
    min_area_threshold = 0.025 * (img.shape[0] * img.shape[1])
    
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
    
    if candidates:
        # We start with the largest 4-sided contour (which could be the paper boundary)
        best_contour = candidates[0][1]
        best_area = candidates[0][0]
        
        # We search from smallest to largest to find the innermost nested contour
        # that is at least 60% of the largest one (likely the actual black grid frame).
        for area, approx in reversed(candidates):
            if area >= 0.6 * best_area:
                grid_contour = approx
                break
                
    # Draw detected contour (or original if none)
    img_contour = img.copy()
    if grid_contour is not None:
        cv2.drawContours(img_contour, [grid_contour], -1, (0, 255, 0), 3)
    cv2.imwrite(os.path.join("output", "02_contour.jpg"), img_contour)
    
    if grid_contour is None:
        raise ValueError(
            f"No sudoku grid detected. Could not find a 4-sided contour with area >= {min_area_threshold:.1f} pixels."
        )
        
    # 4. Corner ordering
    # Reshape from (4, 1, 2) to (4, 2)
    pts = grid_contour.reshape(4, 2)
    ordered_pts = order_corners(pts)
    
    # Print ordered corner coordinates to console
    print("Detected ordered corners (TL, TR, BR, BL):")
    for name, pt in zip(["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"], ordered_pts):
        print(f"  {name:12}: ({int(pt[0])}, {int(pt[1])})")
        
    # 5. Perspective warp
    try:
        # Destination coordinates for a 900x900 perfect square
        dst = np.array([
            [0, 0],
            [900, 0],
            [900, 900],
            [0, 900]
        ], dtype="float32")
        
        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        
        # Warp perspective
        warped = cv2.warpPerspective(img, M, (900, 900))
        cv2.imwrite(os.path.join("output", "03_warped.jpg"), warped)
    except Exception as e:
        raise RuntimeError(f"Failed to perform perspective warp on the detected grid: {str(e)}")
        
    # 6. Cell splitting
    os.makedirs(os.path.join("output", "cells"), exist_ok=True)
    cells = []
    
    for r in range(9):
        for c in range(9):
            # Each cell is 100x100 pixels
            cell = warped[r * 100 : (r + 1) * 100, c * 100 : (c + 1) * 100]
            
            # Save cell image
            cell_filename = os.path.join("output", "cells", f"cell_r{r}_c{c}.jpg")
            cv2.imwrite(cell_filename, cell)
            
            # Add to return list
            cells.append(cell)
            
    return cells
