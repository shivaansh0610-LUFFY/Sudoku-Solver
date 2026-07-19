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

def draw_solution_on_warped(warped_img, original_grid, solved_grid):
    """
    Takes the 900x900 warped image (from Day 1), the originally recognized grid (with 0s for blanks),
    and the fully solved grid.
    
    Draws the solved digits on a copy of the warped image, in green, ONLY for cells that were
    originally empty (original_grid[r][c] == 0).
    """
    annotated = warped_img.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    thickness = 3
    color = (0, 180, 0) # Green color in BGR (or RGB since both G values match)
    
    for r in range(9):
        for c in range(9):
            # ONLY draw on cells that were originally empty
            if original_grid[r][c] == 0:
                digit = solved_grid[r][c]
                if digit == 0:
                    continue # Skip if it wasn't solved or still empty
                
                text = str(digit)
                
                # Measure text size to center it horizontally in the 100x100 cell
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Cell bounding coordinates: [c*100, (c+1)*100] horizontally
                cell_left_x = c * 100
                x = cell_left_x + (100 - text_width) // 2
                
                # Cell bounding coordinates: [r*100, (r+1)*100] vertically
                # Place baseline at (100 + text_height) // 2 for perfect centering
                cell_top_y = r * 100
                y = cell_top_y + (100 + text_height) // 2 - 2
                
                # Draw the text onto the warped grid
                cv2.putText(annotated, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
                
    return annotated

def unwarp_overlay(original_img, warped_overlay_img, original_corners, output_dir="output"):
    """
    Takes the original (unwarped, possibly skewed) input image and the annotated 900x900 warped+solved image.
    Computes the INVERSE perspective transform, warps the overlay back, and composites it onto the original.
    Saves the result to output_dir/04_solved_overlay.jpg.
    """
    # 1. Validate corners
    if original_corners is None or len(original_corners) != 4:
        raise ValueError("Invalid original corners: must contain exactly 4 points.")
        
    is_convex = cv2.isContourConvex(original_corners.astype(np.int32))
    if not is_convex:
        print("WARNING: The detected grid corners do not form a convex quadrilateral. The unwarped overlay might be distorted.")
        
    h_orig, w_orig = original_img.shape[:2]
    
    # Destination points in warp (source points in inverse warp)
    dst = np.array([
        [0, 0],
        [900, 0],
        [900, 900],
        [0, 900]
    ], dtype="float32")
    
    # 2. Compute inverse perspective transform
    M_inv = cv2.getPerspectiveTransform(dst, original_corners.astype(np.float32))
    
    # 3. Warp the annotated overlay back to the original image dimensions
    warped_back = cv2.warpPerspective(warped_overlay_img, M_inv, (w_orig, h_orig))
    
    # 4. Create and warp mask to prevent edge seams/gaps
    warped_mask = np.ones((900, 900), dtype=np.uint8) * 255
    mask = cv2.warpPerspective(warped_mask, M_inv, (w_orig, h_orig), flags=cv2.INTER_LINEAR)
    
    # 5. Composite using alpha blending
    mask_f = mask.astype(np.float32) / 255.0
    if len(original_img.shape) == 3:
        mask_3d = np.repeat(mask_f[:, :, np.newaxis], 3, axis=2)
    else:
        mask_3d = mask_f
        
    composited = (original_img.astype(np.float32) * (1.0 - mask_3d) + warped_back.astype(np.float32) * mask_3d)
    composited = np.clip(composited, 0, 255).astype(np.uint8)
    
    # Ensure output directory exists and save
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "04_solved_overlay.jpg"), composited)
    
    return composited
