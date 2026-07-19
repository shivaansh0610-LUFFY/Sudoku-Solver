import sys
import os
import copy
import time
from grid_extractor import extract_cells
from digit_recognizer import build_grid
from solver import is_valid_puzzle, solve, pretty_print_grid

def main():
    # Verify input arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_sudoku_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: The specified file '{image_path}' does not exist.")
        sys.exit(1)
        
    is_pdf = image_path.lower().endswith(".pdf")
    temp_pdf_image_path = None
    
    if is_pdf:
        print(f"Detected PDF file: {image_path}. Converting first page to image...")
        try:
            import fitz  # PyMuPDF
            import numpy as np
            import cv2
            
            doc = fitz.open(image_path)
            if doc.page_count < 1:
                print("Error: The specified PDF contains no pages.")
                sys.exit(1)
                
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=150)
            
            # Convert pixmap to numpy BGR image
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                bgr_img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                bgr_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                
            temp_pdf_image_path = "temp_pdf_page.jpg"
            cv2.imwrite(temp_pdf_image_path, bgr_img)
            image_path = temp_pdf_image_path
        except Exception as e:
            print(f"Error converting PDF to image: {e}")
            sys.exit(1)
        
    print(f"Processing Sudoku image: {image_path}...")
    
    try:
        # Run extraction pipeline
        cells = extract_cells(image_path)
        
        # Output confirmation and details
        print("\nExtraction Success!")
        print(f"Successfully extracted {len(cells)} cells (9x9 grid).")
        print("Debug output images saved to:")
        print("  - output/01_threshold.jpg")
        print("  - output/02_contour.jpg")
        print("  - output/03_warped.jpg")
        print("  - output/cells/cell_r{row}_c{col}.jpg")
        
        # Run digit recognition
        print("\nRunning digit recognition on extracted cells...")
        grid, confidence_grid = build_grid()
        
        # Print grid
        print("\nDetected Sudoku Grid:")
        pretty_print_grid(grid)
        print()
        
        # Check validity
        if not is_valid_puzzle(grid):
            print("Recognized grid is contradictory — check digit recognition output above for misread cells")
            sys.exit(1)
            
        # Solve a deep copy of the grid
        grid_copy = copy.deepcopy(grid)
        start_time = time.time()
        solved, backtracks = solve(grid_copy)
        end_time = time.time()
        
        if solved:
            print("Solved Sudoku Grid:")
            pretty_print_grid(grid_copy)
            print(f"\nSolving took {(end_time - start_time) * 1000:.2f} ms with {backtracks} backtracks")
            
            # Day 4: Overlay solved digits back onto the original photo
            print("\nGenerating solution overlay...")
            import cv2
            from overlay import detect_corners, draw_solution_on_warped, unwarp_overlay
            
            original_img = cv2.imread(image_path)
            if original_img is None:
                raise ValueError(f"Could not load original image at path: {image_path}")
                
            warped_img_path = os.path.join("output", "03_warped.jpg")
            warped_img = cv2.imread(warped_img_path)
            if warped_img is None:
                raise ValueError(f"Could not load warped image at path: {warped_img_path}")
                
            # Detect original corners
            original_corners = detect_corners(original_img)
            
            # Draw solved digits on warped image
            warped_overlay = draw_solution_on_warped(warped_img, grid, grid_copy)
            
            # Inverse perspective warp back to original image
            unwarp_overlay(original_img, warped_overlay, original_corners)
            
            print("Confirmation: Saved solved overlay to output/04_solved_overlay.jpg")
        else:
            print("No solution exists for this Sudoku grid.")
        
    except FileNotFoundError as fnfe:
        print(f"\nConfiguration Error: {fnfe}")
        sys.exit(1)
    except ValueError as ve:
        print(f"\nDetection Error: {ve}")
        sys.exit(1)
    except RuntimeError as re:
        print(f"\nProcessing Error: {re}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        if temp_pdf_image_path and os.path.exists(temp_pdf_image_path):
            try:
                os.remove(temp_pdf_image_path)
            except:
                pass

if __name__ == "__main__":
    main()

