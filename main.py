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
        grid = build_grid()
        
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
        solved = solve(grid_copy)
        end_time = time.time()
        
        if solved:
            print("Solved Sudoku Grid:")
            pretty_print_grid(grid_copy)
            print(f"\nSolving took {(end_time - start_time) * 1000:.2f} ms")
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

if __name__ == "__main__":
    main()

