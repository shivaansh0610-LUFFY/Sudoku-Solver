import sys
import os
from grid_extractor import extract_cells
from digit_recognizer import build_grid

def print_sudoku_grid(grid):
    """Prints a 9x9 Sudoku grid in a readable format with 3x3 block separators."""
    print("\nDetected Sudoku Grid:")
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            val = grid[r][c]
            row_str += f"{val if val != 0 else '.'} "
        print(row_str.strip())
    print()

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
        print_sudoku_grid(grid)
        
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

