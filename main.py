import sys
import os
from grid_extractor import extract_cells

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
