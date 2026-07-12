# Sudoku Grid Extractor

A computer vision pipeline that takes a photo of a Sudoku puzzle and extracts a clean, deskewed grid — ready for digit recognition and solving.

This is Phase 1 of a full Sudoku Solver: photo in → 81 individual cell images out.

## Demo

| Input | Threshold | Detected Grid | Warped Output |
|---|---|---|---|
| Raw photo | `output/01_threshold.jpg` | `output/02_contour.jpg` | `output/03_warped.jpg` |

The pipeline locates the puzzle boundary in any reasonably clean photo, corrects for camera angle/perspective, and slices the result into a perfect 9×9 grid of cells.

## How it works

1. **Preprocess** — grayscale, Gaussian blur, and adaptive thresholding to handle uneven lighting from a phone camera. A morphological dilation step closes small gaps in the grid lines that adaptive thresholding tends to leave, which otherwise causes contour detection to miss the boundary.
2. **Detect the grid** — finds the largest 4-sided external contour in the frame, which is assumed to be the puzzle boundary.
3. **Order the corners** — sorts the 4 detected points into a consistent top-left → top-right → bottom-right → bottom-left order using coordinate sum/difference, so the perspective warp doesn't invert or mirror the image.
4. **Warp** — applies a perspective transform to map the (possibly skewed) grid onto a flat 900×900px square.
5. **Split** — slices the warped square into 81 evenly-spaced 100×100px cell images, saved individually for downstream digit recognition.

## Usage

```bash
pip install -r requirements.txt
python main.py path/to/your/sudoku_photo.jpg
```

Output is written to:
```
output/
├── 01_threshold.jpg      # preprocessed binary image
├── 02_contour.jpg        # detected grid boundary overlaid on original
├── 03_warped.jpg         # deskewed, top-down 900x900 grid
└── cells/
    ├── cell_r0_c0.jpg
    ├── cell_r0_c1.jpg
    └── ...               # 81 cells total, row-major order
```

## Requirements

- Python 3.9+
- `opencv-python`
- `numpy`

## Known limitations

- Assumes the Sudoku grid is the largest 4-sided object in the frame — a cluttered background (hands, other rectangular objects) may cause misdetection.
- Tuned for printed puzzles; handwritten/pen-filled grids are not yet supported.
- Detection can fail on photos with heavy glare or very low contrast between the grid lines and background.

## Roadmap

- [x] Grid detection and perspective correction
- [ ] Digit recognition (blank-cell detection + CNN classifier on printed digits)
- [ ] Backtracking Sudoku solver
- [ ] Solution overlay back onto the original photo
- [ ] Simple web demo (Streamlit)

## License

MIT