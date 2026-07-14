# Sudoku Solver

A computer vision and deep learning pipeline that takes a photo or screenshot of a Sudoku puzzle, extracts the grid, recognizes the printed digits, and solves the puzzle. 

This project is split into phases:
- **Phase 1 (Day 1)**: Grid extraction and cell segmentation using OpenCV.
- **Phase 2 (Day 2)**: Digit recognition using a custom-trained Convolutional Neural Network (CNN) in TensorFlow/Keras.
- **Phase 3 (Upcoming)**: Backtracking Sudoku solver and solution rendering.

---

## Demo

| Input Photo | Warped Grid | Preprocessed Cells | Reconstructed Grid |
|---|---|---|---|
| Raw photo / Screenshot | `output/03_warped.jpg` | Aligned and binarized digits | Console text output |

Example output printed to console:
```text
8 . . | 4 . 6 | . . 7
. . . | . . . | 4 . .
. 1 . | . . . | 6 5 .
---------------------
5 . 9 | . 3 . | 7 8 .
. . . | . 7 . | . . .
. 4 8 | . 2 . | 1 . 3
---------------------
. 5 2 | . . . | . 9 .
. . 1 | . . . | . . .
3 . . | 9 . 2 | . . 5
```

---

## How It Works

### 1. Grid Extraction (OpenCV)
- **Preprocess** — Grayscale conversion, Gaussian blurring, and adaptive thresholding to handle uneven lighting. Morphological dilation closes small gaps in black grid lines.
- **Warp** — Locates the nested black grid contour, orders the corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left), and applies a perspective transform to map it to a flat `900x900px` square.
- **Segment** — Slices the warped image into 81 evenly-spaced `100x100px` individual cell images.

### 2. Digit Recognition (TensorFlow/Keras)
- **Binarization & Cleaning** — Clears the outer 20-pixel border of each cell to erase remaining grid lines. Keeps only the largest connected component (foreground stroke) to discard split serifs or small dust particles.
- **Centering** — Crops the digit to its bounding box, rescales it to fit within a `24x24px` box, and centers it on a `32x32px` white canvas. This makes the recognition translation- and scale-invariant.
- **Blank Cell Detection** — Analyzes the center 60% of each cleared cell using standard deviation and ink pixel density (< 2.0% area is classified as blank).
- **CNN Classifier** — A custom CNN model (`Conv2D -> MaxPool -> Dropout -> Conv2D -> MaxPool -> Dropout -> Dense -> Dropout -> Dense`) trained on a diverse synthetic dataset of printed fonts.

### 3. Synthetic Dataset Generator
Since handwritten MNIST digits look significantly different from printed puzzle fonts, the pipeline includes a synthetic training generator that:
- Renders digits 1-9 onto blank canvases using common macOS system fonts (Arial, Helvetica, Verdana, Georgia, Times New Roman, Courier New, Trebuchet MS) and modern sans-serifs (Helvetica Neue, SFNS).
- Applies random scaling (`60-85`pt), rotation (`-10` to `10` degrees), shearing, random translation, and noise.
- Translates digits up to `[-10, 10]`px to naturally simulate boundary cropping caused by cell borders.
- Randomly renders the digit `1` as `'1'`, `'I'`, or `'|'` to ensure robustness for fonts representing it as a simple vertical line.

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Model
To generate the training dataset (saved to `synthetic_digits/`) and train the CNN classifier (saved to `model/digit_classifier.h5`):
```bash
# Generate the synthetic digits
python digit_recognizer.py --generate

# Train the CNN model
python digit_recognizer.py --train
```

### 3. Run the Pipeline
Run the end-to-end extractor and digit recognition pipeline on any Sudoku puzzle photo or screenshot:
```bash
python main.py path/to/sudoku_image.jpg
```

---

## Project Structure

```text
Sudoku Solver/
├── main.py                  # Main CLI entrypoint
├── grid_extractor.py        # OpenCV grid warping and cell segmenter
├── digit_recognizer.py      # Synthetic dataset generator, CNN model, and OCR pipeline
├── requirements.txt         # Project dependencies
├── .gitignore               # Ignored caches, datasets, and output images
├── test_images/             # Test photos and screenshots
├── model/
│   └── digit_classifier.h5  # Trained Keras CNN model weights
└── output/                  # Generated intermediate images
    ├── 01_threshold.jpg     # Binary thresholded image
    ├── 02_contour.jpg       # Detected grid outline
    ├── 03_warped.jpg        # Deskewed 900x900 grid
    └── cells/               # Folder containing 81 cell images
```

---

## Requirements

- Python 3.9+
- OpenCV (`opencv-python`)
- NumPy
- TensorFlow
- Pillow (PIL)

---

## Roadmap

- [x] Grid detection and perspective correction
- [x] Digit recognition (blank-cell detection + CNN classifier on printed digits)
- [ ] Backtracking Sudoku solver (Phase 3)
- [ ] Solution overlay back onto the original photo
- [ ] Simple web demo (Streamlit)

---

## License

MIT