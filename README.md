# 🧩 Sudoku Solver Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

An end-to-end computer vision and deep learning pipeline that takes a photograph or screenshot of a printed Sudoku puzzle, extracts and deskews the grid, recognizes the pre-filled digits with high confidence, and renders the solved board back onto the original perspective.

🚀 **Live Web App Demo**: Explore the interactive visual solver at [sudoku-solver-0610.streamlit.app](https://sudoku-solver-0610.streamlit.app)

---

## ⚡ Features & Capabilities

- **Interactive Solver Web UI**: Beautiful custom light parchment theme layout featuring staggered loading states, interactive stats, recognized digits lookup table, and solving metric logs.
- **Confidence Heatmap Overlay**: Softmax confidence scores derived from the CNN model can be toggled on-grid to color-tint cells dynamically (highlighting low-confidence digit reads).
- **Difficulty Estimator Alert**: Dynamically estimates puzzle difficulty (e.g. *Easy, Medium, Hard, Extreme*) using pre-filled clue numbers and recursive backtrack search counts.
- **Robust OCR Pipeline**: Cleans cell borders, removes grid noise using Connected Component Analysis (CCA), and centers the digits for invariant scale/translation predictions.
- **Synthetic Font Canvas Generator**: Custom dataset generator trained on computer-printed macOS system fonts and noise distributions (instead of MNIST hand-drawn samples) for perfect alignment on printed boards.

---

## 📸 Pipeline Visual Flow

| 1. Original Input | 2. Detected Contour | 3. Deskewed Warped Grid | 4. Solved Perspective Overlay |
| :---: | :---: | :---: | :---: |
| ![Original](test_images/sample_hard.jpg) | ![Contour](output/02_contour.jpg) | ![Warped](output/03_warped.jpg) | ![Overlay](output/04_solved_overlay.jpg) |

---

## ⚙️ How It Works

### 1. Grid Extraction (OpenCV)
- **Preprocess**: Grayscale conversion, Gaussian blurring, and adaptive thresholding handle non-uniform shadows. Morphological dilation closes structural gaps in grid lines.
- **Deskew & Warp**: Detects the largest 4-sided contour, orders its corner vectors (Top-Left, Top-Right, Bottom-Right, Bottom-Left), and applies a perspective warp to project it to a flat `900x900px` canvas.
- **Segment**: Slices the flat board into 81 uniform `100x100px` cells.

### 2. Digit Recognition (TensorFlow & Keras)
- **Border Clearing**: Shaves the outer 20-pixel frame of each cell to remove bleeding grid lines.
- **Ink Clean**: Isolates the largest connected component (foreground stroke) using Connected Component Analysis (CCA), discarding split serifs and dust.
- **Scale Centering**: Bounding-box crops the digit, scales it to fit within a `24x24px` viewport, and centers it on a `32x32px` blank canvas.
- **Blank Detection**: Classifies empty cells using standard deviation and pixel density thresholds (< 2.0% ink is treated as blank).
- **CNN Classifier**: Custom Convolutional Neural Network trained on standard macOS printed fonts with added rotation (`-10°` to `+10°`), translation, shearing, and noise.

### 3. Backtracking Logic Solver & Validator
- **Integrity Validation**: Sanitizes the input grid by verifying that already-filled cells do not violate row, column, or 3x3 box rules.
- **DF Backtracking Algorithm**: A depth-first recursive solver search that assigns candidate numbers 1-9 in row-major order, checks puzzle constraints, and backtracks upon dead ends.

---

## 🛠️ Installation & CLI Usage

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/shivaansh0610-LUFFY/Sudoku-Solver.git
cd Sudoku-Solver
pip install -r requirements.txt
```

### 2. Local Training (Optional)
Generate the synthetic digits dataset and train the CNN classifier weights locally:
```bash
# Generate the synthetic digits (saved under synthetic_digits/)
python digit_recognizer.py --generate

# Train the CNN model (saves model to model/digit_classifier.h5)
python digit_recognizer.py --train
```

### 3. Run Pipeline CLI
Process and solve any puzzle photo directly from the command line:
```bash
python main.py path/to/sudoku_image.jpg
```

---

## 🖥️ Local Web Deployment
Launch the interactive Streamlit dashboard locally:
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
Sudoku-Solver/
├── app.py                   # Streamlit web application dashboard (UI/UX)
├── main.py                  # Main CLI pipeline entry point
├── solver.py                # Backtracking solver algorithm & grid validation
├── grid_extractor.py        # OpenCV image warping & cell segmenter
├── digit_recognizer.py      # Synthetic dataset generator, CNN classifier & OCR
├── requirements.txt         # Project package dependencies
├── .streamlit/
│   └── config.toml          # Streamlit theme colors config (Warm parchment)
├── assets/
│   └── sudoku_hero.jpg      # Premium hero header image
├── test_images/             # Sample images (Standard, Hard, Empty)
├── model/
│   └── digit_classifier.h5  # Trained Keras CNN model weights
└── output/                  # Generated intermediate pipeline steps
    ├── 01_threshold.jpg     # Binary thresholded scan
    ├── 02_contour.jpg       # Detected grid boundaries
    ├── 03_warped.jpg        # Deskewed 900x900 grid
    └── 04_solved_overlay.jpg# Solved overlay warped back to original photo
```

---

## 🗺️ Project Roadmap

- [x] Grid extraction & perspective warp correction
- [x] Blank-cell thresholding & printed font CNN classification
- [x] Recursive Depth-First Search solver & cell validation
- [x] Inverse perspective overlay projection back onto raw photos
- [x] Staggered interactive Streamlit Web App dashboard
- [x] Confidence heatmap grid visualization
- [x] Difficulty analysis & backtrack metric logs

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
