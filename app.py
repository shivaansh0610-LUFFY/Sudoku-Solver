import os
import sys

# Set environment variables for Keras and Matplotlib to use local workspace directories
os.environ["KERAS_HOME"] = os.path.abspath("./.keras")
os.environ["MPLCONFIGDIR"] = os.path.abspath("./.matplotlib")

import streamlit as st
import cv2
import numpy as np
import copy

from grid_extractor import extract_cells
from digit_recognizer import build_grid
from solver import is_valid_puzzle, solve
from overlay import detect_corners, draw_solution_on_warped, unwarp_overlay

# Inject Google Fonts and custom CSS for the engineering theme
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    /* Global style overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Monospace elements */
    code, pre, .mono-font, [data-testid="stCodeBlock"], [data-testid="stCodeBlock"] * {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Custom header styling */
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #E0E0D8;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 1.2rem;
        color: #111111;
    }
    .header-tag {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# 1. Header Bar
st.markdown("""
<div class="header-bar">
    <span class="header-title">SUDOKU_SOLVER_PIPELINE</span>
    <span class="header-tag">v1.0.0 &middot; ENGINEERING_DEMO</span>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a Sudoku Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    original_img = cv2.imread(temp_path)
    if original_img is not None:
        # Create asymmetric layout columns
        col_left, col_right = st.columns([1.3, 1])
        
        with col_left:
            st.markdown("<h3 style='margin-top:0;'>Solver Playback</h3>", unsafe_allow_html=True)
            st.info("Grid will be rendered here.")
            
            st.markdown("<h3>Solved on Original Photo</h3>", unsafe_allow_html=True)
            st.caption("Final solution warped back to original photo perspective")
            
        with col_right:
            st.markdown("<h3 style='margin-top:0;'>Pipeline Status</h3>", unsafe_allow_html=True)
            with st.spinner("Processing pipeline..."):
                cells = extract_cells(temp_path)
                grid, confidence_grid = build_grid()
                
                if not is_valid_puzzle(grid):
                    st.error("Contradictory recognized grid.")
                else:
                    grid_copy = copy.deepcopy(grid)
                    solved, backtracks = solve(grid_copy)
                    if solved:
                        warped_img = cv2.imread("output/03_warped.jpg")
                        original_corners = detect_corners(original_img)
                        warped_overlay = draw_solution_on_warped(warped_img, grid, grid_copy)
                        unwarp_overlay(original_img, warped_overlay, original_corners)
                        
            st.success("Pipeline execution complete.")
            
            with col_left:
                if os.path.exists("output/04_solved_overlay.jpg"):
                    st.image("output/04_solved_overlay.jpg", caption="Solved!", use_container_width=True)
