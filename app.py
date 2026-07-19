import os
import sys

# Set environment variables for Keras and Matplotlib to use local workspace directories
os.environ["KERAS_HOME"] = os.path.abspath("./.keras")
os.environ["MPLCONFIGDIR"] = os.path.abspath("./.matplotlib")

import streamlit as st
import cv2
import numpy as np
import copy
import time

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

def render_status(placeholder, step_num, step_name, state):
    """
    Renders a engineering-style checklist item for pipeline progress.
    """
    if state == "pending":
        icon = "&nbsp;&nbsp;"
        border_color = "#E0E0D8"
        text_color = "#888888"
        bg_color = "#FAFAF8"
    elif state == "running":
        icon = "&bull;&nbsp;"
        border_color = "#185FA5"
        text_color = "#111111"
        bg_color = "#F4F7FB"
    elif state == "done":
        icon = "&check;&nbsp;"
        border_color = "#1D9E75"
        text_color = "#1D9E75"
        bg_color = "#F0F9F6"
    else: # error
        icon = "&cross;&nbsp;"
        border_color = "#D9534F"
        text_color = "#D9534F"
        bg_color = "#FDF4F4"
        
    placeholder.markdown(f"""
    <div style="border: 1px solid {border_color}; background-color: {bg_color}; padding: 0.5rem 0.8rem; margin-bottom: 0.5rem; border-radius: 2px; font-family: 'IBM Plex Mono', monospace; color: {text_color}; font-size: 0.9rem;">
        <span style="font-weight: 700;">{step_num}</span> &middot; {step_name} <span style="float: right; font-weight: 700;">{icon}</span>
    </div>
    """, unsafe_allow_html=True)

# 1. Header Bar
st.markdown("""
<div class="header-bar">
    <span class="header-title">SUDOKU_SOLVER_PIPELINE</span>
    <span class="header-tag">v1.0.0 &middot; ENGINEERING_DEMO</span>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

# File uploader
uploaded_file = st.file_uploader("Upload a Sudoku Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.pipeline_results = None
        
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    original_img = cv2.imread(temp_path)
    if original_img is not None:
        # Columns layout
        col_left, col_right = st.columns([1.3, 1])
        
        with col_left:
            st.markdown("<h3 style='margin-top:0;'>Solver Playback</h3>", unsafe_allow_html=True)
            st.info("Grid will be rendered here.")
            
            st.markdown("<h3>Solved on Original Photo</h3>", unsafe_allow_html=True)
            st.caption("Final solution warped back to original photo perspective")
            
        with col_right:
            st.markdown("<h3 style='margin-top:0;'>Pipeline Status</h3>", unsafe_allow_html=True)
            status_p1 = st.empty()
            status_p2 = st.empty()
            status_p3 = st.empty()
            status_p4 = st.empty()
            
            if st.session_state.pipeline_results is None:
                render_status(status_p1, "01", "grid detect", "pending")
                render_status(status_p2, "02", "digit recognition", "pending")
                render_status(status_p3, "03", "solve", "pending")
                render_status(status_p4, "04", "overlay", "pending")
                
                try:
                    # 1. Grid detection
                    render_status(status_p1, "01", "grid detect", "running")
                    time.sleep(0.2)
                    cells = extract_cells(temp_path)
                    render_status(status_p1, "01", "grid detect", "done")
                    
                    # 2. Digit recognition
                    render_status(status_p2, "02", "digit recognition", "running")
                    time.sleep(0.2)
                    grid, confidence_grid = build_grid()
                    render_status(status_p2, "02", "digit recognition", "done")
                    
                    # 3. Solve
                    render_status(status_p3, "03", "solve", "running")
                    time.sleep(0.2)
                    
                    if not is_valid_puzzle(grid):
                        render_status(status_p3, "03", "solve", "error")
                        st.error("Recognized grid is contradictory.")
                        st.session_state.pipeline_results = {"success": False, "error": "Contradictory grid"}
                    else:
                        grid_copy = copy.deepcopy(grid)
                        solved, backtracks = solve(grid_copy)
                        if not solved:
                            render_status(status_p3, "03", "solve", "error")
                            st.error("No solution exists.")
                            st.session_state.pipeline_results = {"success": False, "error": "No solution"}
                        else:
                            render_status(status_p3, "03", "solve", "done")
                            
                            # 4. Overlay
                            render_status(status_p4, "04", "overlay", "running")
                            time.sleep(0.2)
                            warped_img = cv2.imread("output/03_warped.jpg")
                            original_corners = detect_corners(original_img)
                            warped_overlay = draw_solution_on_warped(warped_img, grid, grid_copy)
                            unwarp_overlay(original_img, warped_overlay, original_corners)
                            render_status(status_p4, "04", "overlay", "done")
                            
                            st.session_state.pipeline_results = {
                                "success": True,
                                "grid": grid,
                                "solved_grid": grid_copy,
                                "overlay_saved": True
                            }
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.pipeline_results = {"success": False, "error": str(e)}
            else:
                res = st.session_state.pipeline_results
                if res.get("success"):
                    render_status(status_p1, "01", "grid detect", "done")
                    render_status(status_p2, "02", "digit recognition", "done")
                    render_status(status_p3, "03", "solve", "done")
                    render_status(status_p4, "04", "overlay", "done")
                else:
                    render_status(status_p1, "01", "grid detect", "done")
                    render_status(status_p2, "02", "digit recognition", "done")
                    render_status(status_p3, "03", "solve", "error")
                    
            if st.session_state.pipeline_results and st.session_state.pipeline_results.get("success"):
                st.markdown("<h3>Intermediate Outputs</h3>", unsafe_allow_html=True)
                st.image("output/02_contour.jpg", caption="Grid detected", use_container_width=True)
                
        if st.session_state.pipeline_results and st.session_state.pipeline_results.get("success"):
            with col_left:
                if os.path.exists("output/04_solved_overlay.jpg"):
                    st.image("output/04_solved_overlay.jpg", caption="Solved!", use_container_width=True)
