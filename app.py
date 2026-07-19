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
import streamlit.components.v1 as components

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
    /* Entrance animations keyframes */
    @keyframes slideInRight {
        from { transform: translateX(120px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideInLeft {
        from { transform: translateX(-120px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    /* Animation class assignments with braking easing curve */
    .header-bar {
        animation: slideInRight 0.7s cubic-bezier(0.15, 0.85, 0.35, 1) forwards;
    }
    [data-testid="column"]:nth-of-type(1) {
        animation: slideInLeft 0.7s cubic-bezier(0.15, 0.85, 0.35, 1) 0.30s forwards;
        opacity: 0;
    }
    [data-testid="column"]:nth-of-type(2) {
        animation: slideInLeft 0.7s cubic-bezier(0.15, 0.85, 0.35, 1) 0.45s forwards;
        opacity: 0;
    }

    /* Global style overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    /* Subtle background grid-line watermark */
    .stApp {
        background-color: #EDE9DC !important;
        background-image: 
            linear-gradient(135deg, rgba(29, 158, 117, 0.02) 0%, rgba(24, 95, 165, 0.02) 100%),
            linear-gradient(to right, rgba(26, 26, 23, 0.025) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(26, 26, 23, 0.025) 1px, transparent 1px) !important;
        background-size: auto, 40px 40px, 40px 40px !important;
        background-attachment: fixed !important;
    }
    
    /* Constrain and center main block container */
    [data-testid="stAppViewBlockContainer"], .block-container {
        max-width: 1000px !important;
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
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
        color: #1A1A17;
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

# Script to inject styles to prevent animation replay on reruns
st.html("""
<script>
    (function() {
        const targetWindow = window.parent || window;
        if (targetWindow.sessionStorage.getItem('sudoku_animated') === 'true') {
            const style = targetWindow.document.createElement('style');
            style.innerHTML = `
                .header-bar, [data-testid="column"]:nth-of-type(1), [data-testid="column"]:nth-of-type(2) {
                    animation: none !important;
                    opacity: 1 !important;
                    transform: none !important;
                }
            `;
            targetWindow.document.head.appendChild(style);
        } else {
            targetWindow.sessionStorage.setItem('sudoku_animated', 'true');
        }
    })();
</script>
""", unsafe_allow_javascript=True)

def render_status(placeholder, step_num, step_name, state):
    """
    Renders a engineering-style checklist item for pipeline progress.
    """
    if state == "pending":
        icon = "&nbsp;&nbsp;"
        border_color = "#E0E0D8"
        text_color = "#888888"
        bg_color = "#EDE9DC"
    elif state == "running":
        icon = "&bull;&nbsp;"
        border_color = "#185FA5" # Active running state (blue)
        text_color = "#1A1A17"
        bg_color = "#F4F7FB"
    elif state == "done":
        icon = "&check;&nbsp;"
        border_color = "#1D9E75" # Accent color (green)
        text_color = "#1D9E75"
        bg_color = "#F0F9F6"
    else: # error
        icon = "&cross;&nbsp;"
        border_color = "#D9534F" # Error state (red)
        text_color = "#D9534F"
        bg_color = "#FDF4F4"
        
    placeholder.markdown(f"""
    <div style="border: 1px solid {border_color}; background-color: {bg_color}; padding: 0.5rem 0.8rem; margin-bottom: 0.5rem; border-radius: 2px; font-family: 'IBM Plex Mono', monospace; color: {text_color}; font-size: 0.9rem;">
        <span style="font-weight: 700;">{step_num}</span> &middot; {step_name} <span style="float: right; font-weight: 700;">{icon}</span>
    </div>
    """, unsafe_allow_html=True)

def render_animated_grid(original_grid, solved_grid, confidence_grid=None, show_heatmap=False):
    """
    Renders a 9x9 Sudoku grid using HTML/CSS inside an iframe component.
    Solved cells are revealed in a diagonal wave animation.
    Optionally tints original clue backgrounds based on prediction confidence scores.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;700&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                display: flex;
                justify-content: center;
                align-items: center;
                font-family: 'IBM Plex Mono', monospace;
            }}
            .sudoku-grid {{
                display: grid;
                grid-template-columns: repeat(9, 1fr);
                width: 360px;
                height: 360px;
                border: 2px solid #1A1A17;
                background-color: #EDE9DC;
            }}
            .cell {{
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 1.35rem;
                border-right: 1px solid #E0E0D8;
                border-bottom: 1px solid #E0E0D8;
                box-sizing: border-box;
            }}
            /* 3x3 block borders */
            .cell-col-2, .cell-col-5 {{
                border-right: 2px solid #1A1A17;
            }}
            .cell-col-8 {{
                border-right: none;
            }}
            .cell-row-2, .cell-row-5 {{
                border-bottom: 2px solid #1A1A17;
            }}
            .cell-row-8 {{
                border-bottom: none;
            }}
            
            .original {{
                color: #1A1A17;
                font-weight: 700;
            }}
            .solved {{
                color: #1D9E75;
                font-weight: 700;
                opacity: 0;
                transform: scale(1.6);
                transition: opacity 300ms cubic-bezier(0.1, 0.8, 0.3, 1), transform 300ms cubic-bezier(0.1, 0.8, 0.3, 1);
            }}
            .solved.reveal {{
                opacity: 1;
                transform: scale(1);
            }}
        </style>
    </head>
    <body>
        <div class="sudoku-grid">
    """
    
    for r in range(9):
        for c in range(9):
            orig_val = original_grid[r][c]
            solved_val = solved_grid[r][c]
            
            cell_classes = f"cell cell-row-{r} cell-col-{c}"
            if orig_val != 0:
                bg_color = "transparent"
                if show_heatmap and confidence_grid is not None:
                    conf = confidence_grid[r][c]
                    if conf < 0.75:
                        bg_color = "rgba(217, 83, 79, 0.22)"  # Low confidence - warm warning red
                    elif conf < 0.90:
                        bg_color = "rgba(217, 119, 6, 0.12)"  # Medium confidence - warm orange
                    else:
                        bg_color = "rgba(29, 158, 117, 0.08)"  # High confidence - very faint green
                
                style_str = f'style="background-color: {bg_color};"' if bg_color != "transparent" else ""
                html_content += f'<div class="{cell_classes} original" {style_str}>{orig_val}</div>\n'
            else:
                diag = r + c
                html_content += f'<div class="{cell_classes} solved" data-diag="{diag}">{solved_val}</div>\n'
                
    html_content += f"""
        </div>
        <script>
            document.addEventListener("DOMContentLoaded", () => {{
                const cells = document.querySelectorAll(".solved");
                cells.forEach(cell => {{
                    const diag = parseInt(cell.getAttribute("data-diag"));
                    setTimeout(() => {{
                        cell.classList.add("reveal");
                    }}, diag * 90);
                }});
            }});
        </script>
    </body>
    </html>
    """
    st.iframe(html_content, height=380)

def generate_confidence_table(grid, conf_grid):
    """
    Generates a clean HTML table representing the recognized digits and their confidence scores.
    """
    html = ['<table style="width:100%; border-collapse: collapse; font-family: \'IBM Plex Mono\', monospace; border: 2px solid #1A1A17; font-size: 0.85rem; text-align: center; background-color: #EDE9DC;">']
    for r in range(9):
        row_style = ""
        if r in [2, 5]:
            row_style = 'style="border-bottom: 2px solid #1A1A17;"'
        elif r == 8:
            row_style = 'style="border-bottom: none;"'
        else:
            row_style = 'style="border-bottom: 1px solid #E0E0D8;"'
            
        html.append(f'<tr {row_style}>')
        for c in range(9):
            cell_style = "padding: 6px 2px;"
            if c in [2, 5]:
                cell_style += " border-right: 2px solid #1A1A17;"
            elif c == 8:
                cell_style += " border-right: none;"
            else:
                cell_style += " border-right: 1px solid #E0E0D8;"
                
            val = grid[r][c]
            conf = conf_grid[r][c]
            
            if val == 0:
                cell_content = "."
            else:
                conf_pct = int(conf * 100)
                if conf > 0.90:
                    color = "#1D9E75" # High confidence - Forest Green
                elif conf < 0.75:
                    color = "#D9534F" # Low confidence - Warning Red
                else:
                    color = "#1A1A17" # Medium confidence - Normal Text
                cell_content = f'<div style="font-weight:700; color:{color};">{val}</div><div style="font-size:0.65rem; color:#666;">{conf_pct}%</div>'
                
            html.append(f'<td style="{cell_style}">{cell_content}</td>')
        html.append('</tr>')
    html.append('</table>')
    return "\n".join(html)

def render_stats_bar(digits_count, backtracks, solve_time_ms):
    """
    Renders a 4-column metric row styled to match the mono/thin-border engineering aesthetic,
    plus a difficulty estimator label.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    def render_stat(col, value, label):
        col.markdown(f"""
        <div style="border: 1px solid #E0E0D8; padding: 0.5rem; border-radius: 2px; text-align: center; background-color: #EDE9DC;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #1A1A17;">{value}</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.65rem; color: #666; text-transform: uppercase; margin-top: 0.15rem; letter-spacing: 0.05em;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    render_stat(col1, "81", "cells")
    render_stat(col2, str(digits_count), "digits")
    render_stat(col3, str(backtracks), "backtracks")
    render_stat(col4, f"{solve_time_ms:.1f}ms", "solve time")

    # Difficulty estimate calculation based on clues and solver search depth
    if backtracks > 10000:
        difficulty = "Extreme"
    elif backtracks > 2000 or digits_count < 22:
        difficulty = "Hard"
    elif backtracks > 100 or digits_count < 28:
        difficulty = "Medium"
    else:
        difficulty = "Easy"

    st.markdown(f"""
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; color: #333333; margin-top: 0.8rem; padding: 0.4rem 0.6rem; border-left: 2px solid #1D9E75; background-color: #F0F9F6; border-radius: 0 2px 2px 0;">
        <strong>Difficulty Estimate:</strong> {difficulty} &mdash; {digits_count} clues, {backtracks:,} backtracks
    </div>
    """, unsafe_allow_html=True)

# 1. Header Bar
st.markdown("""
<div class="header-bar">
    <span class="header-title">SUDOKU_SOLVER_PIPELINE</span>
    <span class="header-tag">v1.0.0 &middot; ENGINEERING_DEMO</span>
</div>
""", unsafe_allow_html=True)

# 2. Landing Hero Section
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; color: #333333; line-height: 1.5; margin: 0 0 0.5rem 0;">
        Photo in, solved grid out. This pipeline extracts the board using OpenCV perspective warps, recognizes cells with a custom CNN classifier, and solves the puzzle using a backtracking algorithm.
    </p>
    <p style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: #666666; margin: 0; letter-spacing: 0.05em;">
        OPENCV &middot; TENSORFLOW/KERAS &middot; BACKTRACKING &middot; STREAMLIT
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_file_v3" not in st.session_state:
    st.session_state.current_file_v3 = None
if "pipeline_results_v3" not in st.session_state:
    st.session_state.pipeline_results_v3 = None
if "selected_example_v3" not in st.session_state:
    st.session_state.selected_example_v3 = None

# Example selection row
st.markdown("<p style='font-family: \"Space Grotesk\", sans-serif; font-size: 0.9rem; font-weight: 500; margin-bottom: 0.4rem;'>Try a sample image:</p>", unsafe_allow_html=True)
col_ex1, col_ex2, col_ex3 = st.columns(3)
if col_ex1.button("Standard (30 clues)", width="stretch"):
    st.session_state.selected_example_v3 = "test_images/Screenshot 2026-07-14 at 3.40.04 PM.png"
    st.session_state.current_file_v3 = "Screenshot 2026-07-14 at 3.40.04 PM.png"
    st.session_state.pipeline_results_v3 = None
if col_ex2.button("Hard (17 clues)", width="stretch"):
    st.session_state.selected_example_v3 = "test_images/sample_hard.jpg"
    st.session_state.current_file_v3 = "sample_hard.jpg"
    st.session_state.pipeline_results_v3 = None
if col_ex3.button("Empty Grid (0 clues)", width="stretch"):
    st.session_state.selected_example_v3 = "test_images/sample.jpg"
    st.session_state.current_file_v3 = "sample.jpg"
    st.session_state.pipeline_results_v3 = None

# File uploader
uploaded_file = st.file_uploader("Upload a Sudoku Image", type=["jpg", "jpeg", "png"])

image_path_to_use = None
if uploaded_file is not None:
    # Clear cache if a new file is uploaded
    if st.session_state.current_file_v3 != uploaded_file.name:
        st.session_state.current_file_v3 = uploaded_file.name
        st.session_state.selected_example_v3 = None
        st.session_state.pipeline_results_v3 = None
        
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    original_img = cv2.imread(temp_path)
    image_path_to_use = temp_path
elif st.session_state.selected_example_v3 is not None:
    original_img = cv2.imread(st.session_state.selected_example_v3)
    image_path_to_use = st.session_state.selected_example_v3
else:
    original_img = None

if image_path_to_use is not None and original_img is None:
    st.error(f"Error: Failed to read image at path '{image_path_to_use}'")

if original_img is not None:
    # Columns layout
    col_left, col_right = st.columns([1.3, 1])
    
    with col_left:
        st.markdown("<h3 style='margin-top:0;'>Solver Playback</h3>", unsafe_allow_html=True)
        
    with col_right:
        st.markdown("<h3 style='margin-top:0;'>Pipeline Status</h3>", unsafe_allow_html=True)
        status_p1 = st.empty()
        status_p2 = st.empty()
        status_p3 = st.empty()
        status_p4 = st.empty()
        
        # If not in cache, run the pipeline step-by-step
        if st.session_state.pipeline_results_v3 is None:
            render_status(status_p1, "01", "grid detect", "pending")
            render_status(status_p2, "02", "digit recognition", "pending")
            render_status(status_p3, "03", "solve", "pending")
            render_status(status_p4, "04", "overlay", "pending")
            
            try:
                # 1. Grid detection
                render_status(status_p1, "01", "grid detect", "running")
                time.sleep(0.2)
                cells = extract_cells(image_path_to_use)
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
                    st.error("Recognized grid is contradictory — check digit recognition output above for misread cells")
                    st.session_state.pipeline_results_v3 = {"success": False, "error": "Contradictory grid"}
                else:
                    grid_copy = copy.deepcopy(grid)
                    start_time = time.time()
                    solved, backtracks = solve(grid_copy)
                    end_time = time.time()
                    solve_time_ms = (end_time - start_time) * 1000.0
                    
                    if not solved:
                        render_status(status_p3, "03", "solve", "error")
                        st.error("No solution exists for this Sudoku grid.")
                        st.session_state.pipeline_results_v3 = {"success": False, "error": "No solution"}
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
                        
                        # Cache all success outputs
                        st.session_state.pipeline_results_v3 = {
                            "success": True,
                            "grid": grid,
                            "confidence_grid": confidence_grid,
                            "solved_grid": grid_copy,
                            "overlay_saved": True,
                            "backtracks": backtracks,
                            "solve_time_ms": solve_time_ms,
                            "digits_count": sum(1 for row in grid for val in row if val != 0)
                        }
            except Exception as e:
                st.error(f"Error executing pipeline: {e}")
                st.session_state.pipeline_results_v3 = {"success": False, "error": str(e)}
        
        # If cached, render status checklist based on stored result
        res = st.session_state.pipeline_results_v3
        if res.get("success"):
            render_status(status_p1, "01", "grid detect", "done")
            render_status(status_p2, "02", "digit recognition", "done")
            render_status(status_p3, "03", "solve", "done")
            render_status(status_p4, "04", "overlay", "done")
        else:
            render_status(status_p1, "01", "grid detect", "done")
            render_status(status_p2, "02", "digit recognition", "done")
            render_status(status_p3, "03", "solve", "error")
            st.error(f"Cached pipeline error: {res.get('error')}")
            
        # Show stats bar
        if st.session_state.pipeline_results_v3 and st.session_state.pipeline_results_v3.get("success"):
            res = st.session_state.pipeline_results_v3
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            render_stats_bar(res["digits_count"], res["backtracks"], res["solve_time_ms"])
            
        # Show additional pipeline components in right column
        if st.session_state.pipeline_results_v3 and st.session_state.pipeline_results_v3.get("success"):
            res = st.session_state.pipeline_results_v3
            grid = res["grid"]
            confidence_grid = res["confidence_grid"]
            
            st.markdown("<h3>Intermediate Outputs</h3>", unsafe_allow_html=True)
            st.image("output/02_contour.jpg", caption="Grid detected", width="stretch")
            
            # Digit recognition table / pretty print
            st.markdown("<h3>Recognized Digits</h3>", unsafe_allow_html=True)
            show_conf = st.toggle("Show confidence scores", value=False)
            
            if show_conf:
                html_table = generate_confidence_table(grid, confidence_grid)
                st.markdown(html_table, unsafe_allow_html=True)
                st.caption("Digits recognized with confidence scores")
            else:
                # Pretty printed string representation
                def format_grid_str(g):
                    lines = []
                    for r in range(9):
                        if r % 3 == 0 and r != 0:
                            lines.append("-" * 21)
                        row_str = ""
                        for c in range(9):
                            if c % 3 == 0 and c != 0:
                                row_str += "| "
                            val = g[r][c]
                            row_str += f"{val if val != 0 else '.'} "
                        lines.append(row_str.strip())
                    return "\n".join(lines)
                
                st.code(format_grid_str(grid))
                st.caption("Digits recognized")
                
        if st.session_state.pipeline_results_v3 and st.session_state.pipeline_results_v3.get("success"):
            with col_left:
                res = st.session_state.pipeline_results_v3
                show_heatmap = st.toggle("Show confidence heatmap", value=False)
                # Playback animation
                render_animated_grid(res["grid"], res["solved_grid"], res.get("confidence_grid"), show_heatmap)
                
                st.markdown("<h3>Solved on Original Photo</h3>", unsafe_allow_html=True)
                st.caption("Final solution warped back to original photo perspective")
                if os.path.exists("output/04_solved_overlay.jpg"):
                    st.image("output/04_solved_overlay.jpg", caption="Solved!", width="stretch")
        else:
            with col_left:
                st.info("Upload an image to start pipeline solver playback.")

# 3. How Does This Work Expander
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
with st.expander("How does this work?"):
    st.markdown("""
    <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; line-height: 1.6; color: #333333; padding: 0.5rem 0;">
        <ul style="margin: 0; padding-left: 1.2rem;">
            <li style="margin-bottom: 0.5rem;">
                <strong>Perspective Grid Extraction:</strong> The input image is preprocessed using Gaussian blur and adaptive thresholding. The system extracts the largest 4-sided contour as the board, unwarps it into a clean 900x900 pixel square using an OpenCV perspective transform, and segments it into 81 cell images.
            </li>
            <li style="margin-bottom: 0.5rem;">
                <strong>Synthetic Font CNN Training:</strong> To recognize printed characters, the digit classifier utilizes a custom convolutional network (CNN) trained on machine-printed fonts with added noise, rotations, and line artifacts. This yields robust predictions for structured sudoku puzzles without the hand-drawn bias of MNIST.
            </li>
            <li style="margin-bottom: 0.5rem;">
                <strong>Backtracking Logic Solver:</strong> The backend solver recursively searches the state space using depth-first search. It assigns a valid number to an empty cell, checks constraints, and recursively solves the rest of the board. If a dead end is encountered, it backtracks and attempts the next number.
            </li>
            <li style="margin-bottom: 0.5rem;">
                <strong>Softmax Confidence Propagation:</strong> The CNN outputs a softmax probability vector for each cell. The maximum value represents the confidence score of the digit classification. Lower-confidence classifications can be inspected dynamically via the table or heatmap overlays.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 4. Footer
st.markdown("""
<hr style="border: 0; border-top: 1px solid #E0E0D8; margin-top: 3rem; margin-bottom: 1.5rem;">
<div style="display: flex; justify-content: space-between; font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; color: #888888;">
    <span>Attribution: Built by Shivaansh & Antigravity</span>
    <span><a href="https://github.com/shivaansh0610-LUFFY/Sudoku-Solver" target="_blank" style="color: #888888; text-decoration: none; border-bottom: 1px solid #CCCCCC;">GitHub Repository</a></span>
</div>
""", unsafe_allow_html=True)


