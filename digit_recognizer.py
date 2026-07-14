import os
import random
import glob
import cv2
import numpy as np

def clear_cell_borders(gray_image, border_width=20):
    """
    Clears the outer border of a grayscale image by setting it to white background (255).
    This is extremely effective for removing grid line artifacts.
    Using border_width=20 is the optimal sweet spot to clear grid borders without clipping shifted digits.
    """
    img = gray_image.copy()
    h, w = img.shape[:2]
    img[0:border_width, :] = 255
    img[h-border_width:h, :] = 255
    img[:, 0:border_width] = 255
    img[:, w-border_width:w] = 255
    return img

def keep_largest_component(binary_image):
    """
    Finds the largest connected component in the binary image and discards all others.
    Assumes background is 255 (white) and foreground (digit) is 0 (black).
    This completely cleans up any isolated noise, split serifs, or grid remnants.
    """
    # Invert binary image so foreground is 1 and background is 0
    fg = (binary_image == 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    
    if num_labels <= 1:
        # Only background found
        return binary_image.copy()
        
    # Stats format: [left, top, width, height, area]
    # Label 0 is background, so we slice from index 1 to ignore it
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1 # +1 because of slicing
    
    # Create white canvas and draw only the largest component
    cleaned = np.ones_like(binary_image) * 255
    cleaned[labels == largest_label] = 0
    
    return cleaned

def center_digit(binary_image, target_size=32):
    """
    Crops a binary digit to its bounding box, scales it to fit within a 24x24 box
    preserving aspect ratio, and centers it on a 32x32 white canvas.
    Assumes background is 255 (white) and digit is 0 (black).
    """
    # Find black pixels
    y_indices, x_indices = np.where(binary_image == 0)
    
    if len(y_indices) == 0:
        # No digit found, return a pure white canvas
        return np.ones((target_size, target_size), dtype=np.uint8) * 255
        
    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()
    
    # Crop to bounding box
    digit_crop = binary_image[ymin:ymax+1, xmin:xmax+1]
    
    # Determine target scale to fit inside 24x24 (leaves 4px margin on all sides)
    box_size = int(target_size * 0.75)
    h_crop, w_crop = digit_crop.shape
    
    scale = box_size / max(h_crop, w_crop)
    new_h = max(1, int(h_crop * scale))
    new_w = max(1, int(w_crop * scale))
    
    # Resize keeping aspect ratio
    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create white canvas and center the digit
    canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
    dy = (target_size - new_h) // 2
    dx = (target_size - new_w) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized_digit
    
    return canvas

def generate_synthetic_dataset(output_dir="synthetic_digits", samples_per_digit=300):
    """
    Generates a synthetic dataset of digits 1-9 for training the digit classifier.
    Applies random font choice (regular & bold), font size, rotation, offsets, contrast, and noise.
    Also varies stroke weight (thickness) randomly (0-1px) to match bold print styles.
    Then applies border clearing (20px), binarization, connected component cleaning, centering, and saves 32x32 images.
    """
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(output_dir, exist_ok=True)

    # Common macOS font paths, including Helvetica, SFNS, and regular & bold variants
    common_fonts = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
        "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf",
    ]

    loaded_fonts = []
    for path in common_fonts:
        if os.path.exists(path):
            loaded_fonts.append(path)

    # Fallback to search any .ttf files
    if not loaded_fonts:
        import glob
        ttf_files = glob.glob("/System/Library/Fonts/**/*.ttf", recursive=True)
        if ttf_files:
            loaded_fonts.extend(ttf_files[:8])

    # Fallback to default
    if not loaded_fonts:
        print("Warning: No TrueType fonts found on the system. Falling back to default PIL font.")
        loaded_fonts = ["default"]

    print(f"Using fonts for generation: {loaded_fonts}")

    for digit in range(1, 10):
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)

        for n in range(samples_per_digit):
            bg_color = random.randint(220, 255)
            text_color = random.randint(0, 50)

            # Create a 100x100 canvas (grayscale mode "L")
            img = Image.new("L", (100, 100), bg_color)
            draw = ImageDraw.Draw(img)

            # Random font and size (60 to 85pt matches standard digit scales)
            font_choice = random.choice(loaded_fonts)
            font_size = random.randint(60, 85)

            # Load font safely
            if font_choice == "default":
                try:
                    font = ImageFont.load_default(size=font_size)
                except TypeError:
                    font = ImageFont.load_default()
            else:
                try:
                    font = ImageFont.truetype(font_choice, font_size)
                except Exception:
                    try:
                        font = ImageFont.load_default(size=font_size)
                    except TypeError:
                        font = ImageFont.load_default()

            # Random translation/offset [-10, 10] (simulates boundary cropping)
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            cx = 50 + dx
            cy = 50 + dy

            # Random stroke thickness (0 to 1 pixels to prevent over-thickening blobs)
            stroke_w = random.randint(0, 1)

            # Draw the digit. If it is 1, randomly use alternate characters '|' or 'I'
            char_to_draw = str(digit)
            if digit == 1:
                char_to_draw = random.choice(['1', 'I', '|'])

            # Render digit
            try:
                draw.text((cx, cy), char_to_draw, fill=text_color, font=font, anchor="mm",
                          stroke_width=stroke_w, stroke_fill=text_color)
            except Exception:
                try:
                    bbox = font.getbbox(char_to_draw)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                except AttributeError:
                    w, h = draw.textsize(char_to_draw, font=font)
                draw.text((cx - w / 2, cy - h / 2), char_to_draw, fill=text_color, font=font,
                          stroke_width=stroke_w, stroke_fill=text_color)

            # Random rotation
            angle = random.uniform(-10.0, 10.0)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)

            # Random shear (affine transform)
            shear_x = random.uniform(-0.15, 0.15)
            shear_y = random.uniform(-0.05, 0.05)
            try:
                img = img.transform(img.size, Image.AFFINE, (1, shear_x, 0, shear_y, 1, 0),
                                    resample=Image.BICUBIC, fillcolor=bg_color)
            except Exception:
                pass

            # Gaussian noise
            img_arr = np.array(img, dtype=np.float32)
            noise_std = random.uniform(1.5, 3.5)
            noise = np.random.normal(0, noise_std, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            
            # Apply border clearing (using border_width=20)
            img_arr = clear_cell_borders(img_arr, border_width=20)
            
            # Binarize
            _, img_arr = cv2.threshold(img_arr, 127, 255, cv2.THRESH_BINARY)
            
            # Keep largest connected component (ensures matching structure with inference)
            img_arr = keep_largest_component(img_arr)
            
            # Crop to center 80% (pixels 10 to 90)
            crop_margin = 10
            img_cropped = img_arr[crop_margin:100-crop_margin, crop_margin:100-crop_margin]
            
            # Center the digit inside a 32x32 canvas
            img_centered = center_digit(img_cropped, target_size=32)

            # Save the preprocessed synthetic training image directly
            save_path = os.path.join(digit_dir, f"img_{n}.png")
            cv2.imwrite(save_path, img_centered)

def train_classifier():
    """
    Loads synthetic dataset, processes images to 32x32, trains a CNN in TF/Keras,
    and saves the model to model/digit_classifier.h5.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    output_dir = "synthetic_digits"
    if not os.path.exists(output_dir):
        print(f"Error: Training dataset directory '{output_dir}' does not exist.")
        print("Please run dataset generation first: python digit_recognizer.py --generate")
        return

    print("Loading synthetic dataset...")
    images = []
    labels = []

    for digit in range(1, 10):
        digit_dir = os.path.join(output_dir, str(digit))
        search_path = os.path.join(digit_dir, "*.png")
        files = glob.glob(search_path)
        print(f"  Digit {digit}: found {len(files)} samples")
        for img_path in files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize to 32x32 (no-op since they are already 32x32, but keeps it robust)
                img_resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                images.append(img_resized)
                # Label is digit - 1
                labels.append(digit - 1)

    if not images:
        print("Error: No images loaded. Dataset might be empty.")
        return

    images = np.array(images, dtype=np.float32) / 255.0
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels, dtype=np.int32)

    num_samples = len(images)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    split_idx = int(0.8 * num_samples)
    x_train, x_val = images[:split_idx], images[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]

    print(f"Dataset split: {len(x_train)} train, {len(x_val)} validation.")

    # CNN architecture with Dropout
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(9, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nStarting CNN model training...")
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_val, y_val),
        verbose=1
    )

    val_acc = history.history['val_accuracy'][-1]
    print(f"\nTraining completed. Final validation accuracy: {val_acc:.4f}")

    os.makedirs("model", exist_ok=True)
    model_path = "model/digit_classifier.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")
