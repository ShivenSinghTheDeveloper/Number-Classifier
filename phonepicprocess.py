import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

#IMPORTANT AFTER DEPLOYING PROJECT
#1.- The project only works when its a white paper, black sharpee and good hand writing
#2.- By depending on the sklearn datasets we are limited to train on our own way for our code to work 100%
#3.- IF we want make this more customize we need to add 15 to 20 pictures of each number on the folder that we see on the left
#4.- However, the code for data.py needs to be change to make this work

# Shared threshold settings (change here; both functions pick them up)
THRESH_BLOCK = 31   # must be odd; larger handles more uneven lighting
THRESH_C     = 10   # subtracted from mean; raise if background noise appears
DILATE_ITER  = 1    # stroke-thickening passes
MIN_AREA     = 50   # contours smaller than this are noise


def _pipeline(gray):
    """Shared gray -> binary threshold used by BOTH preprocess_pic and debug_view."""
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        THRESH_BLOCK, THRESH_C
    )
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=DILATE_ITER)
    return thresh


def _filter_contours(contours):
    filtered = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        if w > h * 3:
            continue
        if h > w * 8 and w < 8:
            continue
        filtered.append(cnt)
    return filtered


def _crop_and_square(thresh, contour):
    x, y, w, h = cv2.boundingRect(contour)
    digit = thresh[y:y + h, x:x + w]
    size  = max(w, h)
    pad_x = (size - w) // 2
    pad_y = (size - h) // 2
    sq = cv2.copyMakeBorder(
        digit,
        pad_y, size - h - pad_y,
        pad_x, size - w - pad_x,
        borderType=cv2.BORDER_CONSTANT, value=0
    )
    pad = max(4, size // 8)
    sq = cv2.copyMakeBorder(sq, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return sq


def _to_feature_vector(digit_sq, out_size=(8, 8)):
    """
    Resize and normalize to [0, 16] matching sklearn load_digits.

    WHY per-image normalization instead of dividing by 255:
    cv2.INTER_AREA averages pixels when shrinking. A thin white stroke (255)
    on a large black canvas produces averages as low as 1-2 after resize.
    Stretching so the brightest pixel = 16 gives the model the same dynamic
    range it was trained on.
    """
    resized   = cv2.resize(digit_sq, out_size, interpolation=cv2.INTER_AREA)
    arr       = resized.astype(np.float32)
    pixel_max = arr.max()

    if pixel_max < 1.0:
        raise ValueError(
            "Preprocessed image is nearly blank (max pixel < 1). "
            "Run debug_view() to inspect each pipeline stage."
        )

    arr = (arr / pixel_max) * 16.0   # brightest pixel -> 16
    return arr


def preprocess_pic(path, out_size=(8, 8)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image could not be read: {path}")

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh   = _pipeline(gray)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found. Try better lighting or a thicker digit.")

    filtered = _filter_contours(contours)
    if not filtered:
        raise ValueError("No digit-like contour found after filtering.")

    largest  = max(filtered, key=cv2.contourArea)
    digit_sq = _crop_and_square(thresh, largest)
    arr      = _to_feature_vector(digit_sq, out_size)

    return arr.reshape(-1)


def debug_view(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = _pipeline(gray)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    filtered = _filter_contours(contours)
    if not filtered:
        raise ValueError("No valid contour found after filtering.")

    largest       = max(filtered, key=cv2.contourArea)
    digit_sq      = _crop_and_square(thresh, largest)
    arr           = _to_feature_vector(digit_sq, (8, 8))
    digit_resized = cv2.resize(digit_sq, (8, 8), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
    axes[1].imshow(gray, cmap="gray");                    axes[1].set_title("Gray")
    axes[2].imshow(thresh, cmap="gray");                  axes[2].set_title(f"Threshold (block={THRESH_BLOCK}, C={THRESH_C})")
    axes[3].imshow(digit_sq, cmap="gray");                axes[3].set_title("Cropped + Padded")
    axes[4].imshow(digit_resized, cmap="gray");           axes[4].set_title("Final 8x8 (what model sees)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\n8x8 pixel stats -> min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f}")
    print("max should be 16.0")
    print("\nPixel grid (0-16 scale):")
    print(np.round(arr.reshape(8, 8)).astype(int))


def predict_digit(model_path, image_path):
    model = joblib.load(model_path)
    x     = preprocess_pic(image_path, out_size=(8, 8))
    pred  = model.predict([x])[0]
    return pred


def test_model_on_sklearn_dataset(model_path):
    model  = joblib.load(model_path)
    digits = load_digits()
    x, y   = digits.data[0], digits.target[0]
    pred   = model.predict([x])[0]
    print(f"Sklearn sample -> real={y}  predicted={pred}  {'OK' if pred == y else 'WRONG'}")


if __name__ == "__main__":
    MODEL_PATH = "digit_model.joblib"
    IMAGE_PATH = "one.jpg"

    test_model_on_sklearn_dataset(MODEL_PATH)
    debug_view(IMAGE_PATH)
    print("Prediction:", predict_digit(model_path=MODEL_PATH, image_path=IMAGE_PATH))