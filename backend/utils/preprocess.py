import cv2
import numpy as np

def enhance_xray(image_path: str, save_path: str) -> None:
    """
    Enhance chest X-ray contrast using CLAHE.
    CLAHE works on small regions of the image separately,
    so it brings out subtle lung patterns that might be too dark/light overall.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    cv2.imwrite(save_path, enhanced)

def enhance_retinal(image_path: str, save_path: str) -> None:
    """
    Enhance retinal fundus image.
    We work in LAB color space (separates brightness from color)
    so we only enhance brightness without distorting the red/green colors
    that indicate blood vessels and lesions.
    """
    img = cv2.imread(image_path)
    # LAB = Lightness, A channel, B channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Only enhance the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path, enhanced)

def check_image_quality(image_path: str) -> dict:
    """
    Basic sanity checks before we waste time running the AI model
    on a completely unusable image.
    """
    img = cv2.imread(image_path)

    if img is None:
        return {"valid": False, "reason": "Cannot read image file"}

    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return {"valid": False, "reason": "Image resolution too low (minimum 100x100)"}

    # Check brightness — too dark or washed out images are unreliable
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 20:
        return {"valid": False, "reason": "Image too dark"}
    if brightness > 240:
        return {"valid": False, "reason": "Image overexposed / too bright"}

    return {
        "valid": True,
        "width": w,
        "height": h,
        "brightness": round(float(brightness), 2)
    }