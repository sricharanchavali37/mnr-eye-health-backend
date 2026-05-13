import cv2
import numpy as np
import base64
import os
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "AMD",
    "Hypertension",
    "Myopia",
    "Other"
]

CLASS_INFO = {
    "Normal": {
        "risk_level": "low",
        "description": (
            "No obvious eye disease detected. Your eye appears healthy "
            "based on this screening. Continue with regular annual eye "
            "examinations to maintain good vision health."
        ),
        "recommendation": (
            "Maintain regular annual eye check-ups. Follow the 20-20-20 "
            "rule for screen time: every 20 minutes, look 20 feet away "
            "for 20 seconds."
        )
    },
    "Diabetes": {
        "risk_level": "high",
        "description": (
            "Indicators consistent with Diabetic Retinopathy detected. "
            "This condition occurs when high blood sugar damages blood "
            "vessels in the retina. Early detection is critical to "
            "prevent vision loss."
        ),
        "recommendation": (
            "Consult an ophthalmologist urgently for a dilated eye exam. "
            "Maintain strict blood sugar control. Schedule follow-up "
            "every 3-6 months."
        )
    },
    "Glaucoma": {
        "risk_level": "high",
        "description": (
            "Indicators consistent with Glaucoma detected. Glaucoma "
            "damages the optic nerve and can cause permanent vision loss "
            "if untreated. It is often called the silent thief of sight "
            "because early stages have no symptoms."
        ),
        "recommendation": (
            "Seek urgent ophthalmology evaluation for intraocular "
            "pressure measurement. Early treatment with eye drops or "
            "surgery can prevent progression."
        )
    },
    "Cataract": {
        "risk_level": "moderate",
        "description": (
            "Indicators consistent with Cataract detected. A cataract "
            "is a clouding of the eye's natural lens which leads to "
            "blurry vision. Cataracts are very common in older adults "
            "and are treatable with surgery."
        ),
        "recommendation": (
            "Schedule an appointment with an ophthalmologist to assess "
            "cataract severity. Cataract surgery is safe and highly "
            "effective with a quick recovery time."
        )
    },
    "AMD": {
        "risk_level": "high",
        "description": (
            "Indicators consistent with Age-related Macular Degeneration "
            "detected. AMD affects the central part of the retina and "
            "can cause loss of central vision. Early intervention can "
            "slow progression significantly."
        ),
        "recommendation": (
            "Consult a retinal specialist urgently. Anti-VEGF injections "
            "and lifestyle changes (diet, no smoking) can slow AMD. "
            "Use an Amsler grid daily to monitor changes."
        )
    },
    "Hypertension": {
        "risk_level": "moderate",
        "description": (
            "Indicators consistent with Hypertensive Retinopathy detected. "
            "High blood pressure can damage blood vessels in the retina "
            "causing vision problems. Controlling blood pressure is the "
            "primary treatment."
        ),
        "recommendation": (
            "Monitor blood pressure regularly and consult your physician "
            "to optimise blood pressure medication. Follow up with an "
            "ophthalmologist every 6 months."
        )
    },
    "Myopia": {
        "risk_level": "low",
        "description": (
            "Indicators consistent with Myopia (short-sightedness) "
            "detected. Myopia is very common and means distant objects "
            "appear blurry. It is correctable with glasses, contact "
            "lenses, or laser surgery."
        ),
        "recommendation": (
            "Schedule a refraction test with an optometrist to get the "
            "correct prescription. Spend more time outdoors to slow "
            "myopia progression, especially in children."
        )
    },
    "Other": {
        "risk_level": "moderate",
        "description": (
            "An eye condition outside the primary categories has been "
            "detected. This may include conditions such as retinal "
            "detachment risk, optic disc abnormalities, or other retinal "
            "pathologies that require professional evaluation."
        ),
        "recommendation": (
            "Schedule a comprehensive eye examination with an "
            "ophthalmologist to identify and treat the specific condition."
        )
    }
}

# ── Model loading ─────────────────────────────────────────────────────────────
_model = None
WEIGHTS_PATH = "eye_weights.weights.h5"
GDRIVE_FILE_ID = "1pT1tv8UNTLU8UTDA5fw9BnpkkxDE8y9O"


def download_weights_from_gdrive():
    """Download weights file from Google Drive if not present."""
    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Weights already exist at {WEIGHTS_PATH}")
        return True

    logger.info("Downloading weights from Google Drive...")
    try:
        import requests
        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = session.get(url, stream=True)

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm={value}"
                response = session.get(url, stream=True)
                break

        if response.status_code != 200:
            logger.error(f"Download failed: HTTP {response.status_code}")
            return False

        total = 0
        with open(WEIGHTS_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

        size_mb = total / (1024 * 1024)
        logger.info(f"Weights downloaded: {size_mb:.1f} MB")
        return True

    except Exception as e:
        logger.error(f"Weights download failed: {e}")
        if os.path.exists(WEIGHTS_PATH):
            os.remove(WEIGHTS_PATH)
        return False


def build_model():
    """
    Rebuild MobileNetV2 architecture in code.
    No file format issues — architecture defined here not loaded from file.
    Must match exactly what was used during training.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model

    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(8, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=output)
    return model


def load_model():
    """
    Build architecture in code, download weights, load weights.
    Bypasses all Keras version compatibility issues completely.
    """
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(WEIGHTS_PATH):
        success = download_weights_from_gdrive()
        if not success:
            logger.warning("Weights unavailable. Using rule-based fallback.")
            return None

    try:
        import tensorflow as tf
        logger.info("Building MobileNetV2 architecture...")
        model = build_model()
        logger.info("Loading weights into model...")
        model.load_weights(WEIGHTS_PATH)
        _model = model
        logger.info("Model ready — weights loaded successfully!")
        return _model
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return None


# ── Image processing ──────────────────────────────────────────────────────────
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to OpenCV BGR array."""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def detect_eye_region(image: np.ndarray) -> np.ndarray:
    """Detect and crop the eye region using Haar Cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        logger.info(f"Eye region detected at x={x} y={y} w={w} h={h}")
        return image[y1:y2, x1:x2]
    logger.warning("No eye region detected — using full image")
    return image


def check_image_quality(image: np.ndarray) -> dict:
    """Check if image is usable for analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    issues = []
    if blur_score < 10:
        issues.append("too blurry")
    if brightness < 30:
        issues.append("too dark")
    if brightness > 240:
        issues.append("too bright / overexposed")

    return {
        "usable": len(issues) == 0,
        "issues": issues,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2)
    }


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """Resize and normalise image for MobileNetV2 input."""
    resized = cv2.resize(image, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


# ── Classification ────────────────────────────────────────────────────────────
def classify_with_model(image: np.ndarray) -> dict:
    """Run ML model prediction on preprocessed image."""
    model = load_model()
    if model is None:
        return classify_with_rules(image)

    try:
        img_input = preprocess_for_model(image)
        predictions = model.predict(img_input, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx]) * 100

        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                "condition": CLASS_NAMES[i],
                "confidence": round(float(predictions[i]) * 100, 1)
            }
            for i in top3_idx
        ]

        condition = CLASS_NAMES[top_idx]
        info = CLASS_INFO[condition]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_ratio = round(
            (np.sum((mask1 + mask2) > 0) / mask1.size) * 100, 2
        )
        brightness = round(float(np.mean(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )), 2)
        sharpness = round(float(cv2.Laplacian(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            cv2.CV_64F
        ).var()), 2)

        logger.info(
            f"Model prediction: {condition} "
            f"({confidence:.1f}%) | top3: {top3}"
        )

        return {
            "condition": condition,
            "confidence": round(confidence, 1),
            "risk_level": info["risk_level"],
            "description": info["description"],
            "recommendation": info["recommendation"],
            "top3_predictions": top3,
            "metrics": {
                "redness": red_ratio,
                "brightness": brightness,
                "sharpness": sharpness
            },
            "model_used": "MobileNetV2 (trained on ODIR-5K)"
        }

    except Exception as e:
        logger.error(f"Model prediction failed: {e} — using rules")
        return classify_with_rules(image)


def classify_with_rules(image: np.ndarray) -> dict:
    """Rule-based fallback when model is unavailable."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    redness = round(
        (np.sum((mask1 + mask2) > 0) / mask1.size) * 100, 2
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = round(float(np.mean(gray)), 2)
    sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    if redness > 15:
        condition = "Other"
        info = CLASS_INFO["Other"]
    elif brightness < 60:
        condition = "Cataract"
        info = CLASS_INFO["Cataract"]
    else:
        condition = "Normal"
        info = CLASS_INFO["Normal"]

    return {
        "condition": condition,
        "confidence": 60.0,
        "risk_level": info["risk_level"],
        "description": info["description"],
        "recommendation": info["recommendation"],
        "top3_predictions": [
            {"condition": condition, "confidence": 60.0}
        ],
        "metrics": {
            "redness": redness,
            "brightness": brightness,
            "sharpness": sharpness
        },
        "model_used": "Rule-based fallback"
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def analyze_eye_image(base64_image: str) -> dict:
    """
    Full pipeline:
    1. Decode base64 image
    2. Check image quality
    3. Detect eye region
    4. Run ML model (or fallback)
    5. Return result
    """
    image = decode_base64_image(base64_image)

    quality = check_image_quality(image)
    if not quality["usable"]:
        return {
            "condition": "Image Quality Issue",
            "confidence": 95.0,
            "risk_level": "low",
            "description": (
                f"Image could not be analysed: {', '.join(quality['issues'])}. "
                "Please retake in good lighting with a steady hand."
            ),
            "recommendation": (
                "Move to a well-lit area and retake the photo. "
                "Ensure the camera is steady and your eye is in focus."
            ),
            "top3_predictions": [],
            "metrics": {
                "redness": 0,
                "brightness": quality["brightness"],
                "sharpness": quality["blur_score"]
            },
            "model_used": "Quality check"
        }

    eye_region = detect_eye_region(image)
    result = classify_with_model(eye_region)
    return result