import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.deps import get_current_user
from app.models.user import User
from app.services.cv_service import analyze_eye_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eye-scan", tags=["Eye Scan"])


class EyeScanRequest(BaseModel):
    image: str


class EyeScanResponse(BaseModel):
    condition: str
    confidence: float
    risk_level: str
    description: str
    recommendation: str
    top3_predictions: list = []
    metrics: dict
    model_used: str = ""


@router.post("/analyze", response_model=EyeScanResponse)
async def analyze_eye(
    payload: EyeScanRequest,
    current_user: User = Depends(get_current_user)
):
    if not payload.image or len(payload.image) < 100:
        raise HTTPException(
            status_code=400,
            detail="Invalid image data. Please capture a clear eye image."
        )
    try:
        result = analyze_eye_image(payload.image)
        logger.info(
            f"Eye scan completed for user {current_user.id} "
            f"— condition: {result['condition']}"
        )
        return result
    except Exception as e:
        logger.error(f"Eye scan failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Eye analysis failed. Please retake the image and try again."
        )


@router.get("/health")
def eye_scan_health():
    try:
        import cv2
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        return {
            "status": "ok",
            "opencv_version": cv2.__version__,
            "message": "OpenCV is working correctly"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }