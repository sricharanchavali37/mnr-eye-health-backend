import json
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

_gemini_client = None


def _get_client():
    global _gemini_client
    if _gemini_client is None and settings.GEMINI_API_KEY and settings.GEMINI_API_KEY not in ("", "dummy-key-replace-with-real-key"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            _gemini_client = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini client: {e}")
    return _gemini_client


def _build_prompt(screening_data: dict, risk_score: float, risk_level: str, flags: list) -> str:
    symptoms = screening_data.get("symptoms") or []
    if isinstance(symptoms, str):
        try:
            symptoms = json.loads(symptoms)
        except Exception:
            symptoms = []

    return f"""You are a compassionate, professional eye health AI assistant for MNR Eye Health Platform.

A patient has completed a digital vision screening. Based on the data below, provide:
1. A clear, empathetic 2-3 sentence summary of their eye health status
2. 4-6 specific, actionable, personalized recommendations
3. Whether urgent referral to an eye specialist is needed (true/false and why)

SCREENING DATA:
- Age: {screening_data.get('age', 'Not provided')}
- Visual Acuity Left: {screening_data.get('visual_acuity_left', 'Not tested')}
- Visual Acuity Right: {screening_data.get('visual_acuity_right', 'Not tested')}
- Color Vision Score: {screening_data.get('color_vision_score', 'Not tested')}/100
- Contrast Sensitivity: {screening_data.get('contrast_sensitivity', 'Not tested')}
- Field of View Score: {screening_data.get('field_of_view_score', 'Not tested')}/100
- Symptoms: {', '.join(symptoms) if symptoms else 'None'}
- Symptom Duration: {screening_data.get('symptom_duration_days', 'Unknown')} days
- Last Eye Exam: {screening_data.get('last_eye_exam_months', 'Unknown')} months ago
- Has Diabetes: {screening_data.get('has_diabetes', False)}
- Has Hypertension: {screening_data.get('has_hypertension', False)}
- Family History: {screening_data.get('family_history', False)}
- Daily Screen Time: {screening_data.get('screen_time_hours', 'Unknown')} hours

RISK ASSESSMENT:
- Risk Score: {risk_score}/100
- Risk Level: {risk_level.upper()}
- Clinical Flags: {', '.join(flags) if flags else 'None'}

Respond ONLY with this exact JSON format (no markdown, no extra text):
{{
  "summary": "...",
  "recommendations": ["...", "...", "...", "...", "..."],
  "urgent_referral": true or false,
  "referral_reason": "..."
}}"""


def _fallback_recommendation(risk_level: str, flags: list) -> dict:
    level_messages = {
        "low": "Your vision screening results appear to be within normal range. Continue with regular preventive eye care and maintain healthy vision habits.",
        "moderate": "Your screening indicates some areas that need attention. While not immediately urgent, we recommend scheduling an eye examination soon to address the identified concerns.",
        "high": "Your screening has identified significant vision concerns. We strongly recommend scheduling an appointment with an eye specialist within the next 2-4 weeks.",
        "critical": "Your screening has detected critical vision risk indicators. Please seek immediate professional eye care evaluation as soon as possible."
    }

    base_recs = [
        "Schedule a comprehensive eye examination with a qualified optometrist or ophthalmologist",
        "Follow the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds",
        "Ensure adequate lighting when reading or using screens",
        "Stay hydrated and maintain a diet rich in vitamins A, C, E and omega-3 fatty acids",
        "Wear UV-protective sunglasses when outdoors",
        "Complete the next MNR screening in 3-6 months to track your vision health"
    ]

    if risk_level in ("high", "critical"):
        base_recs.insert(0, "Seek professional eye care evaluation as soon as possible")

    return {
        "summary": level_messages.get(risk_level, level_messages["moderate"]),
        "recommendations": base_recs[:5],
        "urgent_referral": risk_level in ("high", "critical"),
        "referral_reason": "Risk score and clinical flags indicate need for professional evaluation" if risk_level in ("high", "critical") else "Routine preventive care recommended"
    }


async def get_ai_analysis(screening_data: dict, risk_score: float, risk_level: str, flags: list) -> dict:
    client = _get_client()

    if client:
        try:
            prompt = _build_prompt(screening_data, risk_score, risk_level, flags)
            response = client.generate_content(prompt)
            text = response.text.strip()

            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
            text = text.strip()

            result = json.loads(text)
            assert "summary" in result
            assert "recommendations" in result
            assert isinstance(result["recommendations"], list)
            return result

        except Exception as e:
            logger.error(f"Gemini API error: {e} — using fallback")

    return _fallback_recommendation(risk_level, flags)
