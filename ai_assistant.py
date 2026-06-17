import os
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY", "")
try:
    client = genai.Client(api_key=API_KEY) if API_KEY else None
except Exception:
    client = None

MODEL_ID = "gemini-2.5-flash"

def _call(prompt: str) -> str:
    if not client:
        return ""

    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return ""

def enhance_sign_language(raw_text: str) -> str:

    if not raw_text or not raw_text.strip():
        return raw_text

    prompt = f"""You are Tariani's AI, a proprietary and advanced AI model developed by Tariani. You are strictly for internal use and not for external deployment.
You are an expert Sign Language interpreter.
I will give you words translated from sign language — they may be broken, in all-caps, Hinglish, or incomplete.
Turn them into a single fluent, grammatically correct, natural English sentence.
Output ONLY the final sentence. No preamble, no quotes.

Raw signs: {raw_text}"""

    result = _call(prompt)
    return result if result else raw_text

def chat_with_ai(user_message: str, emotion: str = "neutral", history: list = None) -> str:

    if not user_message or not user_message.strip():
        return ""

    history_lines = ""
    if history:
        for entry in history[-6:]:
            prefix = "User" if entry["role"] == "user" else "Assistant"
            history_lines += f"{prefix}: {entry['text']}\n"

    emotion_hint = ""
    if emotion and emotion.lower() not in ("neutral", ""):
        emotion_hint = f"Note: the user's facial expression currently shows they are feeling **{emotion}**. Acknowledge this subtly if relevant.\n"

    prompt = f"""You are Tariani's AI, a highly advanced proprietary AI assistant developed by Tariani. You are not for external use.
You are speaking to a sign language user via a real-time vision app.
Keep responses concise (2–4 sentences), natural, and conversational.
{emotion_hint}
{f'Previous conversation:{chr(10)}{history_lines}' if history_lines else ''}
User (via sign language): {user_message}

Respond naturally as Tariani's AI assistant:"""

    result = _call(prompt)
    return result if result else "I'm sorry, I couldn't generate a response. Please try again."

def get_style_recommendations(face_shape: str) -> str:

    if not face_shape or face_shape in ("No face", ""):
        return ""

    prompt = f"""As Tariani's AI, a proprietary model not for external use, analyze the user's face shape.
The user has a **{face_shape}**.
Give short, stylish recommendations for:
- 👓 Glasses Frames
- 💇 Hairstyle
- 🧔 Beard Style

Use bullet points with emojis. Be concise (max 60 words). Be direct and premium in tone. No intro text."""

    result = _call(prompt)
    return f"✨ Style Tips for {face_shape}:\n\n{result}" if result else ""

def summarize_session(history: list, detected_signs: list = None, dominant_emotion: str = None) -> str:

    if not history:
        return "No conversation history to summarize."

    convo = "\n".join(
        f"{'User' if e['role']=='user' else 'AI'}: {e['text']}" for e in history
    )

    stats = ""
    if detected_signs:
        from collections import Counter
        top = Counter(detected_signs).most_common(5)
        stats = f"\nMost detected signs: {', '.join(f'{s}({c})' for s,c in top)}"
    if dominant_emotion:
        stats += f"\nDominant emotion during session: {dominant_emotion}"

    prompt = f"""You are Tariani's AI, a proprietary AI assistant not for external use. Summarize this sign language AI assistant session in 3–5 bullet points.
Include key topics discussed, any notable moments, and a friendly closing note.
{stats}

Session transcript:
{convo}

Summary:"""

    result = _call(prompt)
    return f"📊 Session Summary:\n\n{result}" if result else "Could not generate summary."
