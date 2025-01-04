import os
import pickle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_models():
    # Load all models: gesture recognizer, body language, and emotion detection.
    
    # Load Gesture Recognizer
    model_path = os.path.join('models', 'gesture_recognizer.task')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Gesture recognizer model not found at {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(options)

    # Load Body Language Model
    with open(os.path.join('models', 'body_language.pkl'), 'rb') as f:
        body_language_model = pickle.load(f)

    # Load Emotion Detection Model
    with open(os.path.join('models', 'emotion_detection.pkl'), 'rb') as f:
        emotion_detection_model = pickle.load(f)

    return gesture_recognizer, body_language_model, emotion_detection_model
