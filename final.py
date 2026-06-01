from flask import Flask, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow import keras
from collections import deque
import time
from datetime import datetime
import os
import json
import threading
import atexit

app = Flask(__name__)

# Configuration
SEQUENCE_LENGTH = 60
NUM_FEATURES = 63
WORD_BUFFER_SIZE = 30  # Same as training model

# Paths - ABSOLUTE PATHS (hardcoded to where your models are)
BASE_DIR = r"F:\new baas"

# Models are in the Model subdirectory inside HandSignDetector
WORD_MODEL_PATH = r"C:\Users\Samud Rjkr\Downloads\HandSignDetector\HandSignDetector\Model\word_model.h5"
WORD_LABEL_PATH = r"C:\Users\Samud Rjkr\Downloads\HandSignDetector\HandSignDetector\Model\word_label_mapping.pkl"

LETTER_MODEL_PATH = r"C:\Users\Samud Rjkr\Downloads\HandSignDetector\HandSignDetector\Model\letter_model.h5"
LETTER_LABEL_PATH = r"C:\Users\Samud Rjkr\Downloads\HandSignDetector\HandSignDetector\Model\letter_label_mapping.pkl"
# UI Colors (BGR format)
COLOR_BG = (40, 40, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_WORD = (0, 255, 0)
COLOR_LETTER = (100, 200, 255)
COLOR_CONFIDENCE_HIGH = (0, 255, 0)
COLOR_CONFIDENCE_MED = (0, 165, 255)
COLOR_CONFIDENCE_LOW = (0, 0, 255)
COLOR_AUTO = (255, 165, 0)  # Orange for automatic mode

class WebHybridTranslator:
    """Web-enabled hybrid translator with FULLY AUTOMATIC mode switching"""
    
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING TRANSLATOR")
        print("="*70)
        
        # Load models with better error handling
        self.word_model = None
        self.word_mapping = {}
        self.letter_model = None
        self.letter_mapping = {}
        
        # Try loading word model
        print(f"\n1. WORD MODEL")
        print(f"   Path: {WORD_MODEL_PATH}")
        print(f"   File exists: {os.path.exists(WORD_MODEL_PATH)}")
        print(f"   Label exists: {os.path.exists(WORD_LABEL_PATH)}")
        
        if os.path.exists(WORD_MODEL_PATH) and os.path.exists(WORD_LABEL_PATH):
            try:
                print(f"   Loading model...")
                self.word_model = keras.models.load_model(
                    WORD_MODEL_PATH, compile=False
                )
                print(f"   Loading labels...")
                with open(WORD_LABEL_PATH, 'rb') as f:
                    self.word_mapping = pickle.load(f)
                print(f"   ✓ Word model loaded successfully!")
                print(f"   - Classes: {len(self.word_mapping)}")
            except Exception as e:
                print(f"   ✗ Error loading word model: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ✗ Word model files not found")
        
        # Try loading letter model
        print(f"\n2. LETTER MODEL")
        print(f"   Path: {LETTER_MODEL_PATH}")
        print(f"   File exists: {os.path.exists(LETTER_MODEL_PATH)}")
        print(f"   Label exists: {os.path.exists(LETTER_LABEL_PATH)}")
        
        if os.path.exists(LETTER_MODEL_PATH) and os.path.exists(LETTER_LABEL_PATH):
            try:
                print(f"   Loading model...")
                self.letter_model = keras.models.load_model(
                    LETTER_MODEL_PATH, compile=False
                )
                print(f"   Loading labels...")
                with open(LETTER_LABEL_PATH, 'rb') as f:
                    self.letter_mapping = pickle.load(f)
                print(f"   ✓ Letter model loaded successfully!")
                print(f"   - Classes: {len(self.letter_mapping)}")
            except Exception as e:
                print(f"   ✗ Error loading letter model: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ✗ Letter model files not found")
        
        # Check if at least one model loaded
        print(f"\n3. RESULTS")
        print(f"   Word model loaded: {self.word_model is not None}")
        print(f"   Letter model loaded: {self.letter_model is not None}")
        
        if self.word_model is None and self.letter_model is None:
            print(f"\n   ✗ CRITICAL ERROR: No models loaded!")
            print(f"   Check the error messages above for details.")
            print("="*70 + "\n")
            raise Exception(f"No models loaded! Check paths:\nWord: {WORD_MODEL_PATH}\nLetter: {LETTER_MODEL_PATH}")
        
        print("="*70 + "\n")
        
        # Initialize MediaPipe - EXACT same settings as training
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Mode - Always start with LETTER if available, else WORD
        self.mode = "LETTER" if self.letter_model else "WORD"
        self.sequence_buffer = deque(maxlen=WORD_BUFFER_SIZE)
        
        # Detection state
        self.current_detection = ""
        self.current_confidence = 0.0
        self.detection_type = ""
        
        # Sentence
        self.sentence = []
        self.last_add_time = 0
        self.add_cooldown = 1.0
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        # Letter hold
        self.letter_hold_start = 0
        self.letter_hold_duration = 1.2
        self.last_letter = ""
        
        # Camera
        self.camera = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # IMPROVED Word detection
        self.last_word_time = 0
        self.word_check_interval = 0.15  # Faster checks (was 0.3s)
        self.confidence_threshold = 0.70  # Higher confidence for accuracy (was 0.6)
        
        # Word stability tracking
        self.last_predicted_word = ""
        self.word_confidence_history = deque(maxlen=3)  # Track last 3 predictions
        self.stable_word_threshold = 2  # Need 2/3 same predictions for stability
        
        # ============ FULLY AUTOMATIC MODE SWITCHING ============
        self.auto_mode = True  # Always enabled, no manual toggle
        self.movement_history = deque(maxlen=10)  # Track hand movement over last 10 frames
        self.last_hand_position = None
        self.movement_threshold = 0.015  # Threshold for detecting movement (adjustable)
        self.still_frames_for_letter = 8  # Need 8 still frames to switch to LETTER mode
        self.moving_frames_for_word = 6  # Need 6 moving frames to switch to WORD mode
        self.mode_switch_cooldown = 1.0  # Cooldown between mode switches (seconds)
        self.last_mode_switch = 0

        # --- NEW: track whether a hand is currently detected ---
        self.hand_detected = False
    
    def cleanup(self):
        """Properly release all resources"""
        with self.lock:
            self.is_running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            if self.hands is not None:
                self.hands.close()
                self.hands = None
        print("✓ Camera and resources released")
    
    def normalize_landmarks(self, features):
        """EXACT normalization from training model"""
        features = features.reshape(21, 3)
        wrist = features[0].copy()
        features = features - wrist
        
        hand_size = np.linalg.norm(features[12] - features[0])
        if hand_size > 1e-6:
            features = features / hand_size
        
        return features.flatten()
    
    def extract_landmarks(self, frame):
        """Extract landmarks matching training model"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            features = []
            for lm in landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features)
            features = self.normalize_landmarks(features)
            return features, results.multi_hand_landmarks[0]
        return None, None
    
    def interpolate_to_60(self, buffer_30):
        """EXACT interpolation method from training model"""
        if len(buffer_30) < WORD_BUFFER_SIZE:
            return None
        
        # Convert to numpy array
        buffer_array = np.array(buffer_30[:WORD_BUFFER_SIZE])
        
        # Linear interpolation - EXACT same as training
        indices = np.linspace(0, len(buffer_array) - 1, SEQUENCE_LENGTH)
        
        sequence = np.array([
            np.interp(indices, np.arange(len(buffer_array)), buffer_array[:, i])
            for i in range(NUM_FEATURES)
        ]).T
        
        return sequence
    
    def calculate_hand_movement(self, hand_landmarks):
        """Calculate hand movement to determine if hand is moving or still"""
        # Get center of hand (average of all landmarks)
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        z_coords = [lm.z for lm in hand_landmarks.landmark]
        
        current_position = np.array([np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)])
        
        if self.last_hand_position is not None:
            # Calculate Euclidean distance
            movement = np.linalg.norm(current_position - self.last_hand_position)
            self.movement_history.append(movement)
        
        self.last_hand_position = current_position
        
        # Determine if hand is moving or still
        if len(self.movement_history) >= 5:
            avg_movement = np.mean(list(self.movement_history)[-5:])
            return avg_movement > self.movement_threshold
        
        return False
    
    def auto_switch_mode(self, is_moving):
        """Automatically switch between WORD and LETTER mode based on hand movement"""
        current_time = time.time()
        
        # Check cooldown to prevent rapid switching
        if current_time - self.last_mode_switch < self.mode_switch_cooldown:
            return
        
        # Count recent movement states
        if len(self.movement_history) >= 10:
            recent_movements = list(self.movement_history)[-10:]
            moving_count = sum(1 for m in recent_movements if m > self.movement_threshold)
            still_count = 10 - moving_count
            
            # Switch to WORD mode if hand is moving consistently
            if moving_count >= self.moving_frames_for_word and self.mode == "LETTER" and self.word_model:
                self.mode = "WORD"
                self.sequence_buffer.clear()
                self.current_detection = ""
                self.current_confidence = 0.0
                self.word_confidence_history.clear()
                self.last_mode_switch = current_time
                print("🔄 Auto-switched to WORD mode (movement detected)")
            
            # Switch to LETTER mode if hand is still
            elif still_count >= self.still_frames_for_letter and self.mode == "WORD" and self.letter_model:
                self.mode = "LETTER"
                self.sequence_buffer.clear()
                self.current_detection = ""
                self.current_confidence = 0.0
                self.word_confidence_history.clear()
                self.last_mode_switch = current_time
                print("🔄 Auto-switched to LETTER mode (hand still)")
    
    def predict_letter(self, features):
        if self.letter_model is None:
            return None, 0.0
        features_input = np.expand_dims(features, axis=0)
        predictions = self.letter_model.predict(features_input, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        letter = self.letter_mapping[predicted_idx]
        return letter, confidence
    
    def predict_word(self):
        """IMPROVED prediction with stability checking"""
        if self.word_model is None or len(self.sequence_buffer) < WORD_BUFFER_SIZE:
            return None, 0.0
        
        # Get exactly 30 frames from buffer
        buffer_30 = list(self.sequence_buffer)[:WORD_BUFFER_SIZE]
        
        # Interpolate to 60 frames - EXACT same method as training
        sequence = self.interpolate_to_60(buffer_30)
        
        if sequence is None:
            return None, 0.0
        
        # Predict
        sequence = np.expand_dims(sequence, axis=0)
        predictions = self.word_model.predict(sequence, verbose=0)[0]
        
        # Get top prediction
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        word = self.word_mapping[predicted_idx]
        
        # Check prediction stability - must be consistent across multiple checks
        self.word_confidence_history.append((word, confidence))
        
        # Count how many times the current word appears in recent history
        if len(self.word_confidence_history) >= 2:
            word_counts = {}
            for w, c in self.word_confidence_history:
                if c >= self.confidence_threshold:  # Only count high-confidence predictions
                    word_counts[w] = word_counts.get(w, 0) + 1
            
            # Check if current word is stable (appears multiple times)
            if word_counts.get(word, 0) >= self.stable_word_threshold:
                return word, confidence
            else:
                # Not stable enough, return None to avoid premature detection
                return None, 0.0
        
        return word, confidence
    
    def get_hand_center(self, hand_landmarks, frame_shape):
        h, w = frame_shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords)) - 80
        return center_x, center_y
    
    def draw_floating_text(self, frame, text, position, confidence, detection_type):
        x, y = position
        text_color = COLOR_LETTER if detection_type == "LETTER" else COLOR_WORD
        
        if confidence > 0.7:
            conf_color = COLOR_CONFIDENCE_HIGH
        elif confidence > 0.5:
            conf_color = COLOR_CONFIDENCE_MED
        else:
            conf_color = COLOR_CONFIDENCE_LOW
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0 if detection_type == "LETTER" else 1.5
        thickness = 4
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        padding = 20
        bg_x1 = x - text_w // 2 - padding
        bg_y1 = y - text_h - padding
        bg_x2 = x + text_w // 2 + padding
        bg_y2 = y + padding
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), text_color, 3)
        
        text_x = x - text_w // 2
        text_y = y
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        conf_text = f"{confidence*100:.0f}%"
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, 0.6, 2)
        conf_x = x - conf_w // 2
        conf_y = bg_y2 + conf_h + 10
        cv2.putText(frame, conf_text, (conf_x, conf_y), font, 0.6, conf_color, 2)
        
        type_text = f"[{detection_type}]"
        (type_w, type_h), _ = cv2.getTextSize(type_text, font, 0.5, 2)
        type_x = x - type_w // 2
        type_y = bg_y1 - 10
        cv2.putText(frame, type_text, (type_x, type_y), font, 0.5, (200, 200, 200), 2)
    
    def draw_mode_indicator(self, frame):
        h, w = frame.shape[:2]
        mode_color = COLOR_LETTER if self.mode == "LETTER" else COLOR_WORD
        
        # Display mode with AUTO indicator
        mode_text = f"Mode: {self.mode} [AUTO]"
        cv2.putText(frame, mode_text, (w-320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_AUTO, 2)
        
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w-320, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        
        # Show buffer status for word mode
        if self.mode == "WORD":
            buffer_pct = (len(self.sequence_buffer) / WORD_BUFFER_SIZE) * 100
            buffer_color = COLOR_WORD if len(self.sequence_buffer) >= WORD_BUFFER_SIZE else (0, 165, 255)
            cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%", (w-320, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, buffer_color, 2)
        
        # --- CHANGED: Show "No Hand" when no hand is detected, otherwise Moving/Still ---
        if not self.hand_detected:
            cv2.putText(frame, "Hand: No Hand", (w-320, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        elif len(self.movement_history) > 0:
            avg_movement = np.mean(list(self.movement_history)[-5:]) if len(self.movement_history) >= 5 else 0
            is_moving = avg_movement > self.movement_threshold
            movement_text = "Moving" if is_moving else "Still"
            movement_color = COLOR_WORD if is_moving else COLOR_LETTER
            cv2.putText(frame, f"Hand: {movement_text}", (w-320, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, movement_color, 2)
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        
        # FPS
        self.frame_count += 1
        if time.time() - self.last_fps_update >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = time.time()
        
        features, hand_landmarks = self.extract_landmarks(frame)
        
        if features is not None:
            # --- NEW: mark hand as detected ---
            self.hand_detected = True

            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # ============ AUTOMATIC MODE SWITCHING ============
            is_moving = self.calculate_hand_movement(hand_landmarks)
            self.auto_switch_mode(is_moving)
            
            if self.mode == "LETTER":
                letter, letter_conf = self.predict_letter(features)
                if letter and letter_conf > 0.5:
                    current_time = time.time()
                    if letter == self.last_letter:
                        hold_time = current_time - self.letter_hold_start
                        if hold_time >= self.letter_hold_duration:
                            self.current_detection = letter
                            self.current_confidence = letter_conf
                            self.detection_type = "LETTER"
                    else:
                        self.last_letter = letter
                        self.letter_hold_start = current_time
                        self.current_detection = letter
                        self.current_confidence = letter_conf * 0.7
                        self.detection_type = "LETTER"
            
            elif self.mode == "WORD":
                # Add features to buffer
                self.sequence_buffer.append(features)
                
                # Check for word prediction at faster intervals
                current_time = time.time()
                if (len(self.sequence_buffer) >= WORD_BUFFER_SIZE and 
                    current_time - self.last_word_time >= self.word_check_interval):
                    
                    word, word_conf = self.predict_word()
                    self.last_word_time = current_time
                    
                    # Only update if we got a stable, high-confidence prediction
                    if word and word_conf >= self.confidence_threshold:
                        self.current_detection = word
                        self.current_confidence = word_conf
                        self.detection_type = "WORD"
                        self.last_predicted_word = word
            
            if self.current_detection:
                hand_center = self.get_hand_center(hand_landmarks, frame.shape)
                self.draw_floating_text(frame, self.current_detection, hand_center,
                                      self.current_confidence, self.detection_type)
        else:
            # --- NEW: mark hand as NOT detected ---
            self.hand_detected = False

            self.letter_hold_start = 0
            self.last_letter = ""
            self.last_hand_position = None  # Reset hand position when no hand detected
            # Clear word history when hand is not detected
            if self.mode == "WORD":
                self.word_confidence_history.clear()
        
        self.draw_mode_indicator(frame)
        return frame

# Global translator instance
translator = None
camera_lock = threading.Lock()

def cleanup_resources():
    """Cleanup function called on exit"""
    global translator
    if translator is not None:
        translator.cleanup()

# Register cleanup
atexit.register(cleanup_resources)

def initialize_translator():
    """Initialize translator with proper error handling"""
    global translator
    if translator is None:
        try:
            translator = WebHybridTranslator()
            print("✓ Translator initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Error initializing translator: {e}")
            import traceback
            traceback.print_exc()
            return False
    return True

@app.route('/')
def index():
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Translator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 48px;
            font-weight: 300;
            margin-bottom: 10px;
        }

        .header .highlight {
            font-weight: 600;
        }

        .header .subtitle {
            font-size: 18px;
            opacity: 0.9;
        }

        .error-banner {
            background: #ff6b6b;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            display: none;
        }

        .error-banner.show {
            display: block;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }

        .video-section {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }

        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }

        .control-panel {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .stat-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: #333;
        }

        .mode-indicator {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }

        .mode-letter {
            background: linear-gradient(135deg, #64b4ff 0%, #4a9eff 100%);
            color: white;
        }

        .mode-word {
            background: linear-gradient(135deg, #00d984 0%, #00b871 100%);
            color: white;
        }

        .auto-badge {
            display: inline-block;
            background: linear-gradient(135deg, #ff9a56 0%, #ff6a00 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }

        .btn {
            width: 100%;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #e9ecef;
            color: #333;
        }

        .btn:hover {
            background: #dee2e6;
            transform: translateY(-2px);
        }

        /* ===== LEARN BUTTON ===== */
        .btn-learn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-bottom: 12px;
        }

        .btn-learn:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a3f96 100%);
            transform: translateY(-2px);
        }

        /* ===== LEARN MODAL ===== */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.75);
            z-index: 1000;
            overflow-y: auto;
            padding: 20px;
        }

        .modal-overlay.active {
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }

        .modal {
            background: #fff;
            border-radius: 16px;
            width: 100%;
            max-width: 1100px;
            margin: auto;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }

        .modal-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 24px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .modal-header h2 {
            color: white;
            font-size: 26px;
            font-weight: 300;
            letter-spacing: 1px;
        }

        .modal-header h2 span {
            font-weight: 700;
        }

        .modal-close {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            font-size: 22px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }

        .modal-close:hover {
            background: rgba(255,255,255,0.35);
        }

        .modal-tabs {
            display: flex;
            border-bottom: 2px solid #e9ecef;
            background: #f8f9fa;
        }

        .tab-btn {
            flex: 1;
            padding: 16px;
            border: none;
            background: transparent;
            font-size: 15px;
            font-weight: 600;
            color: #888;
            cursor: pointer;
            transition: all 0.2s;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
        }

        .tab-btn.active {
            color: #764ba2;
            border-bottom-color: #764ba2;
            background: white;
        }

        .tab-btn:hover:not(.active) {
            color: #555;
            background: #eee;
        }

        .modal-body {
            padding: 30px;
        }

        .tab-panel {
            display: none;
        }

        .tab-panel.active {
            display: block;
        }

        /* ===== ALPHABET GRID ===== */
        .section-title {
            font-size: 14px;
            font-weight: 700;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 20px;
        }

        .alphabet-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
            gap: 12px;
        }

        .letter-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: center;
        }

        .letter-card:hover {
            border-color: #764ba2;
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(118,75,162,0.2);
        }

        .letter-card .letter-img-wrap {
            width: 100%;
            aspect-ratio: 1;
            background: #e9ecef;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .letter-card .letter-img-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .letter-card .letter-img-wrap .letter-placeholder {
            font-size: 28px;
            font-weight: 700;
            color: #764ba2;
            opacity: 0.4;
        }

        .letter-card .letter-label {
            padding: 6px;
            font-size: 18px;
            font-weight: 700;
            color: #333;
        }

        /* ===== WORD GRID ===== */
        .word-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 16px;
        }

        .word-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .word-card:hover {
            border-color: #667eea;
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(102,126,234,0.2);
        }

        .word-card .video-thumb {
            width: 100%;
            aspect-ratio: 16/9;
            background: #1a1a2e;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }

        .word-card .video-thumb video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .word-card .video-thumb .play-icon {
            position: absolute;
            width: 48px;
            height: 48px;
            background: rgba(255,255,255,0.9);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: #764ba2;
            pointer-events: none;
            transition: transform 0.2s;
        }

        .word-card:hover .play-icon {
            transform: scale(1.1);
        }

        .word-card .word-label {
            padding: 12px 16px;
            font-size: 16px;
            font-weight: 700;
            color: #333;
            text-transform: capitalize;
        }

        /* ===== LIGHTBOX for letter images ===== */
        .lightbox {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.92);
            z-index: 2000;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        .lightbox.active {
            display: flex;
        }

        .lightbox img {
            max-width: 80vw;
            max-height: 75vh;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }

        .lightbox-label {
            color: white;
            font-size: 48px;
            font-weight: 700;
            margin-top: 20px;
        }

        .lightbox-close {
            position: absolute;
            top: 20px;
            right: 30px;
            background: rgba(255,255,255,0.15);
            border: none;
            color: white;
            font-size: 28px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .lightbox-close:hover {
            background: rgba(255,255,255,0.3);
        }

        /* ===== VIDEO MODAL ===== */
        .video-modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.92);
            z-index: 2000;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        .video-modal.active {
            display: flex;
        }

        .video-modal video {
            max-width: 80vw;
            max-height: 70vh;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }

        .video-modal-label {
            color: white;
            font-size: 32px;
            font-weight: 700;
            margin-top: 20px;
            text-transform: capitalize;
        }

        .video-modal-close {
            position: absolute;
            top: 20px;
            right: 30px;
            background: rgba(255,255,255,0.15);
            border: none;
            color: white;
            font-size: 28px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-modal-close:hover {
            background: rgba(255,255,255,0.3);
        }

        .media-note {
            background: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 13px;
            color: #856404;
            margin-bottom: 20px;
        }

        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SIGN<span class="highlight">TRANSLATOR</span></h1>
            <p class="subtitle">Automatic Sign Language Translation System</p>
        </div>

        <div class="error-banner" id="errorBanner">
            ⚠️ System initialization error. Please check model files.
        </div>

        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img src="/video_feed" alt="Video Stream" id="videoFeed" 
                         onerror="handleVideoError()">
                </div>
            </div>

            <div class="control-panel">
                <div class="stat-box">
                    <div class="stat-label">Current Mode</div>
                    <div class="stat-value">
                        <span id="modeIndicator" class="mode-indicator mode-letter">LETTER</span>
                        <span class="auto-badge">AUTO</span>
                    </div>
                </div>

                <div class="stat-box">
                    <div class="stat-label">Current Detection</div>
                    <div class="stat-value" id="currentDetection">-</div>
                </div>

                <!-- LEARN BUTTON -->
                <button class="btn btn-learn" onclick="openLearnModal()">📚 Learn Sign Language</button>

                <button class="btn" onclick="clearSentence()">Clear Detection</button>
            </div>
        </div>
    </div>

    <!-- ===== LEARN MODAL ===== -->
    <div class="modal-overlay" id="learnModal">
        <div class="modal">
            <div class="modal-header">
                <h2>LEARN <span>SIGN LANGUAGE</span></h2>
                <button class="modal-close" onclick="closeLearnModal()">✕</button>
            </div>
            <div class="modal-tabs">
                <button class="tab-btn active" onclick="switchTab('alphabet')">🔤 Alphabet (A–Z)</button>
                <button class="tab-btn" onclick="switchTab('words')">🖐 Words</button>
            </div>
            <div class="modal-body">

                <!-- ALPHABET TAB -->
                <div class="tab-panel active" id="tab-alphabet">
                    <div class="media-note">
                       
                    </div>
                    <div class="section-title">American Sign Language — Alphabet</div>
                    <div class="alphabet-grid" id="alphabetGrid"></div>
                </div>

                <!-- WORDS TAB -->
                <div class="tab-panel" id="tab-words">
                    <div class="media-note">
                        
                    </div>
                    <div class="section-title">Common Words</div>
                    <div class="word-grid" id="wordGrid"></div>
                </div>

            </div>
        </div>
    </div>

    <!-- LETTER LIGHTBOX -->
    <div class="lightbox" id="letterLightbox" onclick="closeLightbox()">
        <button class="lightbox-close" onclick="closeLightbox()">✕</button>
        <img id="lightboxImg" src="" alt="">
        <div class="lightbox-label" id="lightboxLabel"></div>
    </div>

    <!-- VIDEO FULLSCREEN MODAL -->
    <div class="video-modal" id="videoModal">
        <button class="video-modal-close" onclick="closeVideoModal()">✕</button>
        <video id="modalVideo" controls autoplay loop style="max-width:80vw;max-height:70vh;border-radius:12px;"></video>
        <div class="video-modal-label" id="videoModalLabel"></div>
    </div>

    <script>
        // ===== TRANSLATOR STATE =====
        let updateInterval;
        let videoErrorCount = 0;

        function handleVideoError() {
            videoErrorCount++;
            if (videoErrorCount > 3) {
                document.getElementById('errorBanner').classList.add('show');
            }
        }

        function updateState() {
            fetch('/get_state')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.warn("Translator not ready yet");
                        return;
                    }

                    const modeEl = document.getElementById('modeIndicator');
                    modeEl.textContent = data.mode;
                    modeEl.className = 'mode-indicator ' + (data.mode === 'LETTER' ? 'mode-letter' : 'mode-word');

                    const detectionEl = document.getElementById('currentDetection');
                    if (data.detection) {
                        detectionEl.textContent = data.detection + ' (' + Math.round(data.confidence * 100) + '%)';
                    } else {
                        detectionEl.textContent = '-';
                    }
                })
                .catch(err => console.error('Update error:', err));
        }

        function clearSentence() {
            fetch('/clear_sentence', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) { updateState(); }
                });
        }

        window.addEventListener('beforeunload', function() {
            fetch('/cleanup', { method: 'POST', keepalive: true });
        });

        updateInterval = setInterval(updateState, 500);
        updateState();

        // ===== LEARN MODAL =====
        const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
        const WORDS = ['cat', 'hello', 'father', 'clean', 'later', 'beautiful', 'eat', 'book'];

        function buildAlphabetGrid() {
            const grid = document.getElementById('alphabetGrid');
            if (grid.children.length > 0) return; // already built
            LETTERS.forEach(letter => {
                const card = document.createElement('div');
                card.className = 'letter-card';
                card.onclick = () => openLightbox(letter);

                const imgSrc = `/static/asl/letters/${letter}.jpg`;
                card.innerHTML = `
                    <div class="letter-img-wrap">
                        <img src="${imgSrc}" alt="ASL ${letter}"
                             onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                        <div class="letter-placeholder" style="display:none;width:100%;height:100%;align-items:center;justify-content:center;">${letter}</div>
                    </div>
                    <div class="letter-label">${letter}</div>
                `;
                grid.appendChild(card);
            });
        }

        function buildWordGrid() {
            const grid = document.getElementById('wordGrid');
            if (grid.children.length > 0) return;
            WORDS.forEach(word => {
                const card = document.createElement('div');
                card.className = 'word-card';
                card.onclick = () => openVideoModal(word);

                const videoSrc = `/static/asl/words/${word}.mp4`;
                card.innerHTML = `
                    <div class="video-thumb">
                        <video src="${videoSrc}" muted preload="metadata"
                               style="pointer-events:none;"
                               onerror="this.style.display='none'"></video>
                        <div class="play-icon">▶</div>
                    </div>
                    <div class="word-label">${word}</div>
                `;
                grid.appendChild(card);
            });
        }

        function openLearnModal() {
            document.getElementById('learnModal').classList.add('active');
            buildAlphabetGrid();
            buildWordGrid();
            document.body.style.overflow = 'hidden';
        }

        function closeLearnModal() {
            document.getElementById('learnModal').classList.remove('active');
            document.body.style.overflow = '';
            closeVideoModal(); // stop any playing video
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + tab).classList.add('active');
            event.target.classList.add('active');
        }

        // Letter lightbox
        function openLightbox(letter) {
            document.getElementById('lightboxImg').src = `/static/asl/letters/${letter}.jpg`;
            document.getElementById('lightboxLabel').textContent = letter;
            document.getElementById('letterLightbox').classList.add('active');
        }

        function closeLightbox() {
            document.getElementById('letterLightbox').classList.remove('active');
        }

        // Word video modal
        function openVideoModal(word) {
            const video = document.getElementById('modalVideo');
            video.src = `/static/asl/words/${word}.mp4`;
            document.getElementById('videoModalLabel').textContent = word;
            document.getElementById('videoModal').classList.add('active');
            video.play();
        }

        function closeVideoModal() {
            const video = document.getElementById('modalVideo');
            video.pause();
            video.src = '';
            document.getElementById('videoModal').classList.remove('active');
        }

        // Close modals on Escape
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') {
                closeLightbox();
                closeVideoModal();
                closeLearnModal();
            }
        });
    </script>
</body>
</html>
    '''
    return html_content

def generate_frames():
    """Generate video frames with proper error handling"""
    global translator
    
    # Initialize translator if not already done
    if not initialize_translator():
        # If initialization failed, send error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Model Loading Error", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(error_frame, "Check console for details", (50, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    with camera_lock:
        if translator.camera is None:
            translator.camera = cv2.VideoCapture(0)
            translator.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            translator.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            translator.is_running = True
    
    try:
        while translator.is_running:
            with camera_lock:
                if translator.camera is None:
                    break
                success, frame = translator.camera.read()
            
            if not success:
                break
            
            frame = translator.process_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        # Cleanup when stream ends
        with camera_lock:
            if translator.camera is not None:
                translator.camera.release()
                translator.camera = None
        translator.is_running = False
        print("✓ Video stream ended, camera released")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    global translator
    if translator:
        return jsonify({
            'mode': translator.mode,
            'detection': translator.current_detection,
            'confidence': float(translator.current_confidence),
            'sentence': translator.sentence,
            'fps': translator.fps
        })
    return jsonify({'error': 'Translator not initialized'})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global translator
    if translator:
        translator.sentence.clear()
        translator.current_detection = ""
        translator.current_confidence = 0.0
        translator.sequence_buffer.clear()
        translator.word_confidence_history.clear()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Manual cleanup endpoint"""
    global translator
    if translator:
        translator.cleanup()
    return jsonify({'success': True})

if __name__ == '__main__':
    print("="*70)
    print("SIGN LANGUAGE TRANSLATOR - Starting...")
    print("="*70)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"\nChecking model files:")
    print(f"  Word Model: {WORD_MODEL_PATH}")
    print(f"    Exists: {os.path.exists(WORD_MODEL_PATH)}")
    if os.path.exists(WORD_MODEL_PATH):
        size = os.path.getsize(WORD_MODEL_PATH) / 1024
        print(f"    Size: {size:.1f} KB")
    
    print(f"  Word Labels: {WORD_LABEL_PATH}")
    print(f"    Exists: {os.path.exists(WORD_LABEL_PATH)}")
    if os.path.exists(WORD_LABEL_PATH):
        size = os.path.getsize(WORD_LABEL_PATH) / 1024
        print(f"    Size: {size:.1f} KB")
    
    print(f"  Letter Model: {LETTER_MODEL_PATH}")
    print(f"    Exists: {os.path.exists(LETTER_MODEL_PATH)}")
    if os.path.exists(LETTER_MODEL_PATH):
        size = os.path.getsize(LETTER_MODEL_PATH) / 1024
        print(f"    Size: {size:.1f} KB")
    
    print(f"  Letter Labels: {LETTER_LABEL_PATH}")
    print(f"    Exists: {os.path.exists(LETTER_LABEL_PATH)}")
    if os.path.exists(LETTER_LABEL_PATH):
        size = os.path.getsize(LETTER_LABEL_PATH) / 1024
        print(f"    Size: {size:.1f} KB")
    
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)