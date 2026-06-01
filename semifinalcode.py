from flask import Flask, Response, jsonify, request
import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
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

# Paths
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
        # Load models
        try:
            self.word_model = keras.models.load_model(WORD_MODEL_PATH)
            with open(WORD_LABEL_PATH, 'rb') as f:
                self.word_mapping = pickle.load(f)
        except:
            self.word_model = None
            self.word_mapping = {}
        
        try:
            self.letter_model = keras.models.load_model(LETTER_MODEL_PATH)
            with open(LETTER_LABEL_PATH, 'rb') as f:
                self.letter_mapping = pickle.load(f)
        except:
            self.letter_model = None
            self.letter_mapping = {}
        
        if self.word_model is None and self.letter_model is None:
            raise Exception("No models loaded!")
        
        # Initialize MediaPipe - EXACT same settings as training
        self.mp_hands = mp_hands  # keep reference to module for HAND_CONNECTIONS
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        # Mode - Always start with WORD if available so word model runs immediately
        self.mode = "WORD" if self.word_model else "LETTER"
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
        
        # Show movement indicator
        if len(self.movement_history) > 0:
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
    global translator
    if translator is None:
        try:
            translator = WebHybridTranslator()
            print("✓ Translator initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing translator: {e}")
            import traceback
            traceback.print_exc()

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

        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img src="/video_feed" alt="Video Stream" id="videoFeed">
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

                <button class="btn" onclick="clearSentence()">Clear Detection</button>
            </div>
        </div>
    </div>

    <script>
        let updateInterval;

        function updateState() {
            fetch('/get_state')
                .then(response => response.json())
                .then(data => {
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
                    if (data.success) {
                        updateState();
                    }
                });
        }

        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            fetch('/cleanup', { method: 'POST', keepalive: true });
        });

        // Start updates
        updateInterval = setInterval(updateState, 500);
        updateState();
    </script>
</body>
</html>
    '''
    return html_content

def generate_frames():
    global translator
    initialize_translator()
    
    # If translator failed to initialize (e.g., Mediapipe error), stop cleanly
    if translator is None:
        print("✗ Translator is None after initialization. Check model and Mediapipe installation.")
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
  
   
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)