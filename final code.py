"""
Hybrid Real-Time Sign Language Translation System
Supports both static letters and dynamic words
Clean floating display on hand only
MODE: Switchable between LETTER and WORD (no AUTO)
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow import keras
from collections import deque
import time
from datetime import datetime
import os

# Configuration
SEQUENCE_LENGTH = 60
NUM_FEATURES = 63

# Paths - Word Model
WORD_MODEL_PATH = r"F:\new baas\word_model.h5"
WORD_LABEL_PATH = r"F:\new baas\word_label_mapping.pkl"

# Paths - Letter Model
LETTER_MODEL_PATH = r"F:\new baas\letter_model.h5"
LETTER_LABEL_PATH = r"F:\new baas\letter_label_mapping.pkl"

OUTPUT_DIR = r"F:\new baas"

# UI Colors (BGR format)
COLOR_BG = (40, 40, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_WORD = (0, 255, 0)
COLOR_LETTER = (100, 200, 255)
COLOR_CONFIDENCE_HIGH = (0, 255, 0)
COLOR_CONFIDENCE_MED = (0, 165, 255)
COLOR_CONFIDENCE_LOW = (0, 0, 255)

class HybridTranslator:
    """Hybrid translator supporting both letters and words"""
    
    def __init__(self):
        print("\n" + "="*70)
        print(" HYBRID SIGN LANGUAGE TRANSLATION SYSTEM")
        print("="*70)
        print("\nMode: Switchable Letters (Static) / Words (Dynamic)")
        
        # Load word model
        print("\n[1/5] Loading word model...")
        try:
            self.word_model = keras.models.load_model(WORD_MODEL_PATH)
            with open(WORD_LABEL_PATH, 'rb') as f:
                self.word_mapping = pickle.load(f)
            print(f"      ✓ Words loaded: {list(self.word_mapping.values())}")
        except Exception as e:
            print(f"      ✗ Word model not found: {e}")
            self.word_model = None
            self.word_mapping = {}
        
        # Load letter model
        print("[2/5] Loading letter model...")
        try:
            self.letter_model = keras.models.load_model(LETTER_MODEL_PATH)
            with open(LETTER_LABEL_PATH, 'rb') as f:
                self.letter_mapping = pickle.load(f)
            print(f"      ✓ Letters loaded: {list(self.letter_mapping.values())}")
        except Exception as e:
            print(f"      ✗ Letter model not found: {e}")
            self.letter_model = None
            self.letter_mapping = {}
        
        if self.word_model is None and self.letter_model is None:
            raise Exception("No models loaded! Please train at least one model.")
        
        # Initialize MediaPipe
        print("[3/5] Initializing hand detection...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        print("      ✓ MediaPipe initialized")
        
        # Recognition mode: "LETTER" or "WORD"
        if self.letter_model is not None:
            self.mode = "LETTER"
        elif self.word_model is not None:
            self.mode = "WORD"
        
        # Sequence buffer for word recognition
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # Current detection
        self.current_detection = ""
        self.current_confidence = 0.0
        self.detection_type = ""  # "LETTER" or "WORD"
        
        # Sentence building
        self.sentence = []
        self.last_add_time = 0
        self.add_cooldown = 1.5
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        # Letter hold detection
        self.letter_hold_start = 0
        self.letter_hold_duration = 1.5  # seconds to confirm letter
        self.last_letter = ""
        
        print("[4/5] System configuration...")
        print(f"      Mode: {self.mode}")
        print(f"      Word model: {'✓' if self.word_model else '✗'}")
        print(f"      Letter model: {'✓' if self.letter_model else '✗'}")
        
        print("[5/5] System ready!")
        print("\n" + "="*70)
        print(" CONTROLS")
        print("="*70)
        print("  M        - Toggle mode (LETTER ↔ WORD)")
        print("  SPACE    - Add current detection to sentence")
        print("  C        - Clear sentence")
        print("  R        - Reset buffer")
        print("  S        - Save sentence to file")
        print("  Q        - Quit")
        print("\n  LETTER mode: Detects static letters (hold for 1.5s)")
        print("  WORD mode: Detects dynamic word motions")
        print("="*70 + "\n")
    
    def normalize_landmarks(self, features):
        """Normalize landmarks (same as training)"""
        features = features.reshape(21, 3)
        wrist = features[0].copy()
        features = features - wrist
        
        hand_size = np.linalg.norm(features[12] - features[0])
        if hand_size > 1e-6:
            features = features / hand_size
        
        return features.flatten()
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks"""
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
    
    def predict_letter(self, features):
        """Predict static letter"""
        if self.letter_model is None:
            return None, 0.0
        
        features_input = np.expand_dims(features, axis=0)
        predictions = self.letter_model.predict(features_input, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        letter = self.letter_mapping[predicted_idx]
        
        return letter, confidence
    
    def predict_word(self):
        """Predict dynamic word from sequence"""
        if self.word_model is None or len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0
        
        sequence = np.array(list(self.sequence_buffer))
        sequence = np.expand_dims(sequence, axis=0)
        
        predictions = self.word_model.predict(sequence, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        word = self.word_mapping[predicted_idx]
        
        return word, confidence
    
    def get_hand_center(self, hand_landmarks, frame_shape):
        """Get center point of hand for floating text"""
        h, w = frame_shape[:2]
        
        # Calculate center from all landmarks
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords)) - 80  # Offset above hand
        
        return center_x, center_y
    
    def draw_floating_text(self, frame, text, position, confidence, detection_type):
        """Draw floating text with background above hand"""
        x, y = position
        
        # Choose color based on type
        if detection_type == "LETTER":
            text_color = COLOR_LETTER
        else:
            text_color = COLOR_WORD
        
        # Confidence color
        if confidence > 0.7:
            conf_color = COLOR_CONFIDENCE_HIGH
        elif confidence > 0.5:
            conf_color = COLOR_CONFIDENCE_MED
        else:
            conf_color = COLOR_CONFIDENCE_LOW
        
        # Main text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0 if detection_type == "LETTER" else 1.5
        thickness = 4
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background rectangle
        padding = 20
        bg_x1 = x - text_w // 2 - padding
        bg_y1 = y - text_h - padding
        bg_x2 = x + text_w // 2 + padding
        bg_y2 = y + padding
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), text_color, 3)
        
        # Draw text
        text_x = x - text_w // 2
        text_y = y
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Draw confidence below
        conf_text = f"{confidence*100:.0f}%"
        conf_font_scale = 0.6
        conf_thickness = 2
        (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, conf_font_scale, conf_thickness)
        
        conf_x = x - conf_w // 2
        conf_y = bg_y2 + conf_h + 10
        cv2.putText(frame, conf_text, (conf_x, conf_y), font, conf_font_scale, conf_color, conf_thickness)
        
        # Draw type indicator
        type_text = f"[{detection_type}]"
        type_font_scale = 0.5
        (type_w, type_h), _ = cv2.getTextSize(type_text, font, type_font_scale, 2)
        type_x = x - type_w // 2
        type_y = bg_y1 - 10
        cv2.putText(frame, type_text, (type_x, type_y), font, type_font_scale, (200, 200, 200), 2)
    
    def draw_sentence_bar(self, frame):
        """Draw sentence at bottom of screen"""
        h, w = frame.shape[:2]
        
        if len(self.sentence) == 0:
            return
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-60), (w, h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Sentence text
        sentence_text = " ".join(self.sentence)
        if len(sentence_text) > 80:
            sentence_text = "..." + sentence_text[-77:]
        
        cv2.putText(frame, sentence_text, (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    
    def draw_mode_indicator(self, frame):
        """Draw current mode in corner"""
        h, w = frame.shape[:2]
        
        # Mode with color
        mode_color = COLOR_LETTER if self.mode == "LETTER" else COLOR_WORD
        mode_text = f"Mode: {self.mode}"
        cv2.putText(frame, mode_text, (w-250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w-250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        
        # Instructions hint
        cv2.putText(frame, "Press M to switch mode", (w-250, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def add_to_sentence(self):
        """Add current detection to sentence"""
        if self.current_detection and self.current_confidence >= 0.5:
            current_time = time.time()
            if current_time - self.last_add_time >= self.add_cooldown:
                self.sentence.append(self.current_detection)
                self.last_add_time = current_time
                print(f"\n[ADDED] {self.current_detection} ({self.detection_type})")
                print(f"Sentence: {' '.join(self.sentence)}")
                
                # Reset
                self.sequence_buffer.clear()
                self.current_detection = ""
                self.current_confidence = 0.0
                self.letter_hold_start = 0
                self.last_letter = ""
                return True
        return False
    
    def save_sentence(self):
        """Save sentence to file"""
        if len(self.sentence) == 0:
            print("\n[SAVE] No sentence to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"sentence_{timestamp}.txt")
        
        with open(filename, 'w') as f:
            f.write(" ".join(self.sentence))
        
        print(f"\n[SAVE] Sentence saved to: {filename}")
        print(f"       Content: {' '.join(self.sentence)}")
    
    def toggle_mode(self):
        """Toggle between LETTER and WORD modes"""
        if self.mode == "LETTER":
            if self.word_model is not None:
                self.mode = "WORD"
            else:
                print("\n[MODE] Word model not available!")
                return
        else:
            if self.letter_model is not None:
                self.mode = "LETTER"
            else:
                print("\n[MODE] Letter model not available!")
                return
        
        print(f"\n[MODE] Switched to: {self.mode}")
        self.sequence_buffer.clear()
        self.current_detection = ""
        self.current_confidence = 0.0
        self.letter_hold_start = 0
        self.last_letter = ""
    
    def run(self):
        """Main translation loop"""
        print("\n[STARTING CAMERA...]")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("[CAMERA READY] Starting translation...\n")
        print(f"Current mode: {self.mode}")
        
        cv2.namedWindow('Hybrid Sign Translator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hybrid Sign Translator', 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            self.frame_count += 1
            if time.time() - self.last_fps_update >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_fps_update = time.time()
            
            # Extract landmarks
            features, hand_landmarks = self.extract_landmarks(frame)
            
            if features is not None:
                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Detection based on current mode
                if self.mode == "LETTER":
                    # Static letter detection
                    letter, letter_conf = self.predict_letter(features)
                    
                    if letter and letter_conf > 0.5:
                        # Hold detection
                        current_time = time.time()
                        
                        if letter == self.last_letter:
                            # Same letter, check hold duration
                            hold_time = current_time - self.letter_hold_start
                            
                            if hold_time >= self.letter_hold_duration:
                                self.current_detection = letter
                                self.current_confidence = letter_conf
                                self.detection_type = "LETTER"
                        else:
                            # New letter
                            self.last_letter = letter
                            self.letter_hold_start = current_time
                            
                            # Show preview (lower confidence)
                            self.current_detection = letter
                            self.current_confidence = letter_conf * 0.7
                            self.detection_type = "LETTER"
                
                elif self.mode == "WORD":
                    # Dynamic word detection
                    self.sequence_buffer.append(features)
                    
                    if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                        word, word_conf = self.predict_word()
                        
                        if word and word_conf > 0.6:
                            self.current_detection = word
                            self.current_confidence = word_conf
                            self.detection_type = "WORD"
                
                # Draw floating text above hand
                if self.current_detection:
                    hand_center = self.get_hand_center(hand_landmarks, frame.shape)
                    self.draw_floating_text(
                        frame, 
                        self.current_detection, 
                        hand_center,
                        self.current_confidence,
                        self.detection_type
                    )
            else:
                # No hand detected, reset
                self.letter_hold_start = 0
                self.last_letter = ""
            
            # Draw sentence bar
            self.draw_sentence_bar(frame)
            
            # Draw mode indicator
            self.draw_mode_indicator(frame)
            
            # Display
            cv2.imshow('Hybrid Sign Translator', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[QUIT] Shutting down...")
                break
            elif key == ord(' '):
                self.add_to_sentence()
            elif key == ord('c'):
                self.sentence.clear()
                print("\n[CLEAR] Sentence cleared")
            elif key == ord('r'):
                self.sequence_buffer.clear()
                self.current_detection = ""
                self.current_confidence = 0.0
                self.letter_hold_start = 0
                self.last_letter = ""
                print("\n[RESET] Buffer cleared")
            elif key == ord('s'):
                self.save_sentence()
            elif key == ord('m'):
                self.toggle_mode()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n" + "="*70)
        print(" FINAL SENTENCE")
        print("="*70)
        if len(self.sentence) > 0:
            print(" ".join(self.sentence))
        else:
            print("(empty)")
        print("="*70 + "\n")

if __name__ == "__main__":
    try:
        translator = HybridTranslator()
        translator.run()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Program stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()