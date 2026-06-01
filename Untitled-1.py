
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow import keras
from collections import deque
import time
from datetime import datetime

# Configuration
SEQUENCE_LENGTH = 60
NUM_FEATURES = 63
MODEL_PATH = r"F:\new baas\word_model.h5"
LABEL_MAPPING_PATH = r"F:\new baas\word_label_mapping.pkl"

# UI Colors (BGR format)
COLOR_BG = (40, 40, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_WORD = (0, 255, 0)
COLOR_SENTENCE = (255, 200, 100)
COLOR_RECORDING = (0, 100, 255)
COLOR_READY = (0, 255, 0)
COLOR_CONFIDENCE_HIGH = (0, 255, 0)
COLOR_CONFIDENCE_MED = (0, 165, 255)
COLOR_CONFIDENCE_LOW = (0, 0, 255)

class RealTimeTranslator:
    """Real-time sign language translation system"""
    
    def __init__(self):
        print("\n" + "="*70)
        print(" REAL-TIME SIGN LANGUAGE TRANSLATION SYSTEM")
        print("="*70)
        
        # Load model
        print("\n[1/4] Loading model...")
        try:
            self.model = keras.models.load_model(MODEL_PATH)
            print(f"      ✓ Model loaded from: {MODEL_PATH}")
        except Exception as e:
            print(f"      ✗ Error loading model: {e}")
            raise
        
        # Load label mapping
        print("[2/4] Loading labels...")
        try:
            with open(LABEL_MAPPING_PATH, 'rb') as f:
                self.label_mapping = pickle.load(f)
            print(f"      ✓ Labels loaded: {list(self.label_mapping.values())}")
        except Exception as e:
            print(f"      ✗ Error loading labels: {e}")
            raise
        
        # Initialize MediaPipe
        print("[3/4] Initializing hand detection...")
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
        
        # Sequence buffer
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
        # Translation state
        self.current_word = ""
        self.current_confidence = 0.0
        self.all_predictions = []
        self.sentence = []
        self.last_add_time = 0
        self.add_cooldown = 2.0  # seconds between auto-adds
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        print("[4/4] System ready!")
        print("\n" + "="*70)
        print(" CONTROLS")
        print("="*70)
        print("  SPACE    - Add current word to sentence")
        print("  C        - Clear sentence")
        print("  R        - Reset sequence buffer")
        print("  S        - Save sentence to file")
        print("  Q        - Quit")
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
    
    def predict_word(self):
        """Predict word from buffer"""
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            return None, 0.0, []
        
        sequence = np.array(list(self.sequence_buffer))
        sequence = np.expand_dims(sequence, axis=0)
        
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            (self.label_mapping[idx], predictions[idx])
            for idx in top_indices
        ]
        
        predicted_word = top_predictions[0][0]
        confidence = top_predictions[0][1]
        
        return predicted_word, confidence, top_predictions
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks with style"""
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
    
    def draw_overlay(self, frame, x, y, w, h, alpha=0.7):
        """Draw semi-transparent overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), COLOR_BG, -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    
    def draw_progress_bar(self, frame, x, y, width, height, progress, color):
        """Draw progress bar"""
        # Background
        cv2.rectangle(frame, (x, y), (x+width, y+height), (100, 100, 100), -1)
        # Progress
        fill_width = int(width * progress)
        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x+fill_width, y+height), color, -1)
        # Border
        cv2.rectangle(frame, (x, y), (x+width, y+height), (200, 200, 200), 2)
    
    def draw_ui(self, frame, hand_landmarks):
        """Draw complete UI"""
        h, w = frame.shape[:2]
        
        # Draw hand landmarks
        self.draw_hand_landmarks(frame, hand_landmarks)
        
        # Top panel - Status and current word
        self.draw_overlay(frame, 0, 0, w, 180, 0.75)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w-150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        
        # Recording status (always show RECORDING)
        buffer_progress = len(self.sequence_buffer) / SEQUENCE_LENGTH
        status_text = "RECORDING"
        status_color = COLOR_READY
        cv2.putText(frame, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        # Buffer progress bar
        self.draw_progress_bar(frame, 20, 60, 300, 20, buffer_progress, status_color)
        cv2.putText(frame, f"{len(self.sequence_buffer)}/{SEQUENCE_LENGTH}", (330, 77),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        
        # Current word prediction
        if self.current_word and self.current_confidence > 0:
            cv2.putText(frame, "Detected Word:", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            cv2.putText(frame, self.current_word, (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WORD, 3)
            
            # Confidence indicator
            conf_color = COLOR_CONFIDENCE_HIGH if self.current_confidence > 0.7 else \
                        COLOR_CONFIDENCE_MED if self.current_confidence > 0.5 else \
                        COLOR_CONFIDENCE_LOW
            conf_text = f"{self.current_confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (w-200, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 3)
            
            # Confidence bar
            self.draw_progress_bar(frame, w-200, 120, 180, 15, 
                                 self.current_confidence, conf_color)
        
        # Middle - Top 3 predictions
        if len(self.all_predictions) > 0:
            y_offset = 220
            cv2.putText(frame, "Top Predictions:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
            
            for i, (word, conf) in enumerate(self.all_predictions[:3]):
                y_pos = y_offset + 30 + (i * 35)
                conf_color = COLOR_CONFIDENCE_HIGH if conf > 0.7 else \
                           COLOR_CONFIDENCE_MED if conf > 0.5 else \
                           COLOR_CONFIDENCE_LOW
                
                text = f"{i+1}. {word}"
                cv2.putText(frame, text, (30, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
                
                # Confidence bar for each
                bar_x = 250
                bar_width = 150
                self.draw_progress_bar(frame, bar_x, y_pos-15, bar_width, 20, conf, conf_color)
                cv2.putText(frame, f"{conf*100:.0f}%", (bar_x + bar_width + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
        
        # Bottom panel - Sentence
        self.draw_overlay(frame, 0, h-120, w, 120, 0.75)
        
        cv2.putText(frame, "Sentence:", (20, h-90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        if len(self.sentence) > 0:
            sentence_text = " ".join(self.sentence)
            # Wrap text if too long
            max_chars = 70
            if len(sentence_text) > max_chars:
                lines = [sentence_text[i:i+max_chars] for i in range(0, len(sentence_text), max_chars)]
                for i, line in enumerate(lines[-2:]):  # Show last 2 lines
                    cv2.putText(frame, line, (20, h-55 + i*30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SENTENCE, 2)
            else:
                cv2.putText(frame, sentence_text, (20, h-55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SENTENCE, 2)
        else:
            cv2.putText(frame, "(empty)", (20, h-55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        # Instructions
        instructions = "SPACE:Add  C:Clear  R:Reset  S:Save  Q:Quit"
        cv2.putText(frame, instructions, (20, h-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    
    def add_word_to_sentence(self):
        """Add current word to sentence"""
        if self.current_word and self.current_confidence >= 0.5:
            current_time = time.time()
            if current_time - self.last_add_time >= self.add_cooldown:
                self.sentence.append(self.current_word)
                self.last_add_time = current_time
                print(f"\n[ADDED] {self.current_word} (confidence: {self.current_confidence*100:.1f}%)")
                print(f"Sentence: {' '.join(self.sentence)}")
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
    
    def run(self):
        """Main translation loop"""
        print("\n[STARTING CAMERA...]")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("[CAMERA READY] Starting translation...\n")
        
        # Create resizable window
        cv2.namedWindow('Sign Language Translator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sign Language Translator', 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Flip frame for mirror effect
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
                self.sequence_buffer.append(features)
                
                # Predict when buffer is full
                if len(self.sequence_buffer) == SEQUENCE_LENGTH:
                    word, confidence, all_preds = self.predict_word()
                    
                    if word:
                        self.current_word = word
                        self.current_confidence = confidence
                        self.all_predictions = all_preds
            
            # Draw UI
            self.draw_ui(frame, hand_landmarks)
            
            # Display
            cv2.imshow('Sign Language Translator', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[QUIT] Shutting down...")
                break
            elif key == ord(' '):
                if self.add_word_to_sentence():
                    self.sequence_buffer.clear()
                    self.current_word = ""
                    self.current_confidence = 0.0
                    self.all_predictions = []
            elif key == ord('c'):
                self.sentence.clear()
                print("\n[CLEAR] Sentence cleared")
            elif key == ord('r'):
                self.sequence_buffer.clear()
                self.current_word = ""
                self.current_confidence = 0.0
                self.all_predictions = []
                print("\n[RESET] Buffer cleared")
            elif key == ord('s'):
                self.save_sentence()
        
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
    import os
    OUTPUT_DIR = r"F:\new baas"
    
    try:
        translator = RealTimeTranslator()
        translator.run()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Program stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()