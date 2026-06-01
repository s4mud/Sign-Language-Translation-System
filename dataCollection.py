"""
FIXED Anti-Overfitting Sign Language Model Training
Addresses "predicting same word" issue with class balancing
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
SEQUENCE_LENGTH = 60
WORD_BUFFER_SIZE = 30
NUM_FEATURES = 63

# Paths
DATA_DIR = r"F:\new baas\Dataset_Augmented"
OUTPUT_DIR = r"F:\new baas"
MODEL_PATH = os.path.join(OUTPUT_DIR, "Model/word_model.h5")
LABEL_MAPPING_PATH = os.path.join(OUTPUT_DIR, "Model/word_label_mapping.pkl")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "model_config.json")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

class RealTimeDataProcessor:
    """Process videos with MORE variability to prevent overfitting"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def normalize_landmarks(self, features):
        """EXACT same normalization as web app"""
        features = features.reshape(21, 3)
        wrist = features[0].copy()
        features = features - wrist
        
        hand_size = np.linalg.norm(features[12] - features[0])
        if hand_size > 1e-6:
            features = features / hand_size
        
        return features.flatten()
    
    def extract_landmarks(self, frame):
        """Extract landmarks matching web app"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            features = []
            for lm in landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array(features)
            features = self.normalize_landmarks(features)
            return features
        return None
    
    def interpolate_to_60(self, buffer_30):
        """EXACT interpolation from web app"""
        if len(buffer_30) < WORD_BUFFER_SIZE:
            return None
        
        buffer_array = np.array(buffer_30[:WORD_BUFFER_SIZE])
        indices = np.linspace(0, len(buffer_array) - 1, SEQUENCE_LENGTH)
        
        sequence = np.array([
            np.interp(indices, np.arange(len(buffer_array)), buffer_array[:, i])
            for i in range(NUM_FEATURES)
        ]).T
        
        return sequence
    
    def process_video_with_variability(self, video_path, add_noise=True):
        """Process with MORE variability to prevent memorization"""
        cap = cv2.VideoCapture(video_path)
        
        valid_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Random frame drops (10-20%)
            if np.random.random() < np.random.uniform(0.10, 0.20):
                continue
            
            features = self.extract_landmarks(frame)
            if features is not None:
                # Add realistic noise during training
                if add_noise:
                    noise = np.random.normal(0, 0.01, features.shape)
                    features = features + noise
                
                valid_frames.append(features)
        
        cap.release()
        
        if len(valid_frames) < WORD_BUFFER_SIZE:
            return None
        
        # Random sampling (not always uniform)
        start_idx = np.random.randint(0, max(1, len(valid_frames) - WORD_BUFFER_SIZE))
        end_idx = start_idx + WORD_BUFFER_SIZE
        
        if end_idx > len(valid_frames):
            end_idx = len(valid_frames)
            start_idx = end_idx - WORD_BUFFER_SIZE
        
        buffer_30 = valid_frames[start_idx:end_idx]
        
        # Interpolate to 60
        sequence = self.interpolate_to_60(buffer_30)
        
        return sequence
    
    def close(self):
        self.hands.close()

def aggressive_augmentation(sequence, label, num_augmentations=4):
    """AGGRESSIVE augmentation to prevent overfitting"""
    augmented = [sequence]
    labels = [label]
    
    for _ in range(num_augmentations):
        aug_seq = sequence.copy()
        
        # 1. Speed variation (larger range)
        speed = np.random.uniform(0.85, 1.15)
        n = len(aug_seq)
        indices = (np.arange(n) * speed).clip(0, n-1).astype(int)
        aug_seq = aug_seq[indices]
        
        # 2. More noise
        noise = np.random.normal(0, 0.015, aug_seq.shape)
        aug_seq = aug_seq + noise
        
        # 3. Rotation (higher chance, larger angles)
        if np.random.random() < 0.6:
            angle = np.random.uniform(-8, 8)
            angle_rad = np.radians(angle)
            aug_seq_3d = aug_seq.reshape(SEQUENCE_LENGTH, 21, 3)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            for i in range(SEQUENCE_LENGTH):
                for j in range(21):
                    x, y = aug_seq_3d[i, j, 0], aug_seq_3d[i, j, 1]
                    aug_seq_3d[i, j, 0] = x * cos_a - y * sin_a
                    aug_seq_3d[i, j, 1] = x * sin_a + y * cos_a
            
            aug_seq = aug_seq_3d.reshape(SEQUENCE_LENGTH, NUM_FEATURES)
        
        # 4. Scaling (simulate different distances)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.90, 1.10)
            aug_seq = aug_seq * scale
        
        # 5. Random frame dropout
        if np.random.random() < 0.4:
            dropout_indices = np.random.choice(SEQUENCE_LENGTH, 
                                              size=int(SEQUENCE_LENGTH * 0.08), 
                                              replace=False)
            for idx in dropout_indices:
                if idx > 0:
                    aug_seq[idx] = aug_seq[idx - 1]
        
        # 6. Small translation
        if np.random.random() < 0.3:
            translation = np.random.normal(0, 0.02, (NUM_FEATURES,))
            aug_seq = aug_seq + translation
        
        augmented.append(aug_seq)
        labels.append(label)
    
    return augmented, labels

def load_dataset():
    """Load with BALANCED classes and strict splitting"""
    print("="*70)
    print(" LOADING DATA (Class-Balanced Mode)")
    print("="*70)
    print()
    
    processor = RealTimeDataProcessor()
    class_data = {}
    
    word_folders = [f for f in os.listdir(DATA_DIR) 
                    if os.path.isdir(os.path.join(DATA_DIR, f))]
    
    if not word_folders:
        print(f"ERROR: No folders in {DATA_DIR}")
        return None
    
    print(f"Found {len(word_folders)} classes\n")
    
    # First pass: count videos per class
    video_counts = {}
    for word in word_folders:
        word_dir = os.path.join(DATA_DIR, word)
        video_files = (glob.glob(os.path.join(word_dir, '*.mp4')) + 
                      glob.glob(os.path.join(word_dir, '*.MP4')) +
                      glob.glob(os.path.join(word_dir, '*.avi')))
        video_counts[word] = len(video_files)
    
    # Find min and max
    min_videos = min(video_counts.values())
    max_videos = max(video_counts.values())
    
    print("VIDEO COUNT PER CLASS:")
    for word, count in video_counts.items():
        marker = " ⚠ IMBALANCED!" if count > min_videos * 2 else ""
        print(f"  {word}: {count} videos{marker}")
    
    if max_videos > min_videos * 2:
        print("\n❌ WARNING: SEVERE CLASS IMBALANCE DETECTED!")
        print(f"   Range: {min_videos} to {max_videos} videos")
        print(f"   This WILL cause 'same word' prediction issue!")
        print(f"\n   SOLUTION: Balance your dataset:")
        print(f"   • Record more videos for: {[w for w, c in video_counts.items() if c == min_videos]}")
        print(f"   • Or remove excess videos from: {[w for w, c in video_counts.items() if c == max_videos]}")
        print(f"   • Target: {min_videos}-{min_videos+3} videos per word\n")
    
    # Process with BALANCED sampling
    target_samples = min_videos * 3  # Generate same number from each class
    
    for word in word_folders:
        word_dir = os.path.join(DATA_DIR, word)
        video_files = (glob.glob(os.path.join(word_dir, '*.mp4')) + 
                      glob.glob(os.path.join(word_dir, '*.MP4')) +
                      glob.glob(os.path.join(word_dir, '*.avi')))
        
        print(f"\n'{word}': processing {len(video_files)} videos...")
        
        sequences = []
        attempts_per_video = max(1, target_samples // len(video_files))
        
        for video_path in video_files:
            for attempt in range(attempts_per_video):
                sequence = processor.process_video_with_variability(video_path, add_noise=True)
                if sequence is not None:
                    sequences.append(sequence)
                    if len(sequences) >= target_samples:
                        break
            if len(sequences) >= target_samples:
                break
        
        # Balance: if we have excess, sample down
        if len(sequences) > target_samples:
            sequences = np.random.choice(sequences, target_samples, replace=False).tolist()
        
        print(f"  Final: {len(sequences)} sequences (target: {target_samples})")
        
        if len(sequences) > 0:
            class_data[word] = sequences
    
    processor.close()
    
    if len(class_data) == 0:
        print("ERROR: No data loaded!")
        return None
    
    # Check balance
    sequence_counts = {word: len(seqs) for word, seqs in class_data.items()}
    print("\n" + "="*70)
    print(" CLASS BALANCE CHECK")
    print("="*70)
    for word, count in sequence_counts.items():
        print(f"{word}: {count} sequences")
    
    balance_ratio = max(sequence_counts.values()) / min(sequence_counts.values())
    if balance_ratio > 1.5:
        print(f"\n⚠ WARNING: Still imbalanced (ratio: {balance_ratio:.2f})")
    else:
        print(f"\n✓ Good balance (ratio: {balance_ratio:.2f})")
    
    # STRICT splitting with stratification
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    print("\n" + "="*70)
    print(" STRICT DATA SPLITTING")
    print("="*70)
    
    for word, sequences in class_data.items():
        n = len(sequences)
        np.random.shuffle(sequences)
        
        # 50% train, 25% val, 25% test
        n_train = max(1, int(n * 0.50))
        n_val = max(1, int(n * 0.25))
        n_test = max(1, n - n_train - n_val)
        
        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train:n_train + n_val]
        test_seqs = sequences[n_train + n_val:]
        
        print(f"{word}: train={n_train}, val={n_val}, test={n_test}")
        
        # Augment training data HEAVILY
        for seq in train_seqs:
            aug_seqs, aug_labels = aggressive_augmentation(seq, word, num_augmentations=4)
            X_train.extend(aug_seqs)
            y_train.extend(aug_labels)
        
        # NO augmentation for val/test
        X_val.extend(val_seqs)
        y_val.extend([word] * len(val_seqs))
        
        X_test.extend(test_seqs)
        y_test.extend([word] * len(test_seqs))
    
    print()
    print(f"Final dataset:")
    print(f"  Train: {len(X_train)} (heavily augmented)")
    print(f"  Val: {len(X_val)} (no augmentation)")
    print(f"  Test: {len(X_test)} (no augmentation)")
    
    return np.array(X_train), np.array(X_val), np.array(X_test), \
           np.array(y_train), np.array(y_val), np.array(y_test)

def create_regularized_model(num_classes):
    """Model with STRONGER regularization to prevent same-word issue"""
    
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        
        # Lighter architecture with MORE dropout
        layers.Conv1D(24, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.6),  # Increased from 0.4
        
        layers.Conv1D(48, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.6),  # Increased from 0.45
        
        # Smaller LSTM with HIGHER dropout
        layers.LSTM(32, return_sequences=False,  # Reduced from 48
                   dropout=0.6, recurrent_dropout=0.5),  # Increased
        layers.BatchNormalization(),
        
        # Smaller dense layers with MORE regularization
        layers.Dense(32, activation='relu',  # Reduced from 48
                    kernel_regularizer=regularizers.l2(0.02)),  # Increased
        layers.Dropout(0.6),  # Increased
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Lower learning rate for better convergence
    optimizer = keras.optimizers.Adam(learning_rate=0.0002)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix as heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix\n(Rows=Actual, Columns=Predicted)', fontsize=14, pad=20)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    
    # Show plot
    plt.show()

def train_model():
    """Train with class balancing"""
    print("\n" + "="*70)
    print(" ANTI-OVERFITTING TRAINING (CLASS-BALANCED)")
    print("="*70)
    print("\nTarget: 70-85% test accuracy with NO prediction bias\n")
    
    data = load_dataset()
    if data is None:
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_val, y_test]))
    
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Compute class weights to handle any remaining imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    print("\n" + "="*70)
    print(" CLASS WEIGHTS (to handle imbalance)")
    print("="*70)
    for i, w in class_weight_dict.items():
        print(f"{label_mapping[i]}: {w:.3f}")
    
    print("\n" + "="*70)
    print(" DATASET")
    print("="*70)
    print(f"Words: {', '.join(label_mapping.values())}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model
    print("\n" + "="*70)
    print(" MODEL (Stronger Regularization)")
    print("="*70)
    
    model = create_regularized_model(len(label_mapping))
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n" + "="*70)
    print(" TRAINING")
    print("="*70)
    
    history = model.fit(
        X_train, y_train_enc,
        validation_data=(X_val, y_val_enc),
        epochs=100,
        batch_size=8,
        class_weight=class_weight_dict,  # Use class weights!
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print(" EVALUATION")
    print("="*70)
    
    train_loss, train_acc = model.evaluate(X_train, y_train_enc, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_enc, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_enc, verbose=0)
    
    print(f"Train Accuracy: {train_acc*100:.1f}%")
    print(f"Val Accuracy: {val_acc*100:.1f}%")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    
    gen_gap = train_acc - test_acc
    
    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*70)
    print(" PER-CLASS PERFORMANCE")
    print("="*70)
    report = classification_report(y_test_enc, y_pred_classes, 
                                   target_names=label_encoder.classes_,
                                   zero_division=0)
    print(report)
    
    print("\n" + "="*70)
    print(" CONFUSION MATRIX")
    print("="*70)
    cm = confusion_matrix(y_test_enc, y_pred_classes)
    print("Rows=Actual, Cols=Predicted")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(cm)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, label_encoder.classes_, CONFUSION_MATRIX_PATH)
    
    # Check for prediction bias
    pred_distribution = np.bincount(y_pred_classes)
    print("\n" + "="*70)
    print(" PREDICTION DISTRIBUTION (Test Set)")
    print("="*70)
    for i, count in enumerate(pred_distribution):
        pct = (count / len(y_pred_classes)) * 100
        print(f"{label_mapping[i]}: {count}/{len(y_pred_classes)} ({pct:.1f}%)")
    
    max_pred_pct = max(pred_distribution) / len(y_pred_classes)
    if max_pred_pct > 0.6:
        print(f"\n❌ WARNING: One class dominates predictions ({max_pred_pct*100:.0f}%)")
        print("   Model still has bias - need more balanced training data!")
    else:
        print(f"\n✓ Good prediction distribution (max: {max_pred_pct*100:.0f}%)")
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    
    with open(LABEL_MAPPING_PATH, 'wb') as f:
        pickle.dump(label_mapping, f)
    
    config = {
        'num_classes': len(label_mapping),
        'classes': list(label_mapping.values()),
        'test_accuracy': float(test_acc),
        'train_accuracy': float(train_acc),
        'generalization_gap': float(gen_gap),
        'max_prediction_bias': float(max_pred_pct)
    }
    
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*70)
    print(" FINAL ASSESSMENT")
    print("="*70)
    
    if max_pred_pct > 0.6:
        print("❌ FAILED: Model predicts one word too often")
        print("   ACTION: Balance your dataset and retrain")
    elif test_acc < 0.60:
        print("⚠ WARNING: Low accuracy")
        print("   ACTION: Need more training data or better quality videos")
    elif gen_gap > 0.25:
        print("⚠ WARNING: High overfitting")
        print("   ACTION: Record more diverse videos")
    else:
        print("✓✓ SUCCESS: Model ready for deployment!")
        print(f"   Test accuracy: {test_acc*100:.1f}%")
        print(f"   Prediction balance: {(1-max_pred_pct)*100:.0f}% diversity")
    
    print("\n" + "="*70)
    print(" FILES SAVED")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Labels: {LABEL_MAPPING_PATH}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Confusion Matrix: {CONFUSION_MATRIX_PATH}")
    print("="*70 + "\n")
    
    return model, label_mapping, history

if __name__ == "__main__":
    try:
        np.random.seed(42)
        train_model()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()