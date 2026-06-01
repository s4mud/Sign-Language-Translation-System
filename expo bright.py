
import cv2
import numpy as np
import os
import glob
from pathlib import Path

# Configuration
INPUT_DIR = r"F:\new baas\Dataset"
OUTPUT_DIR = r"F:\new baas\Dataset_Augmented"

# 5 exposure levels from -1 (darker) to +1 (brighter)
EXPOSURE_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]

def adjust_exposure(frame, exposure_value):
    """
    Adjust frame exposure/brightness
    exposure_value: -1.0 (very dark) to +1.0 (very bright)
    """
    # Convert exposure value to multiplier
    # -1.0 -> 0.5x darker, 0.0 -> 1.0x normal, +1.0 -> 1.5x brighter
    multiplier = 1.0 + (exposure_value * 0.5)
    
    # Apply exposure adjustment
    adjusted = frame.astype(np.float32) * multiplier
    
    # Clip values to valid range [0, 255]
    adjusted = np.clip(adjusted, 0, 255)
    
    return adjusted.astype(np.uint8)

def process_video(input_path, output_path, exposure_value):
    """Process a single video with given exposure level"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"  ❌ Failed to open: {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply exposure adjustment
        adjusted_frame = adjust_exposure(frame, exposure_value)
        
        # Write frame
        out.write(adjusted_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return frame_count

def augment_dataset():
    """Augment entire dataset with different exposure levels"""
    print("="*70)
    print(" VIDEO EXPOSURE AUGMENTATION")
    print("="*70)
    print(f"\nInput: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Exposure levels: {EXPOSURE_LEVELS}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all word folders
    word_folders = [f for f in os.listdir(INPUT_DIR) 
                    if os.path.isdir(os.path.join(INPUT_DIR, f))]
    
    if not word_folders:
        print(f"❌ ERROR: No folders found in {INPUT_DIR}")
        return
    
    print(f"Found {len(word_folders)} classes: {', '.join(word_folders)}\n")
    
    total_original = 0
    total_generated = 0
    
    # Process each word folder
    for word in word_folders:
        input_word_dir = os.path.join(INPUT_DIR, word)
        output_word_dir = os.path.join(OUTPUT_DIR, word)
        
        # Create output folder for this word
        os.makedirs(output_word_dir, exist_ok=True)
        
        # Get all video files
        video_files = (glob.glob(os.path.join(input_word_dir, '*.mp4')) + 
                      glob.glob(os.path.join(input_word_dir, '*.MP4')) +
                      glob.glob(os.path.join(input_word_dir, '*.avi')))
        
        if not video_files:
            print(f"⚠ '{word}': No videos found, skipping...")
            continue
        
        print(f"📁 '{word}': {len(video_files)} videos")
        total_original += len(video_files)
        
        # Process each video
        for video_idx, video_path in enumerate(video_files, 1):
            video_name = Path(video_path).stem
            
            print(f"  Processing {video_idx}/{len(video_files)}: {video_name}")
            
            # Generate 5 versions with different exposures
            for exp_idx, exposure in enumerate(EXPOSURE_LEVELS):
                # Create output filename
                exposure_label = f"exp{exposure:+.1f}".replace('.', '_').replace('+', 'p').replace('-', 'n')
                output_filename = f"{video_name}_{exposure_label}.mp4"
                output_path = os.path.join(output_word_dir, output_filename)
                
                # Process video
                frames = process_video(video_path, output_path, exposure)
                
                if frames:
                    exp_desc = "darker" if exposure < 0 else "brighter" if exposure > 0 else "normal"
                    print(f"    ✓ {exposure:+.1f} ({exp_desc}): {frames} frames -> {output_filename}")
                    total_generated += 1
                else:
                    print(f"    ❌ Failed at exposure {exposure:+.1f}")
        
        print()
    
    # Summary
    print("="*70)
    print(" AUGMENTATION COMPLETE")
    print("="*70)
    print(f"Original videos: {total_original}")
    print(f"Generated videos: {total_generated}")
    print(f"Multiplication factor: {total_generated / total_original if total_original > 0 else 0:.1f}x")
    print(f"\nAugmented dataset saved to: {OUTPUT_DIR}")
    print("="*70)
    print("\n💡 NEXT STEPS:")
    print("1. Review some generated videos to ensure quality")
    print("2. Update DATA_DIR in your training script to:")
    print(f"   DATA_DIR = r\"{OUTPUT_DIR}\"")
    print("3. Re-run training with the augmented dataset")
    print("="*70)

def preview_exposure_effects(sample_video_path):
    """
    Preview exposure effects on a single video (optional utility)
    Creates a side-by-side comparison image
    """
    print("\n" + "="*70)
    print(" PREVIEW MODE - Showing exposure effects")
    print("="*70)
    
    cap = cv2.VideoCapture(sample_video_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open: {sample_video_path}")
        return
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Failed to read frame")
        return
    
    # Create comparison image
    h, w = frame.shape[:2]
    comparison = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    
    # Generate 5 exposure variations (will arrange in grid)
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for idx, exposure in enumerate(EXPOSURE_LEVELS):
        row, col = positions[idx]
        adjusted = adjust_exposure(frame, exposure)
        
        # Add text label
        label = f"Exp: {exposure:+.1f}"
        cv2.putText(adjusted, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        comparison[row*h:(row+1)*h, col*w:(col+1)*w] = adjusted
    
    # Save comparison image
    output_path = "exposure_comparison.jpg"
    cv2.imwrite(output_path, comparison)
    
    print(f"\n✓ Comparison image saved: {output_path}")
    print("  Review this to see the exposure effects before full augmentation")
    print("="*70)

if __name__ == "__main__":
    try:
        # Main augmentation
        augment_dataset()
        
        # Optional: Preview effects on first video found
        # Uncomment below to see exposure comparison before processing all videos
        """
        sample_videos = glob.glob(os.path.join(INPUT_DIR, "*", "*.mp4"))
        if sample_videos:
            preview_exposure_effects(sample_videos[0])
        """
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()