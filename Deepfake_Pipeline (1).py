import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import metrics
import glob
import shutil

class DeepfakeDetector:
    def __init__(self):
        pass
    
    def extract_frames(self, video_path, output_dir, skip_frames=5):
        """Extract frames from video (skip for speed)."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Error opening video file: {video_path}")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:  # Skip frames for speed
                cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:06d}.jpg"), frame)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {frame_count // skip_frames} frames")
        return frame_count // skip_frames
    
    def mse(self, imageA, imageB):
        """Mean Squared Error."""
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageB.shape[1])
        return err
    
    def analyze_video(self, video_path, threshold_mse=500, threshold_ssim=0.85):
        """Full deepfake analysis."""
        # 1. Extract frames
        frame_dir = "temp_frames"
        self.extract_frames(video_path, frame_dir)
        
        # 2. Get frame files
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        
        if len(frame_files) < 2:
            shutil.rmtree(frame_dir)
            return {
                'is_deepfake': False,
                'avg_mse': 0.0,
                'avg_ssim': 0.0,
                'confidence': 0.0,
                'error': 'Insufficient frames extracted'
            }
        
        # 3. Compare consecutive frames
        mse_scores = []
        ssim_scores = []
        print("Computing frame similarities...")
        for i in range(len(frame_files) - 1):
            img1 = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(frame_files[i+1], cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                print(f"Warning: Could not read frames {frame_files[i]}, {frame_files[i+1]}")
                continue
            
            m = self.mse(img1, img2)
            s = metrics.structural_similarity(img1, img2, data_range=255.0)
            
            mse_scores.append(m)
            ssim_scores.append(s)
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(frame_files)-1} pairs...")
        print(f"Computed {len(mse_scores)} valid pairs.")
        
        # 4. Detect deepfake
        if not mse_scores or not ssim_scores:
            avg_mse = 0.0
            avg_ssim = 0.0
        else:
            avg_mse = np.mean(mse_scores)
            avg_ssim = np.mean(ssim_scores)
        
        is_deepfake = avg_mse < threshold_mse or avg_ssim > threshold_ssim
        
        # 5. Plot results
        self.plot_analysis(mse_scores, ssim_scores, avg_mse, avg_ssim)
        print(f"\n=== DEEPFAKE VERDICT ===")
        if is_deepfake:
            print("🚨 VIDEO IS DEEPFAKE (low variation detected)")
        else:
            print("✅ VIDEO IS REAL (natural variation detected)")
        print(f"Avg MSE: {avg_mse:.2f} (lower = more consistent)")
        print(f"Avg SSIM: {avg_ssim:.3f} (higher = more similar)")
        print(f"Deepfake confidence: {max(0, 1 - avg_ssim):.2%}")
        print("======================\n")
        
        # Cleanup
        shutil.rmtree(frame_dir)
        
        return {
            'is_deepfake': is_deepfake,
            'avg_mse': avg_mse,
            'avg_ssim': avg_ssim,
            'confidence': max(0, 1 - avg_ssim)
        }

    def plot_analysis(self, mse_scores, ssim_scores, avg_mse, avg_ssim):
        """Visualize frame similarity."""
        if not mse_scores or not ssim_scores:
            print("No data available for plotting.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(mse_scores)
        ax1.axhline(avg_mse, color='r', linestyle='--', label=f'Avg: {avg_mse:.1f}')
        ax1.set_title('MSE Between Consecutive Frames')
        ax1.legend()
        
        ax2.plot(ssim_scores)
        ax2.axhline(avg_ssim, color='r', linestyle='--', label=f'Avg: {avg_ssim:.3f}')
        ax2.set_title('SSIM Between Consecutive Frames')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("deepfake_analysis.png")
        print("Plot saved to deepfake_analysis.png. Also showing interactive plot...")
        plt.show()
        plt.close()

# Usage
detector = DeepfakeDetector()

# Test single video
result = detector.analyze_video('Deepfake.mp4')
if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print(f"\nFinal Result:")
    print(f"Deepfake: {result['is_deepfake']}")
    print(f"MSE: {result['avg_mse']:.2f}, SSIM: {result['avg_ssim']:.3f}")
    print(f"Confidence: {result['confidence']:.2%}")
