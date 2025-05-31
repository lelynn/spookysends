import cv2
import os
import numpy as np
# from moviepy import VideoFileClip, concatenate_videoclips, ImageSequenceClip
from glob import glob
# import matplotlib.pyplot as plt
    

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Auto-rotate if height > width
        if frame.shape[0] < frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        frame_path = os.path.join(output_folder, f'frame_{idx:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()

def compute_homography_and_warp(fixed_frame, moving_frame):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(moving_frame, None)
    kp2, des2 = orb.detectAndCompute(fixed_frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        height, width = fixed_frame.shape[:2]
        warped = cv2.warpPerspective(moving_frame, H, (width, height))
        return warped
    else:
        return moving_frame  # fallback if not enough matches
        
def extract_and_align_frames(video_path, output_folder, reference_frame_path):
    cap = cv2.VideoCapture(video_path)
    ref = cv2.imread(reference_frame_path)
    
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Auto-rotate if height > width
        if frame.shape[0] < frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Align to the reference send frame
        aligned = compute_homography_and_warp(ref, frame)
        
        frame_path = os.path.join(output_folder, f'frame_{idx:04d}.jpg')
        cv2.imwrite(frame_path, aligned)
        idx += 1
    cap.release()


def segment_climber_enhanced(fail_dir, alpha_dir, model, temp_root, temporal_smoothing=True):
    # os.makedirs(output_folder, exist_ok=True)
    alpha_folder = os.path.join(alpha_dir, 'alpha_masks')
    os.makedirs(alpha_folder, exist_ok=True)
    
    background_path = os.path.join(temp_root, "background.jpg")
    print('background_path: ', background_path)
    background = cv2.imread(background_path)

    fail_frame_paths = sorted(glob(os.path.join(fail_dir, '*.jpg')))

    mask_history = []

    for idx, frame_path in enumerate(fail_frame_paths):
        frame = cv2.imread(frame_path)

        # Step 1: Compute motion mask
        motion_diff = cv2.absdiff(frame, background)
        motion_gray = cv2.cvtColor(motion_diff, cv2.COLOR_BGR2GRAY)
        _, motion_thresh = cv2.threshold(motion_gray, 30, 255, cv2.THRESH_BINARY)
        motion_clean = cv2.morphologyEx(motion_thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Step 2: Run YOLOv8 segmentation
        results = model.predict(source=frame, save=False, imgsz=960, conf=0.05)
        masks = results[0].masks
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        if masks is not None and len(masks.data) > 0:
            # Step 3: Filter person masks
            person_masks = [m for m, cls in zip(masks.data, classes) if cls == 0]
            if len(person_masks) == 0:
                print(f"❌ No person mask found in {frame_path}")
                continue

        # break
        # Step 4: Combine and filter with motion
        combined_mask = np.any([m.cpu().numpy() for m in person_masks], axis=0).astype(np.uint8) * 255
        motion_resized = cv2.resize(motion_clean, (combined_mask.shape[1], combined_mask.shape[0]))
        combined_mask = cv2.bitwise_and(combined_mask, motion_resized)

        # Step 5: Morphology
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        kernel_dilate = np.ones((5, 5), np.uint8)

        mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        mask_closed = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close)
        mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=1)
        smoothed_mask = cv2.GaussianBlur(mask_dilated, (21, 21), 0)

        # Step 6: Temporal smoothing
        if temporal_smoothing:
            mask_history.append(smoothed_mask)
            if len(mask_history) > 15:
                mask_history.pop(0)
            avg_mask = np.mean(mask_history, axis=0).astype(np.uint8)
            _, smoothed_mask = cv2.threshold(avg_mask, 40, 255, cv2.THRESH_BINARY)

        # Step 7: Blend mask
        final_mask = cv2.resize(smoothed_mask, (frame.shape[1], frame.shape[0]))
        alpha = final_mask.astype(np.float32) / 255.0

        fade_duration = 1
        fade_factor = min(1.0, idx / fade_duration)
        alpha *= fade_factor
        alpha_3ch = cv2.merge([alpha] * 3)

        frame_f = frame.astype(np.float32)
        background_f = background.astype(np.float32)
        blended = (alpha_3ch * frame_f + (1 - alpha_3ch) * background_f).astype(np.uint8)

        # Save paths using the basename of the current frame_path
        frame_name = os.path.basename(frame_path)
        fail_frame_path = os.path.join(fail_dir, frame_name)
        cv2.imwrite(fail_frame_path, blended)

        alpha_mask_name = os.path.splitext(frame_name)[0] + '_alpha.png'
        alpha_mask_path = os.path.join(alpha_folder, alpha_mask_name)
        cv2.imwrite(alpha_mask_path, final_mask)
        print(f"✅ Saved enhanced ghost + alpha: {os.path.basename(frame_path)}")

# def segment_single_frame(send_frame_path, alpha_output_path, model, background_path):
#     frame = cv2.imread(send_frame_path)
#     background = cv2.imread(background_path)

#     # Step 1: Compute motion mask
#     motion_diff = cv2.absdiff(frame, background)
#     motion_gray = cv2.cvtColor(motion_diff, cv2.COLOR_BGR2GRAY)
#     _, motion_thresh = cv2.threshold(motion_gray, 30, 255, cv2.THRESH_BINARY)
#     motion_clean = cv2.morphologyEx(motion_thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

#     # Step 2: Run YOLOv8 segmentation
#     results = model.predict(source=frame, save=False, imgsz=960, conf=0.05)
#     masks = results[0].masks
#     classes = results[0].boxes.cls.cpu().numpy().astype(int)

#     if masks is not None and len(masks.data) > 0:
#         person_masks = [m for m, cls in zip(masks.data, classes) if cls == 0]
#         if len(person_masks) == 0:
#             print(f"❌ No person mask found in {send_frame_path}")
#             return
#     else:
#         print(f"❌ No masks found at all in {send_frame_path}")
#         return

#     combined_mask = np.any([m.cpu().numpy() for m in person_masks], axis=0).astype(np.uint8) * 255
#     motion_resized = cv2.resize(motion_clean, (combined_mask.shape[1], combined_mask.shape[0]))
#     combined_mask = cv2.bitwise_and(combined_mask, motion_resized)

#     # Morphology
#     kernel_open = np.ones((3, 3), np.uint8)
#     kernel_close = np.ones((7, 7), np.uint8)
#     kernel_dilate = np.ones((5, 5), np.uint8)
#     mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
#     mask_closed = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close)
#     mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=1)
#     smoothed_mask = cv2.GaussianBlur(mask_dilated, (21, 21), 0)

#     final_mask = cv2.resize(smoothed_mask, (frame.shape[1], frame.shape[0]))
#     cv2.imwrite(alpha_output_path, final_mask)
#     print(f"✅ Saved alpha mask for paused frame: {alpha_output_path}")


# def overlay_ghost(background, ghost, alpha=0.5):
#     """Overlay ghost on background using alpha transparency."""
#     mask = cv2.cvtColor(ghost, cv2.COLOR_BGR2GRAY)
#     binary = mask > 10
#     mask_3ch = np.stack([binary]*3, axis=-1)

#     blended = background.copy()
#     blended[mask_3ch] = cv2.addWeighted(background, 1 - alpha, ghost, alpha, 0)[mask_3ch]
#     return blended

# import uuid

# def create_temp_workspace():
#     session_id = str(uuid.uuid4())[:8]
#     base_dir = os.path.join('static', 'temp_sessions', session_id)
#     os.makedirs(base_dir, exist_ok=True)
#     return base_dir
