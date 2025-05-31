import cv2
from glob import glob
import os
import numpy as np

import cv2
from glob import glob
import os
import numpy as np
import modules
def run_spookysends_with_overlay(
    send_frames_dir,# e.g. "frames/send"
    fail_dirs, 
    alpha_dirs, # e.g. ["frames/fail1_alpha", "frames/fail2_alpha"]
    output_path,             # e.g. "output/spookysend.mp4"
    background_path,
    temp_root,
    ghost_appears=[7, 15],   # in seconds
    frame_rate=59.88,
    alpha=0.8,
    dimming_value = 1
):
    print("\n🛠️ Running spookySends with the following paths:")
    print("📂 send_frames_dir:", send_frames_dir)
    print("📂 fail_dirs:", fail_dirs)
    print("📂 alpha_dirs:", alpha_dirs)
    print("📼 output_path:", output_path)
    print("📁 background_path:", background_path)
    
    ghost_point1 = ghost_appears[0]
    ghost_point2 = ghost_appears[1]
    
    send_frame_paths = sorted(glob(os.path.join(send_frames_dir, "frame_*.jpg")))
    fail_masks = {
        ghost_point1: sorted(glob(os.path.join(fail_dirs[0], "frame_*.jpg"))),
        ghost_point2: sorted(glob(os.path.join(fail_dirs[1], "frame_*.jpg")))
    }

    pause_points = {
        int(max(0, ghost_point1 - 3)): fail_masks[ghost_point1],
        int(max(0, ghost_point2 - 2)): fail_masks[ghost_point2]
    }
    pause_frames = [int(t * frame_rate) for t in pause_points.keys()]

    output_frames = []
    i = 0

    while i < len(send_frame_paths):
        if i in pause_frames:
            print(f"👀 Frame {i} is now in the pause_frames loop")
            pause_index = pause_frames.index(i)
            fail_t = list(pause_points.keys())[pause_index]
            fail_mask_frames = pause_points[fail_t]

            send_frame = cv2.imread(send_frame_paths[i])
            alpha_folder = os.path.join(alpha_dirs[pause_index], "alpha_masks")
            os.makedirs(alpha_folder, exist_ok=True)
            send_alpha_path = os.path.join(alpha_folder, os.path.basename(send_frame_paths[i]).replace(".jpg", "_alpha.png"))

            if not os.path.exists(send_alpha_path):
                from ultralytics import YOLO  # for segmentation

                model = YOLO(os.path.join(temp_root, "yolov8n-seg.pt"))  # lightweight YOLOv8 segmentation model
                modules.segment_single_frame(send_frame_paths[i], send_alpha_path, model, background_path)

            send_alpha = cv2.imread(send_alpha_path, cv2.IMREAD_GRAYSCALE)
            if send_alpha is not None:
                send_alpha = send_alpha.astype(np.float32) / 255.0
                faded_climber = (send_frame.astype(np.float32) * send_alpha[..., None] * 0.3).astype(np.uint8)
                background = send_frame.copy()
                background[send_alpha > 0.1] = faded_climber[send_alpha > 0.1]
                base_frame = background
            else:
                base_frame = cv2.imread(background_path)

            for ghost_path in fail_mask_frames:
                ghost_frame = cv2.imread(ghost_path)
                frame_name = os.path.basename(ghost_path).replace(".jpg", "_alpha.png")
                alpha_path = os.path.join(alpha_dirs[pause_index], "alpha_masks", frame_name)
                alpha_mask = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)

                if alpha_mask is None:
                    print(f"❌ Missing alpha mask for {ghost_path}")
                    continue

                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = np.clip(alpha_mask**0.5 * alpha, 0, 1)

                alpha_3ch = cv2.merge([alpha_mask] * 3)
                ghost_f = ghost_frame.astype(np.float32)
                base_f = base_frame.astype(np.float32)
                base_f *= dimming_value

                overlay = (alpha_3ch * ghost_f + (1 - alpha_3ch) * base_f).astype(np.uint8)
                output_frames.append(overlay)

            i += 1
        else:
            frame = cv2.imread(send_frame_paths[i])
            output_frames.append(frame)
            i += 1

    # Save final video
    if not output_frames:
        raise ValueError("❌ No output frames generated.")

    h, w, _ = output_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1') , frame_rate, (w, h))
    for f in output_frames:
        out.write(f)
    out.release()
    print(f"✅ Video saved to: {output_path}, starting compression!")

    import subprocess
    compressed_path = output_path.replace(".mp4", "_compressed.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i", output_path,
        "-c:v", "libopenh264",
        "-b:v", "2M",  # bitrate, adjust as needed
        "-movflags", "+faststart",
        compressed_path
    ]
    subprocess.run(command, check=True)
    print(f"✅ Compressed using OpenH264: {output_path}")
