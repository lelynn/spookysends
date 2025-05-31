import cv2
from glob import glob
import os
import numpy as np

import cv2
from glob import glob
import os
import numpy as np

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
    
    # print("📁 fail_dirs[0]:", fail_dirs[0])
    # print("📁 fail_dirs[1]:", fail_dirs[1])

    # print("🔍 glob result for fail_dirs[0]:", glob(os.path.join(fail_dirs[0], "frame_*.png")))
    # print("🔍 glob result for fail_dirs[1]:", glob(os.path.join(fail_dirs[1], "frame_*.png")))

    # Load paths
    send_frame_paths = sorted(glob(os.path.join(send_frames_dir, "frame_*.jpg")))
    fail_masks = {
        ghost_point1: sorted(glob(os.path.join(fail_dirs[0], "frame_*.jpg"))),
        ghost_point2: sorted(glob(os.path.join(fail_dirs[1], "frame_*.jpg")))
    }

    # Define when ghosts appear (in frame indices)
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
            print(f"👀len (fail_mask_frames): ", len(fail_mask_frames))
            print(f"fail_t: ", fail_t)

            # Use static background (you can change to send_frame_paths[i] if you prefer)
            # base_frame = cv2.imread(background_path)
            base_frame = cv2.imread(send_frame_paths[i])

            for ghost_path in fail_mask_frames:
                # print(f"👀 Frame {i} is now in the pause_frames loop")

                ghost_frame = cv2.imread(ghost_path)

                # Match alpha mask from same folder's /alpha_masks/
                frame_name = os.path.basename(ghost_path).replace(".jpg", "_alpha.png")
                alpha_path = os.path.join(alpha_dirs[pause_index], "alpha_masks", frame_name)
                alpha_mask = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
                if not os.path.exists(alpha_path):
                    print("❗️Alpha mask path does not exist:", alpha_path)

                if alpha_mask is None:
                    print(f"❌ Missing alpha mask for {ghost_path}")
                    continue

                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = np.clip(alpha_mask**0.5 * alpha, 0, 1)

                alpha_3ch = cv2.merge([alpha_mask] * 3)
                ghost_f = ghost_frame.astype(np.float32)
                base_f = base_frame.astype(np.float32)
                # base_f *= dimming_value  # Optional dimming of background

                overlay = (alpha_3ch * ghost_f + (1 - alpha_3ch) * base_f).astype(np.uint8)
                output_frames.append(overlay)

            i += 1  # Continue to next send frame after ghost
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
