from modules import extract_and_align_frames
from modules import extract_frames
from modules import segment_climber_enhanced
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
from ultralytics import YOLO  # for segmentation

import os 
def preprocess_all(send_video_path, #temp_root/send_video.mp4
                   fail_video_paths, #[temp_root/fail_video0.mp4, temp_root/fail_video1.mp4]
                   send_frames_dir, #temp_root/send_frames
                   fail_dirs, # [temp_root/fail_0, temp_root/fail_1/]
                   alpha_dirs, # [temp_root/fail_0_alpha,temp_root/fail_0_alpha]
                   temp_root):
    
    extract_frames(send_video_path, send_frames_dir)
    print('Send frames extracted, Starting fail 0 frame alignments...')

    extract_and_align_frames(fail_video_paths[0], fail_dirs[0], reference_frame_path=os.path.join(send_frames_dir, "frame_0240.jpg"))
    print('Send frames extracted, Starting fail 1 frame alignments...')

    extract_and_align_frames(fail_video_paths[1], fail_dirs[1], reference_frame_path=os.path.join(send_frames_dir, "frame_0240.jpg"))
    print('All alignments done...')

    # Initialize segmentation model
    model = YOLO(os.path.join(temp_root, "yolov8n-seg.pt"))  # lightweight YOLOv8 segmentation model
    print('Starting segmentation 0...')

    segment_climber_enhanced(fail_dirs[0], alpha_dirs[0], model, temp_root)
    print('Starting segmentation 1...')

    segment_climber_enhanced(fail_dirs[1], alpha_dirs[1], model, temp_root)
    print('All segmentation done...')

