import streamlit as st
import streamlit.components.v1 as components

import os
import uuid
from glob import glob
import cv2
import numpy as np
from pipeline import run_spookysends_with_overlay
from preprocess import preprocess_all

import streamlit as st
import os
import uuid
from glob import glob
import cv2
import numpy as np
from pipeline import run_spookysends_with_overlay
from preprocess import preprocess_all
import time
start_time = time.time()
import os
import cv2
import numpy as np
from glob import glob

def create_background_image(background_file, send_frames_dir):
    print('🧠 Step 1: Create background image...')

    background_path = os.path.join(temp_root, "background.jpg")

    # === OPTION A: Use uploaded background video ===
    if background_file is not None:
        print("📽️ Using uploaded background video.")
        background_video_path = os.path.join(temp_root, "background_video.mp4")
        with open(background_video_path, "wb") as f:
            f.write(background_file.read())

        cap = cv2.VideoCapture(background_video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise ValueError("❌ Failed to extract frame from uploaded background video.")
        
        cv2.imwrite(background_path, frame)
        print(f"✅ Saved one frame from uploaded background video: {background_path}")
        return background_path


    else:
        # === OPTION B: Compute mean background from send frames ===
        print("📷 Computing background from send frames.")
        send_frame_paths = sorted(glob(os.path.join(send_frames_dir, "*.jpg")))
        if not send_frame_paths:
            raise ValueError("❌ No frames found in send_frames directory.")

        mean_frame = None
        count = 0

        for p in send_frame_paths:
            img = cv2.imread(p)
            if img is not None:
                img = img.astype(np.float32)
                if mean_frame is None:
                    mean_frame = img
                else:
                    mean_frame += img
                count += 1
            else:
                print(f"⚠️ Warning: could not read {p}")

        if count == 0:
            raise ValueError("❌ No valid frames to compute background.")

        mean_frame /= count
        mean_frame = mean_frame.astype(np.uint8)
        cv2.imwrite(background_path, mean_frame)
        print(f"✅ Saved computed background: {background_path}")
        return background_path

def create_temp_workspace():
    session_id = str(uuid.uuid4())[:8]
    base_dir = os.path.join("static", "temp_sessions", session_id)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

st.set_page_config(page_title="SpookySends", layout="centered")


st.title("🧷 SpookySends: A ghost overlay video tool for beta clips")
st.write("Upload a send video and one or more fail videos to create a ghost overlay.")

# Uploaded files:
send_file = st.file_uploader("Upload send video (.mp4 or .mov)", type=["mp4", "mov"])
fail_files = st.file_uploader("Upload fail video(s)", type=["mp4", "mov"], accept_multiple_files=True)
background_file = st.file_uploader("For better and faster results, add a frame of just the wall without climber (optional) but make sure its from the send video", type=["mp4", "mov"])


if st.button("Generate ghost overlay") and send_file and fail_files:
    with st.spinner("Processing... this might take a while ⏳"):
        
        temp_root = create_temp_workspace()
        # temp_root = os.path.join("static", "temp_sessions", '97ba34fe')  # For dev override, becomes root folder for everything
        st.write(f"📂 Using workspace directory: {temp_root}")

        # Save uploaded videos
        send_video_path = os.path.join(temp_root, "send_video.mp4")
        with open(send_video_path, "wb") as f:
            f.write(send_file.read())
        frame_rate = cv2.VideoCapture(send_video_path).get(cv2.CAP_PROP_FPS)

        #Each fail video is saved to temp_root/fail{i}
        fail_video_paths = []
        for i, uploaded_fail in enumerate(fail_files):
            fail_path = os.path.join(temp_root, f"fail_{i}.mp4") 
            with open(fail_path, "wb") as f:
                f.write(uploaded_fail.read())
            fail_video_paths.append(fail_path) # the paths are saved in a list

        # Where send frames are extracted! temp_root/send_frames
        send_frames_dir = os.path.join(temp_root, "send_frames")
        
        # Where aligned fail frames are extracted! [temp_root/fail_0, temp_root/fail_1/], etc..
        fail_dirs = [os.path.join(temp_root, f"fail{i}") for i in range(len(fail_video_paths))]
        
        # Where alpha masks will be saved and the visual byproduct! [temp_root/fail_0_alpha,temp_root/fail_0_alpha]
        alpha_dirs = [os.path.join(temp_root, f"fail{i}_alpha") for i in range(len(fail_video_paths))]

        # Create necessary folders using defined paths above
        for d in fail_dirs + alpha_dirs + [send_frames_dir]:
            os.makedirs(d, exist_ok=True)
        
        print('Empty directories initialized!! Starting the preprocessing...')

        # 🧠 Step 1: Create background image from send frames
        print('Creating background image...')
        background_path = create_background_image(background_file, send_frames_dir)        
        
        # # 🧼 Step 2: Preprocess videos (extract frames, segment masks, etc.)
        preprocess_all(send_video_path, # temp_root/send_video.mp4
                       fail_video_paths, #[temp_root/fail_video0.mp4, temp_root/fail_video1.mp4]
                       send_frames_dir,# temp_root/send_frames
                       fail_dirs, # [temp_root/fail_0, temp_root/fail_1/]
                       alpha_dirs, # [temp_root/fail_0_alpha,temp_root/fail_0_alpha]
                       temp_root
                       )
        # 👻 Step 3: Generate ghost overlay output video
        os.makedirs(f'{temp_root}/output/', exist_ok = True)
        result_file_path = f'{temp_root}/output/spookysends_output.mp4'
        
        run_spookysends_with_overlay(
        send_frames_dir=send_frames_dir,
        fail_dirs = fail_dirs,
        alpha_dirs=alpha_dirs,
        output_path=result_file_path,
        background_path=background_path,
        temp_root=temp_root,
        frame_rate = frame_rate
    )


        # 🎬 Done
        st.success("🎉 Done! Here's your ghostly video:")
        compressed_path = result_file_path.replace(".mp4", "_compressed.mp4")

        st.video(compressed_path)
        end_time = time.time()
        print(f"⏱️ Total runtime: {(end_time - start_time)/60} minutes")

import streamlit as st

col1, col2 = st.columns(2)

with col1:
    if st.image("frontend/thumb1.PNG", width=150):
        st.markdown("[watch reel](https://www.instagram.com/p/DKL_ifRopcY/)", unsafe_allow_html=True)

with col2:
    if st.image("frontend/thumb2.PNG", width=150):
        st.markdown("[watch reel](https://www.instagram.com/reel/DKRG2mcIs5M/)", unsafe_allow_html=True)
