# read the raw downscale video, then super-resolution,
# and then generate a 4K video of the target detection results

from AIDetector_pytorch import Detector
import imutils
import cv2
import tensorflow_hub as hub
import argparse
import os
from super_resolution import preprocess_image, save_image
import tensorflow as tf
import shutil
import time


parser = argparse.ArgumentParser()

parser.add_argument('--video_source', type=str, help='path to video source')
parser.add_argument('--result_path', type=str, default='video_output', help='path to save output video')

sr_model = hub.load("esrgan-tf2_1")
det = Detector()

def main(args):
    sr_start = time.time()
    video_source =r'4kMAR\input_video\video2.mp4'
    # args.video_source
    cap = cv2.VideoCapture(video_source)
    
    original_images_dir = "saved_assets/original_images"
    sr_images_dir = "saved_assets/sr_images"
    
    if os.path.exists(original_images_dir):
        shutil.rmtree(original_images_dir)
    os.mkdir(original_images_dir)
    
    if os.path.exists(sr_images_dir):
        shutil.rmtree(sr_images_dir)
    os.mkdir(sr_images_dir)

    # Read the source video
    # and use OpencV to save each frame of the video
    # in the directory: 'saved_Assets\Original_images'.
    # ----------------------------------------------------------------
    sr_cnt = 0
    original_images_names = []
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        # original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_image = frame
        image_name = "{}.jpg".format(sr_cnt)
        image_path = os.path.join(original_images_dir, image_name)
        cv2.imwrite(image_path, original_image)
        original_images_names.append(image_name)
        sr_cnt += 1
    # ----------------------------------------------------------------
    cap.release()

    sr_size= ()
    # Super Resolution each frame of the video,
    # and save it in the directory: 'saved_Assets\sr_images',
    # and then organize it into a video via cv2.videoWriter()
    # ------------------------------------------------------------------------------------
    for image_name in original_images_names:
        image_path = os.path.join(original_images_dir, image_name)
        hr_image = preprocess_image(image_path)
        sr_image = sr_model(hr_image)
        # print(type(sr_image))
        # sr_image = cv2.resize(sr_image, (3840, 2160))
        sr_size = (sr_image.shape[2], sr_image.shape[1])
        sr_image = tf.squeeze(sr_image)
        sr_image_path = os.path.join(sr_images_dir, image_name)
        save_image(sr_image, sr_image_path)

    sr_end = time.time()
    sr_time = sr_end-sr_start
    print("The time for SR: ", sr_time)

    # SR image--> video, then save
    sr_images_names = sorted(original_images_names, key=lambda x: int(x.split('.')[0]))
    sr_video_path = "saved_assets\sr_video\sr.mp4"

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
    writer = cv2.VideoWriter(sr_video_path, fourcc, 30, (3840, 2160))

    for sr_image_name in sr_images_names:
        sr_image_path = os.path.join(sr_images_dir, sr_image_name)
        sr_image = cv2.imread(sr_image_path)
        writer.write(sr_image)
    writer.release()
    # -----------------------------------------------------------------------------------

    DectTrack_start = time.time()

    # call the interface feedCap of AIDetector_pytorch
    # to use the post-detection video for target detection model and target video
    # -----------------------------------------------------------------------------------
    cap = cv2.VideoCapture(sr_video_path)
    out_video_path = args.result_path
    final_writer = cv2.VideoWriter(out_video_path, fourcc, 30, sr_size)

    while True:
        _, sr_frame = cap.read()
        if sr_frame is None:
            break

        detect_res = det.feedCap(sr_frame)
        res_image = detect_res['frame']
        final_writer.write(res_image)
    # -----------------------------------------------------------------------------------

    DectTrack_end = time.time()
    DectTrack_time = DectTrack_end - DectTrack_start
    print("Yolov5 + Deep Sort: ", DectTrack_time)

    cap.release()
    final_writer.release()
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
        


