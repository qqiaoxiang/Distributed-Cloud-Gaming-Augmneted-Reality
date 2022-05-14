# The test program for Super Resolution

from json.tool import main
from tkinter.filedialog import SaveAs
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

SAVED_MODEL_PATH = "esrgan-tf2_1" # SR model

def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filepath):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image = image.resize((3840, 2160))
    image.save(filepath)
    return True

if __name__ == '__main__':
    hr_image = preprocess_image("image.png")
    model = hub.load(SAVED_MODEL_PATH)
    resolution_image = model(hr_image)
    resolution_image = tf.squeeze(resolution_image)
    save_image(resolution_image, "1.jpg")
