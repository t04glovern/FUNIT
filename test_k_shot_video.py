"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import threading
import time
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils import get_config
from trainer import Trainer

import argparse
import cv2


class State:
    # Device ID (typically 0)
    video_device_id = 0
    # can alternatively be a RTSP endpoint
    # video_device_id = 'http://192.168.0.3:8080/video/mjpeg'

    # Local render scale factor.
    display_scale = 3

    # Current frame.
    frame = None

    # output frame
    output_frame = None

    # When changed to false, program will be terminated.
    running = True


# Init state for camera
state = State()

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='configs/funit_animals.yaml')
parser.add_argument('--ckpt',
                    type=str,
                    default='pretrained/animal119_gen_00200000.pt')
parser.add_argument('--class_image_folder',
                    type=str,
                    default='images/n02138411')
parser.add_argument('--input',
                    type=str,
                    default='images/input_content.jpg')
parser.add_argument('--output',
                    type=str,
                    default='images/output.jpg')

"""read frame from camera"""
def read_frame_thread():
    try:
        capture = cv2.VideoCapture(state.video_device_id)
        while state.running:
            _, frame = capture.read()
            state.frame = frame
            time.sleep(0.01)

    except Exception as e:
        print(e)
        state.running = False

"""process any key presses"""
def process_events():
    if state.output_frame is not None:
        # display frame
        cv2.imshow('frame', cv2.resize(state.output_frame, None,
                                        fx=state.display_scale, fy=state.display_scale))

    if cv2.waitKey(1) & 0xff == 27:
        state.running = False

"""video stream"""
def video():
    opts = parser.parse_args()
    cudnn.benchmark = True
    opts.vis = True
    config = get_config(opts.config)
    config['batch_size'] = 1
    config['gpus'] = 1

    trainer = Trainer(config)
    trainer.cuda()
    trainer.load_ckpt(opts.ckpt)
    trainer.eval()

    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize((128, 128))] + transform_list
    transform = transforms.Compose(transform_list)

    print('Compute average class codes for images in %s' %
          opts.class_image_folder)
    images = os.listdir(opts.class_image_folder)
    for i, f in enumerate(images):
        fn = os.path.join(opts.class_image_folder, f)
        img = Image.open(fn).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            class_code = trainer.model.compute_k_style(img_tensor, 1)
            if i == 0:
                new_class_code = class_code
            else:
                new_class_code += class_code
    final_class_code = new_class_code / len(images)

    # Run frame thread
    threading.Thread(target=read_frame_thread).start()

    while state.running:
        if state.frame is None:
            time.sleep(0.01)
            continue

        # Get recent frame
        frame = state.frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame)
        content_img = transform(frame_image).unsqueeze(0)

        # TEMP for output
        content_img_out = content_img.detach().cpu().squeeze().numpy()
        content_img_out = np.transpose(content_img_out, (1, 2, 0))

        output_image = trainer.model.translate_simple(
            content_img, final_class_code)
        image = output_image.detach().cpu().squeeze().numpy()
        image = np.transpose(image, (1, 2, 0))

        state.output_frame = np.concatenate((image, content_img_out), axis=1)

        # handle key press events
        process_events()


if __name__ == '__main__':
    video()
