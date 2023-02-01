# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def cutting():
    #Путь к видео и
    capture = cv2.VideoCapture('C:/Users/Hp/Desktop/monodepth2/assets/videos/trm.169.007.avi')
    frameNr = 0

    while (True):
        success, frame = capture.read()
        if success:
            cv2.imwrite(f'C:/Users/Hp/Desktop/monodepth2/assets/images/{frameNr}.jpg', frame)
        else:
            break

        frameNr = frameNr+1

    capture.release()

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


from PIL import Image
path = r'C:/Users/Hp/Desktop/monodepth2/assets/images/'
for file in os.listdir(path):
    if not file.endswith(".jpg"):
        img = Image.open(path + file).convert('RGB')
        file_name, file_ext = os.path.splitext(file)
        img.save(path + file_name + '.jpg')
        os.remove(path + file_name + '.png')

# grey_path = r'C:/Users/Hp/Desktop/monodepth2/assets/greyscale/'
# count = 0
# for file in os.listdir(grey_path):
#     img = Image.open(grey_path + file).convert('RGB')
#     file_name, file_ext = os.path.splitext(file)
#     img.save(grey_path + str(count) + '.png')
#     os.remove(grey_path + file_name + '.png')
#     count += 1



def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = sorted(glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext))), key=numericalSort)
        output_directory = args.image_path
        output_directory = r'C:/Users/Hp/Desktop/monodepth2/assets/result/'
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))


    # PREDICTING ON EACH IMAGE IN TURN
    output_sequences = []
    output_lidar_sequences = []
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                # np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='gray')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)



            input_image = pil.open(image_path).convert('RGB')
            raw_img = np.array(input_image)
            pred = np.array(im)



            lidar_path = 'C:/Users/Hp/Desktop/monodepth2/assets/best_lidar/' + str(idx) + ".png"
            lidar_open = pil.open(lidar_path).convert('RGB')
            lidar = np.array(lidar_open)
            # lid_output = np.concatenate((raw_img, lidar), axis=0)
            # output_lidar_sequences.append(lid_output)



            # dis_path = 'C:/Users/Hp/Desktop/monodepth2/assets/disparity/sum/' + 'disp_' + str(idx) + ".jpg"
            # dis_open = pil.open(dis_path).convert('RGB')
            # disparity = np.array(dis_open)

            # print(lidar_open.size)
            # print(input_image.size)
            # if (lidar_open.size == input_image.size):
            output = np.concatenate((raw_img, pred, lidar), axis=0)
            # output = np.concatenate((raw_img, pred, disparity), axis=0)
            # output = np.concatenate((raw_img, pred), axis=0)
            output_sequences.append(output)



            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')

    #Оборачиваю картинки в видео
    cap = cv2.VideoCapture(0)
    width = int(output_sequences[0].shape[1] + 0.5)
    height = int(output_sequences[0].shape[0] + 0.5)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        os.path.join('C:/Users/Hp/Desktop/monodepth2/assets', 'result.avi'), fourcc, 8.0, (width, height))

    for frame in output_sequences:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        # uncomment to display the frames
        # cv2.imshow('demo', frame)

        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
        # out.release()

    #Сравнение лидара и карты глубины
    # cap = cv2.VideoCapture(0)
    # width = int(output_lidar_sequences[0].shape[1] + 0.5)
    # height = int(output_lidar_sequences[0].shape[0] + 0.5)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(
    #     os.path.join('C:/Users/Hp/Desktop/monodepth2/assets', 'lidar.avi'), fourcc, 20.0, (width, height))

    # for frame in output_lidar_sequences:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     out.write(frame)
    #     # uncomment to display the frames
    #     cv2.imshow('demo', frame)

    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
        # out.release()


if __name__ == '__main__':
    # cutting()
    args = parse_args()
    test_simple(args)
