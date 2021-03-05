"""
    Test script for PeeledHuman. You can load model checkpoints to test your model using this
    script. It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
    It first creates model and dataset given the option. It will hard-code some parameters.
    It then runs inference for '--num_test' images.

    python test.py                          \
    --test_folder_path /path/to/images/dir/ \
    --results_dir /path/to/results/dir/     \
    --name /checkpoint/name/                \
    --direction AtoB                        \
    --model pix2pix                         \
    --netG resnet_18blocks                  \
    --output_nc 4                           \
    --load_size 512                         \
    --eval
"""

import os
import csv
import cv2
import torch
import imageio
import numpy as np
from shutil import copyfile
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from options.test_options import TestOptions
from models import create_model
from custom_dataset import DepthDataset

def load_image(image_path):

    # Convert image to tensor
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # image
    image = Image.open(image_path).convert('RGB')
    image = self.to_tensor(image)
    return { 'A': image.unsqueeze(0) }


def save_depth_and_mesh(fake_B, rgb, save_dir, filename):


    obj = open(save_dir + filename + '_mesh.obj', 'w+')

    for j in range(4):

        out = np.squeeze(np.transpose(fake_B.cpu().numpy(), (0,2,3,1)))[:,:,j] + 1.0
        plt.imsave(save_dir + filename + str(j) + '_depth.png', 255.0*(2 - out), cmap='gray')
        rgb = None
        if rgb is not None:
            fake_rgb = rgb[:, :, 3*j: 3*j+3]
        else:
            fake_rgb = None
        out_pcl = pointcloud(out, 2*221.70250337, fake_rgb).tolist()

        for vert in out_pcl:
          if rgb is None:
            obj.write("v "+ str(vert[0]) + " "+str(-1 * vert[1])+" "+ str(-1 * vert[2]) + '\n')

            # Thresholding to remove white pixels
          elif not (vert[3] >= 215 and vert[4] >= 215 and vert[5] >= 215):
            obj.write("v "+ str(vert[0]) + " "+str(-1 * vert[1])+" "+ str(-1 * vert[2])+ " " +str(vert[3])+" "+str(vert[4])+" "+str(vert[5])+'\n')
          else:
            pass


def save_rgb(fake_C, save_dir, filename):

    for r in range(fake_C.shape[1]//3):

        out = np.squeeze(np.transpose(fake_C.data.cpu().numpy(), (0,2,3,1)))[:,:,3*r:3*r+3]
        out = ((out+1)/2)*255.0
        try :
            cv2.imwrite(save_dir, filename + str(r) + '_rgb.png', cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        except :
            pass

def pointcloud(depth, foc, rgbs):

    fy = fx = foc
    height = depth.shape[0]
    width = depth.shape[1]
    mask = np.where((depth > 1.1)&(depth < 1.95))

    x = mask[1]
    y = mask[0]

    normalized_x = (x - width * 0.5)
    normalized_y = (y - width * 0.5)

    world_x = normalized_x * depth[y, x] / fx
    world_y = normalized_y * depth[y, x] / fy
    world_z = depth[y, x]

    if rgbs is not None:
      rgb_color = rgbs[y,x,:]
      return np.vstack((world_x, world_y, world_z, rgb_color[:,0], rgb_color[:,1], rgb_color[:,2])).T

    return np.vstack((world_x, world_y, world_z)).T

if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    result_dir = opt.results_dir

    opt.num_threads = 0         # test code only supports num_threads = 1
    opt.batch_size = 1          # test code only supports batch_size = 1
    opt.serial_batches = True   # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True          # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1         # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)   # create a model given opt.model and other options
    model.setup(opt)            # regular setup: load and print networks; create schedulers
    model.eval()

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f)]

    print("Number of images: ", min(int(opt.num_test), len(test_images) ) )

    for idx, image_path in enumerate(test_images):

        if idx == int(opt.num_test):
           break

        try:
            data = load_image(image_path)
            model.set_input(data)
            with torch.no_grad():
                model.forward()

            print('Test sample: ',  idx+1, ": ",  image_path)
            fake_B, fake_C = model.return_test()
            real_B, real_C = model.return_input()

            filepath = image_path.split('/')[-1]
            filename = filepath.split('.')[0]
            rgb = np.squeeze(np.transpose(data['A'].data.cpu().numpy(), (0,2,3,1)))
            cv2.imwrite(result_dir + filename + '_inp.png', 255*cv2.cvtColor(((rgb+1)/2),cv2.COLOR_BGR2RGB ))

            rgb_cat = None
            rgb_cat = np.concatenate([(255*(rgb+1)/2), (255*((np.squeeze(np.transpose(fake_C[:,:,:,:].data.cpu().numpy(), (0,2,3,1))))+1)/2)], 2)
            save_depth_and_mesh(fake_B, rgb_cat, result_dir, filename)
            save_rgb(fake_C, result_dir, filename)

        except Exception as e:
           print("error:", e.args)
