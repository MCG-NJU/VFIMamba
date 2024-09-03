import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VFIMamba', type=str)
parser.add_argument('--scale', default=0, type=float)

args = parser.parse_args()
assert args.model in ['VFIMamba_S', 'VFIMamba'], 'Model not exists!'


'''==========Model setting=========='''
TTA = False
model = Model.from_pretrained(args.model)
model.eval()
model.device()


print(f'=========================Start Generating=========================')

I0 = cv2.imread('example/im1.png')
I2 = cv2.imread('example/im2.png')

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

mid = (padder.unpad(model.inference(I0_, I2_, True, TTA=TTA, fast_TTA=TTA, scale=args.scale))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
mimsave('example/out_2x_hf.gif', images, fps=3)


print(f'=========================Done=========================')