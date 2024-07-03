import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer_finetune import Model
from benchmark.utils.padder import InputPadder

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VFIMamba_S', type=str)
parser.add_argument('--n', default=16, type=int)
parser.add_argument('--scale', default=0.0, type=float)

args = parser.parse_args()
assert args.model in ['VFIMamba_S', 'VFIMamba'], 'Model not exists!'


'''==========Model setting=========='''
TTA = False
if args.model == 'VFIMamba':
    TTA = True
    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 3, 3]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()

def _recursive_generator(frame1, frame2, down_scale, num_recursions, index):
    if num_recursions == 0:
        yield frame1, index
    else:
        mid_frame = model.inference(frame1, frame2, True, TTA=TTA, fast_TTA=TTA, scale=args.scale)
        id = 2 ** (num_recursions - 1)
        yield from _recursive_generator(frame1, mid_frame, down_scale, num_recursions - 1, index - id)
        yield from _recursive_generator(mid_frame, frame2, down_scale, num_recursions - 1, index + id)


print(f'========================= Start Generating=========================')

I0 = cv2.imread(f'example/im1.png')
I2 = cv2.imread(f'example/im2.png') 

I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

down_scale = 1.0

padder = InputPadder(I0_.shape, divisor=32)
I0_, I2_ = padder.pad(I0_, I2_)

images = [] 
frames = list(_recursive_generator(I0_, I2_, down_scale, int(math.log2(args.n)), args.n//2))
frames = sorted(frames, key = lambda x: x[1])
ans = []

for pred, _ in frames:
    pred = pred[0]
    pred = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    ans.append(pred)

mimsave(f'example/out_{args.n}x.gif', [x[:, :, ::-1] for x in ans], fps=8)

print(f'=========================Done=========================')