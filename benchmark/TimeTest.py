from time import time
import sys
import torch
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer_finetune import Model

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours_small', type=str)
parser.add_argument('--H', default=256, type=int)
parser.add_argument('--W', default=256, type=int)
args = parser.parse_args()
assert args.model in ['VFIMamba_S', 'VFIMamba'], 'Model not exists!'

'''==========Model setting=========='''
TTA = False
if args.model == 'VFIMamba':
    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 3, 3]
    )

model = Model(-1)
model.eval()
model.device()

print("params", sum([param.nelement() for param in model.net.parameters()])/1e6)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

H, W = args.H, args.W
I0 = torch.rand(1, 3, H, W).cuda()
I1 = torch.rand(1, 3, H, W).cuda()

print(f'Test model: {model.name}  TTA: {TTA}')
with torch.no_grad():
    for i in range(50):
        pred = model.inference(I0, I1, True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time()
    for i in range(100):
        pred = model.inference(I0, I1, True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print((time() - time_stamp) / 100 * 1000)