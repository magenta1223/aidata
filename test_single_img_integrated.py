## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

# Load Libraries
import numpy as np
import os
import argparse
from tqdm import tqdm
from glob import glob
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from Restormer.Deraining import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose
import mmcv
import wget

import random


SEG_MODELS = [ m for m in glob("./configs/segmentation/*") if "_" not in m]

def parse_lines(line):
    model_url = [s for s in line.split("|") if "model" in s][0]
    return re.sub("\[model\]|\(|\)|\\\\", "", model_url).strip()

def get_url(model_nm, markdown):
    model_url = [ parse_lines(line) for line in markdown if "https://download.openmmlab.com/mmsegmentation/" in line and model_nm in line][0]
    weights_nm = model_url.split("/")[-1]
    return model_url, weights_nm



class FromImage:
    """MM Segmentation pipeline w/o io"""

    def __call__(self, results):
        results['filename'] = None
        results['ori_filename'] = None
        img = results['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def pad_image(img, args):
    factor = args.factor
    h,w = img.shape[2], img.shape[3]
    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    img = F.pad(img, (0,padw,0,padh), 'reflect')
    return img

def preprocess(img, args, img2):
    img = np.float32(img)/255.
    # if "gaussian" in args.Dconfig.lower() and "denois" in args.Dconfig.lower():
    #     sigmas = np.int_(args.sigmas.split(','))
    #     sigma = random.sample(sigmas, 1)[0]
    #     img += np.random.normal(0, sigma/255., img.shape)

    img = torch.from_numpy(img).permute(2,0,1)

    if "defocusdeblur" in args.Dconfig.lower():
        img2 = np.float32(img2)/255.
        img2 = torch.from_numpy(img2).permute(2,0,1)
        img = torch.cat([img, img2], 0)



    input_ = img.unsqueeze(0).cuda()
    return pad_image(input_, args)


#   real denoising dnd : 뭐가 많이 다름.
#   real denoising sidd : 뭐가 많이 다름.




def run_restormer(restormer, img, args, img2 = None):

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()


    ### Derain ### 
    # img = np.float32(img)/255.
    # img = torch.from_numpy(img).permute(2,0,1)
    # input_ = img.unsqueeze(0).cuda()

    # Padding in case images are not multiples of 8
    # input_ = pad_image(input_, factor)

    h,w = img.shape[0], img.shape[1]
    input_ = preprocess(img, args, img2)

    with torch.no_grad():
        restored = restormer(input_)

    # Unpad images to original dimensions
    restored = restored[:,:,:h,:w]
    restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    return restored

### Segmentation ###

def segmentation(segmentor, img):

    cfg = segmentor.cfg 
    device = next(segmentor.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [FromImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    img_data = dict(img=img)
    img_data = test_pipeline(img_data)
    data.append(img_data)

    data = collate(data, samples_per_gpu=1)

    if next(segmentor.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = segmentor(return_loss=False, rescale=True, **data)

    return result


def merge_results(
    img,
    seg_result,
    classes,
    palette = None, # color mapping
    opacity = 0.5, # hyperparameter for alpha blend
    out_file=None, # save path
    # win_name='', # pyplot's window name when show = True
    # show=False, # show or not
    # wait_time=0 # timeout 
    ): 
    """
    segmentation 결과와 원본 이미지의 alpha blending
    """

    
    seg = seg_result[0]
    if palette is None:
        # Get random state before set seed,
        # and restore random state later.
        # It will prevent loss of randomness, as the palette
        # may be different in each iteration if not specified.
        # See: https://github.com/open-mmlab/mmdetection/issues/5844
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 255, size=(len(classes), 3))
        np.random.set_state(state)

    palette = np.array(palette)
    assert palette.shape[0] == len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)

    mmcv.imwrite(img, out_file)


def main():
    parser = argparse.ArgumentParser(description='Integrated Arguments for Deraining & Segmentation')
    # single image라서 제거 
    # parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')

    # single image path
    parser.add_argument('--img', help='Image file') 
    parser.add_argument('--input_dir', default= None, help='Directory of validation images')


    # deraining configs & weights
    parser.add_argument('--Dconfig', default='Restormer/Deraining/Options/Deraining_Restormer.yml', type=str, help='Path to weights')
    parser.add_argument('--Dweights', default='Restormer/Deraining/pretrained_models/deraining.pth', type=str, help='Path to weights')

    # segmentation configs & weights
    parser.add_argument('--Sconfig', default = "pspnet_r50-d8_512x1024_40k_cityscapes.py", help='Config file')
    parser.add_argument('--Sweights', default = "pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth", help='Checkpoint file')

    # result file path
    parser.add_argument('--out_file', default=None, help='Path to output file')

    # result directory
    parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')

    # gpu 
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    # color palette for segmenation. optional
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')

    # segmentation opacity 
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')

    parser.add_argument('--factor', default=8, help='factor for padding')
    args = parser.parse_args()


    #################  DERAINING    #################

    config_derain = yaml.load(open(args.Dconfig, mode='r'), Loader=Loader)
    s = config_derain['network_g'].pop('type') # unused
    restormer = Restormer(**config_derain['network_g'])
    checkpoint = torch.load(args.Dweights)
    restormer.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ",args.Dweights)
    restormer.cuda()
    restormer = nn.DataParallel(restormer)
    restormer.eval()

    #################   SEGMENTATION  #################
    S = args.Sconfig.split("/")
    
    with open(f"./configs/segmentation/{S[2]}/README.md", "r") as f:    
        markdown = f.readlines()    

    model_url, weights_nm = get_url(S[3].replace(".py", ""), markdown)

    if not os.path.exists(f"./configs/segmentation/{S[2]}/{weights_nm}"):
        print("Weights for this configuration not found. Download pretrained weights")
        wget.download(model_url, out = f"./configs/segmentation/{S[2]}/{weights_nm}")
        
    Segmentor = init_segmentor(args.Sconfig, args.Sweights, device=args.device) # build & load model 

    #################   INFERENCE    #################

    # prepare directory for results
    os.makedirs(args.result_dir, exist_ok=True)

    if args.input_dir is not None:
        imgs = glob(f"{args.input_dir}/*")
    else :
        imgs = [args.img]

    for img in imgs:
        img = utils.load_img(args.img) # 0~255

        img = run_restormer(restormer, img, args) # 0~1
        img = img*255
        segmented = segmentation(Segmentor, img) 

        merge_results(
            img = img,
            seg_result = segmented,
            classes = Segmentor.CLASSES,
            palette = get_palette(args.palette), # color mapping
            opacity = args.opacity, # hyperparameter for alpha blend
            out_file=args.out_file, # save path    
        )


if __name__ == "__main__":
    main()
