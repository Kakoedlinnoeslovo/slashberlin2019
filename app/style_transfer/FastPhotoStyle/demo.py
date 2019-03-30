"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import torch
from . import process_stylization
from .photo_wct import PhotoWCT

from .photo_gif import GIFSmoothing


# parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
# parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
# parser.add_argument('--content_image_path', default='./images/content1.png')
# parser.add_argument('--content_seg_path', default=[])
# parser.add_argument('--style_image_path', default='./images/style1.png')
# parser.add_argument('--style_seg_path', default=[])
# parser.add_argument('--output_image_path', default='./results/example1.png')
# parser.add_argument('--save_intermediate', action='store_true', default=False)
# parser.add_argument('--fast', action='store_true', default=False)
# parser.add_argument('--no_post', action='store_true', default=False)
# parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
# args = parser.parse_args()
#
# Load model
# p_wct = PhotoWCT()
# p_wct.load_state_dict(torch.load(args.model))
#
# if args.fast:
#     from photo_gif import GIFSmoothing
#     p_pro = GIFSmoothing(r=35, eps=0.001)
# else:
#     from photo_smooth import Propagator
#     p_pro = Propagator()
# if args.cuda:
#     p_wct.cuda(0)
#
# process_stylization.stylization(
#     stylization_module=p_wct,
#     smoothing_module=p_pro,
#     content_image_path=args.content_image_path,
#     style_image_path=args.style_image_path,
#     content_seg_path=args.content_seg_path,
#     style_seg_path=args.style_seg_path,
#     output_image_path=args.output_image_path,
#     cuda=args.cuda,
#     save_intermediate=args.save_intermediate,
#     no_post=args.no_post
# )


class MyNvidiaWrapper:
    def __init__(self):
        self._p_wct = PhotoWCT()
        self._p_wct.load_state_dict(
            torch.load('/home/roman/Desktop/deep-anonymizer/app/style_transfer/FastPhotoStyle/PhotoWCTModels/photo_wct.pth'))

        self._p_pro = GIFSmoothing(r=35, eps=0.001)

        self._p_wct.cuda(0)

    def transform(self, content_image_path, style_image_path, output_image_path):
        process_stylization.stylization(
            stylization_module=self._p_wct,
            smoothing_module=self._p_pro,
            content_image_path=content_image_path,
            style_image_path=style_image_path,
            content_seg_path=None,
            style_seg_path=None,
            output_image_path=output_image_path,
            cuda=0,
            save_intermediate=False,
            no_post=False
        )