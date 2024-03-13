import time
import os
import cv2
from datasets.dataset import CrackData,img_CrackData
import argparse
import cfg
from os.path import splitext, join
import logging
import matplotlib.image as mpimg

from model import *
import numpy as np
from PIL import Image
from torch.utils import data
import os, cv2, torch, random, io


def onescale_test(model, args):
    if args.kind == 1:
        img_path = args.picture
        logging.info('path: %s' % img_path)
        img = cv2.imread(img_path)
        logging.info('Processing: %s' % img)



        img = np.asarray(img, dtype=np.float32)
        img = img[:, :, ::-1]  # RGB->BGR
        img = img / 255  ######## lk
        data = []
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        img = np.array(img)

        save_dir = args.res_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if args.cuda:
            model.cuda()
        model.eval()
        start_time = time.time()
        timeRecords = open(join(save_dir, 'timeRecords.txt'), "w")
        timeRecords.write('# filename time[ms]\n')
        scale = [1]

            # image = image[0]
        # img = img.array(img)
        img = img.transpose((2, 0, 1))
        image_in = img.transpose((1, 2, 0))
        _, H, W = image_in.shape

        im_ = img.transpose((2, 0, 1))
        tm = time.time()
        results = model(torch.unsqueeze(torch.from_numpy(image_in).cuda(), 0))
        result = F.sigmoid(results[-1]).cpu().data.numpy()[0, :, :]
        result = result * 255
        result = result[:,np.newaxis]
        result = result.transpose((0,2,1))
        ret, binary = cv2.threshold(result, 127, 255, 0)
        cv2.imwrite(os.path.join(save_dir , '%s' % args.picture), binary )
        print(time.time() - start_time)

        from PIL import Image
        import matplotlib.pyplot as plt
        imgfile = args.picture
        pngfile = os.path.join(save_dir , '%s' % args.picture)

        img = cv2.imread(imgfile, 1)
        mask = cv2.imread(pngfile, 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 20)

        img = img[:, :, ::-1]
        img[..., 2] = np.where(mask == 1, 255, img[..., 2])

        plt.imshow(img)
        plt.show()
        cv2.imwrite(os.path.join(save_dir, 'test.png' ), img)


def main():
    import time
    print(time.localtime())
    args = parse_args()

    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logging.info('Loading model...')

    model = CarNet34(
        Encoder=Encoder_v0_762,
        dp=DownsamplerBlock,
        block=BasicBlock_encoder,
        channels=[3, 16, 64, 128, 256],
        decoder_block=non_bottleneck_1d_2,
        num_classes=1
    )

    logging.info('Loading state...')
    model.load_state_dict(torch.load('%s' % (args.model)))
    logging.info('Start image processing...')

    onescale_test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test model performance')
    parser.add_argument('-ca', '--kind', type=int,
                        default=1, help='1 use picture,2 use video')
    parser.add_argument('-d', '--picture', type=str, choices=cfg.config_test.keys(),
                        default='1.png', help='The picturer path')
    parser.add_argument('-v', '--video', type=str, choices=cfg.config_test.keys(),
                        default='Sun520', help='The video path')
    parser.add_argument('-i', '--inputDir', type=str, default=None, help='Input image directory for testing.')
    parser.add_argument('-c', '--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str,
                        default='Sun520_aug_CarNet/CarNet_15000.pth',
                        help='the model to test')
    parser.add_argument('--res-dir', type=str,
                        default='Sun520_aug_CarNet/onescale_test_15e3',
                        help='the dir to store result')
    return parser.parse_args()

def detect():
    logging.basicConfig(format='%(asctime)s %(levelname)s:\t%(message)s', level=logging.INFO)
    main()


if __name__ == '__main__':
    detect()
