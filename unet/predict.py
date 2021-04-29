import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import UNet2D
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                size=(256,256),
                out_threshold=0.5):
    net.eval()
    full_img = np.array(full_img)
    img = torch.from_numpy(BasicDataset.preprocess(full_img, size))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_chans_out > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.shape[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask
    # Old output
    # return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--size', '-s', type=float,
                        help="Size of the input images",
                        default=(128,128))

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    # Old version
    # out_files = get_output_filenames(args)

    out_files = [os.path.join(os.getcwd(), 'data', 'output30_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output35_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output40_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output45_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output50_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output55_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output60_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output65_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output70_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output75_s.jpg'),
                 os.path.join(os.getcwd(), 'data', 'output80_s.jpg')
                 ]

    net = UNet2D(n_chans_in=1, n_chans_out=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    model_path = "E:\Thesis\conp-dataset\projects\calgary-campinas\CC359\Test\checkpoints\CP_epoch79.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))


    # net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        # Old version
        # img = Image.open(fn)

        img = Image.open(os.path.join(os.getcwd(), 'data', 'input_s.jpg'))

        mask = predict_img(net=net,
                           full_img=img,
                           size=args.size,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            for k in range (0,10):
                result = mask_to_image(mask>0.3+k/20)
                result.save(out_files[k])


            logging.info("Mask saved to {}".format(out_files[i]))

        k = 1

        # segmentation
        seg = np.zeros((100, 100), dtype='int')
        seg[30:70, 30:70] = k

        # ground truth
        gt = np.zeros((100, 100), dtype='int')
        gt[30:70, 40:80] = k

        dice = np.sum(seg[gt == k]) * 2.0 / (np.sum(seg) + np.sum(gt))

        print
        'Dice similarity score is {}'.format(dice)
        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
