import argparse
import cv2
import h5py
import io
import numpy as np
import os
import pickle
import sys
import torch
import torch.nn.functional as F
import tqdm
import yaml

from addict import Dict
# from joblib import delayed, Parallel
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from libs.cam import GradCAM, GradCAMpp
from libs.dataset import MSR_VTT_Features, collate_fn
from libs.model import EncoderCNN, DecoderRNN
from utils.build_vocab import Vocabulary  # noqa


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('config', type=str, help='path to a config file')
    parser.add_argument('mode', type=str, help='training, validation or test')
    parser.add_argument(
        'cam', type=str, help='[gradcam, gradcampp]')
    parser.add_argument(
        '--encoder', type=str, default=None,
        help='path to the trained encoder. If you do not specify, the trained model, \'best_acc1_model.prm\' in result directory will be used.')
    parser.add_argument(
        '--decoder', type=str, default=None,
        help='path to the trained decoder. If you do not specify, the trained model, \'best_acc1_model.prm\' in result directory will be used.')

    return parser.parse_args()


def visualize(clip, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        clip: (Tensor) shape => (1, 3, T, H, W)
        cam: (Tensor) shape => (1, 1, T, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, T, H, W)
    """

    _, _, T, H, W = clip.shape
    cam = F.interpolate(
        cam, size=(T, H, W), mode='trilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmaps = []
    for t in range(T):
        c = cam[t]
        heatmap = cv2.applyColorMap(np.uint8(c), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
        heatmap = heatmap.float() / 255
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])
        heatmaps.append(heatmap)

    heatmaps = torch.stack(heatmaps)
    heatmaps = heatmaps.transpose(1, 0).unsqueeze(0)
    result = heatmaps + clip.cpu()
    result = result.div(result.max())

    return result


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # Dataloader
    print("Dataset: {}".format(CONFIG.dataset))

    # load vocabulary
    with open(CONFIG.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Dataloader
    data = MSR_VTT_Features(
        dataset_dir=CONFIG.dataset_dir,
        feature_dir=CONFIG.feature_dir,
        vocab=vocab,
        ann_file=CONFIG.ann_file,
        mode=args.mode,
        align_size=CONFIG.align_size
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn
    )

    # load encoder, decoder
    print('\n------------------------Loading encoder, decoder------------------------\n')
    encoder = EncoderCNN(CONFIG.in_channels, CONFIG.embed_size)
    decoder = DecoderRNN(
        CONFIG.embed_size, CONFIG.hidden_size, len(vocab), CONFIG.num_layers
    )

    # load the state dicts of the encoder and the decoder
    if args.encoder is not None:
        encoder_state_dict = torch.load(args.encoder)
    else:
        encoder_state_dict = torch.load(
            os.path.join(CONFIG.result_path, 'best_loss_encoder.prm')
        )
    if args.decoder is not None:
        decoder_state_dict = torch.load(args.encoder)
    else:
        decoder_state_dict = torch.load(
            os.path.join(CONFIG.result_path, 'best_loss_decoder.prm')
        )

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    # target_layer for visualizing grad cam
    target_layer = encoder.conv

    if args.cam == 'gradcam':
        print('Grad-CAM will be used for visualization')
        wrapped_model = GradCAM(encoder, decoder, target_layer)
    elif args.cam == 'gradcampp':
        print('Grad-CAM++ will be used for visualization')
        wrapped_model = GradCAMpp(encoder, decoder, target_layer)
    else:
        print('You have to choose gradcam or gradcampp.')
        sys.exit(1)

    # mkdir
    save_dir = os.path.join(CONFIG.result_path, args.cam)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('Making a directory for saving cams...')

    # switch model mode
    wrapped_model.encoder.eval()
    wrapped_model.decoder.eval()

    # record saved videos
    records = set()
    for sample in tqdm.tqdm(loader, total=len(loader)):
        idx = sample['id'][0]

        # ignore videos whose cams are saved once
        if idx in records:
            continue
        else:
            records.add(idx)

        feature = sample['feature']
        caption = sample['caption']
        length = sample['length']

        # generate cam
        cam = wrapped_model(feature, caption, length)

        # load clip
        video_path = os.path.join(
            CONFIG.dataset_dir,
            CONFIG.hdf5_dir,
            "video{}.hdf5".format(idx)
        )
        with h5py.File(video_path, 'r') as f:
            video = f['video']
            clip = []
            n_frames = len(video)
            for j in range(n_frames):
                img = Image.open(io.BytesIO(video[j]))
                img = transforms.functional.to_tensor(img)
                clip.append(img)

        clip = torch.stack(clip)
        clip = clip.unsqueeze(0).transpose(1, 2)

        # synthesize cam and clip
        # shape (1, 3, T, H, W)
        heatmaps = visualize(clip, cam)
        # shape (3, T, H, W)
        heatmaps = heatmaps.squeeze(0)

        # save cams as image
        video_cam_dir = os.path.join(save_dir, "video{}".format(idx))
        if not os.path.exists(video_cam_dir):
            os.mkdir(video_cam_dir)

        for j in range(n_frames):
            heatmap = heatmaps[:, j]
            save_image(
                heatmap,
                os.path.join(video_cam_dir, "cam{:06}.png".format(j))
            )

    print('Done')
    print("")


if __name__ == '__main__':
    main()
