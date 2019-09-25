import argparse
import os
import pandas as pd
import pickle
import torch
import yaml

from addict import Dict
from torch.utils.data import DataLoader

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
    parser.add_argument('mode', type=str, help='validation or test')
    parser.add_argument(
        '--encoder', type=str, default=None,
        help='path to the trained encoder. If you do not specify, the trained model, \'best_acc1_model.prm\' in result directory will be used.')
    parser.add_argument(
        '--decoder', type=str, default=None,
        help='path to the trained decoder. If you do not specify, the trained model, \'best_acc1_model.prm\' in result directory will be used.')
    parser.add_argument(
        '--cpu', action='store_true', help='Add --cpu option if you use cpu.')

    return parser.parse_args()


def test(loader, encoder, decoder, vocab, device):
    # TODO: support test when batchsize > 1
    # switch to evaluate mode
    encoder.eval()
    decoder.eval()

    df = pd.DataFrame(
        columns=["video_id", "pred_caption", "one of gt_captions"]
    )

    print("------------------- Predicted Captions -------------------")
    with torch.no_grad():
        for sample in loader:
            features = sample['feature']
            captions = sample['caption']
            ids = sample['id']

            # send to device
            features = features.to(device)

            # forward
            features = encoder(features)
            sampled_ids = decoder.sample(features)
            sampled_ids = sampled_ids[0].to('cpu').numpy()

            # convert word ids to words
            gt_caption = []
            captions = captions[0].numpy()
            for word_id in captions:
                word = vocab.idx2word[word_id]
                gt_caption.append(word)
                if word == "<end>":
                    break
            gt_caption = ' '.join(gt_caption)

            pred_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                pred_caption.append(word)
                if word == "<end>":
                    break
            pred_caption = ' '.join(pred_caption)

            print("Video ID: {}\tCap: {}".format(ids[0], pred_caption))

            tmp = pd.Series([
                ids[0],
                pred_caption,
                gt_caption,
            ], index=df.columns)

            df = df.append(tmp, ignore_index=True)

    return df


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # cpu or gpu
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True

    # Dataloader
    print("Dataset: {}".format(CONFIG.dataset))

    # load vocabulary
    with open(CONFIG.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

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

    # send the encoder, decoder to cuda/cpu
    encoder.to(device)
    decoder.to(device)

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

    # train and validate model
    print('\n------------------------Start testing------------------------\n')

    # test
    df = test(loader, encoder, decoder, vocab, device)

    df.to_csv(
        os.path.join(CONFIG.result_path, '{}_log.csv').format(args.mode), index=False)

    print("Done")
    print("")


if __name__ == '__main__':
    main()
