import adabound
import argparse
import os
import pandas as pd
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from addict import Dict
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from libs.checkpoint import save_checkpoint, resume
from libs.dataset import MSR_VTT_Features, collate_fn
from libs.meter import AverageMeter, ProgressMeter
from libs.model import EncoderCNN, DecoderRNN
from utils.build_vocab import Vocabulary  # noqa


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--resume', action='store_true',
                        help='Add --resume option if you start training from checkpoint.')

    return parser.parse_args()


def train(train_loader, encoder, decoder, criterion, optimizer, epoch, config, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    end = time.time()

    # switch training mode
    encoder.train()
    decoder.train()

    # freeze bn
    if config.batch_size == 1:
        for m in encoder, decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        features = sample['feature']
        captions = sample['caption']
        lengths = sample['length']
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        batch_size = features.shape[0]

        # data augumentation
        if config.add_noise:
            noise = torch.normal(
                mean=0, std=config.stddev, size=features.shape)
            features += noise

        # send to device
        features = features.to(device)
        captions = captions.to(device)
        targets = targets.to(device)

        # forward
        outputs = encoder(features)
        outputs = decoder(outputs, captions, lengths)

        loss = criterion(outputs, targets)

        # record loss
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 100 iteration
        if i != 0 and i % 10 == 0:
            progress.display(i)

    return losses.avg


def validate(val_loader, encoder, decoder, criterion, config, device):
    losses = AverageMeter('Loss', ':.4e')

    # switch to evaluate mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for sample in val_loader:
            features = sample['feature']
            captions = sample['caption']
            lengths = sample['length']
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True)[0]

            batch_size = features.shape[0]

            # send to device
            features = features.to(device)
            captions = captions.to(device)
            targets = targets.to(device)

            # forward
            outputs = encoder(features)
            outputs = decoder(outputs, captions, lengths)

            loss = criterion(outputs, targets)

            # record loss
            losses.update(loss.item(), batch_size)

    return losses.avg


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # cpu or cuda
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    else:
        print('You have to use GPUs because training CNN is computationally expensive.')
        sys.exit(1)

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None

    # Dataloader
    print("Dataset: {}".format(CONFIG.dataset))
    print(
        "Batch Size: {}\tNum in channels: {}\tAlignment Size: {}\tNum Workers: {}"
        .format(CONFIG.batch_size, CONFIG.in_channels, CONFIG.align_size, CONFIG.num_workers)
    )

    # load vocabulary
    with open(CONFIG.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    train_data = MSR_VTT_Features(
        dataset_dir=CONFIG.dataset_dir,
        feature_dir=CONFIG.feature_dir,
        vocab=vocab,
        ann_file=CONFIG.ann_file,
        mode="train",
        align_size=CONFIG.align_size
    )

    val_data = MSR_VTT_Features(
        dataset_dir=CONFIG.dataset_dir,
        feature_dir=CONFIG.feature_dir,
        vocab=vocab,
        ann_file=CONFIG.ann_file,
        mode="val",
        align_size=CONFIG.align_size
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn,
        drop_last=True if CONFIG.batch_size > 1 else False
    )

    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG.batch_size,
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

    params = list(decoder.parameters()) + list(encoder.linear.parameters())

    if CONFIG.optimizer == 'Adam':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(params, lr=CONFIG.learning_rate)
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            params,
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov
        )
    elif CONFIG.optimizer == 'AdaBound':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = adabound.AdaBound(
            params,
            lr=CONFIG.learning_rate,
            final_lr=CONFIG.final_lr,
            weight_decay=CONFIG.weight_decay
        )
    else:
        print('There is no optimizer which suits to your option.')
        sys.exit(1)

    # learning rate scheduler
    if CONFIG.scheduler == 'onplateau' and CONFIG.optimizer == "SGD":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience
        )
    else:
        scheduler = None

    # resume if you want
    columns = ['epoch', 'lr', 'train_loss', 'val_loss']
    log = pd.DataFrame(columns=columns)

    begin_epoch = 0
    best_loss = 100

    if args.resume:
        if os.path.exists(os.path.join(CONFIG.result_path, 'checkpoint.pth')):
            print('loading the checkpoint...')
            checkpoint = resume(
                CONFIG.result_path, encoder, decoder, optimizer, scheduler)
            begin_epoch, encoder, decoder, optimizer, best_loss, scheduler = checkpoint
            print('training will start from {} epoch'.format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")
        if os.path.exists(os.path.join(CONFIG.result_path, 'log.csv')):
            print('loading the log file...')
            log = pd.read_csv(os.path.join(CONFIG.result_path, 'log.csv'))
        else:
            print("there is no log file at the result folder.")
            print('Making a log file...')

    # criterion for loss
    criterion = nn.CrossEntropyLoss()

    # train and validate encoder, decoder
    print('\n---------------------------Start training---------------------------\n')
    train_losses = []
    val_losses = []

    for epoch in range(begin_epoch, CONFIG.max_epoch):
        # training
        train_loss = train(
            train_loader, encoder, decoder, criterion, optimizer, epoch, CONFIG, device)

        train_losses.append(train_loss)

        # validation
        val_loss = validate(
            val_loader, encoder, decoder, criterion, CONFIG, device)
        val_losses.append(val_loss)

        # scheduler
        if CONFIG.scheduler == 'onplateau':
            scheduler.step(val_loss)

        # save a encoder, decoder if top1 acc is higher than ever
        if best_loss > val_losses[-1]:
            best_loss = val_losses[-1]
            torch.save(
                encoder.state_dict(),
                os.path.join(
                    CONFIG.result_path, 'best_loss_encoder.prm')
            )

            torch.save(
                decoder.state_dict(),
                os.path.join(
                    CONFIG.result_path, 'best_loss_decoder.prm')
            )

        # save checkpoint every epoch
        save_checkpoint(
            CONFIG.result_path, epoch, encoder, decoder, optimizer, best_loss, scheduler)

        # save checkpoint every 10 epoch
        if epoch % 10 == 0 and epoch != 0:
            save_checkpoint(
                CONFIG.result_path, epoch, encoder, decoder,
                optimizer, best_loss, scheduler, add_epoch2name=True
            )

        # tensorboardx
        if writer is not None:
            writer.add_scalars("loss", {
                'train': train_losses[-1],
                'val': val_losses[-1]
            }, epoch)

        # write logs to dataframe and csv file
        tmp = [
            epoch, optimizer.param_groups[0]['lr'], train_losses[-1], val_losses[-1]
        ]
        tmp_df = pd.Series(tmp, index=log.columns)

        log = log.append(tmp_df, ignore_index=True)
        log.to_csv(os.path.join(CONFIG.result_path, 'log.csv'), index=False)

        print(
            'epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}\tval loss: {:.4f}'
            .format(epoch, optimizer.param_groups[0]['lr'], train_losses[-1], val_losses[-1])
        )

    # save encoder, decoders
    torch.save(
        encoder.state_dict(), os.path.join(CONFIG.result_path, 'final_encoder.prm'))
    torch.save(
        decoder.state_dict(), os.path.join(CONFIG.result_path, 'final_decoder.prm'))

    print("Done!")
    print("")


if __name__ == '__main__':
    main()
