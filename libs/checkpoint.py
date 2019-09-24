import torch
import os


def save_checkpoint(
    result_path, epoch, encoder, decoder, optimizer, best_loss,
    scheduler=None, add_epoch2name=False
):
    save_states = {
        'epoch': epoch,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss,
    }

    if scheduler is not None:
        save_states['scheduler'] = scheduler.state_dict()

    if add_epoch2name:
        torch.save(
            save_states,
            os.path.join(
                result_path, 'epoch{}_checkpoint.pth'.format(epoch))
        )
    else:
        torch.save(
            save_states, os.path.join(result_path, 'checkpoint.pth'))


def resume(result_path, encoder, decoder, optimizer, scheduler):

    resume_path = os.path.join(result_path, 'checkpoint.pth')
    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(
        resume_path, map_location=lambda storage, loc: storage)

    begin_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    # confirm whether the optimizer matches that of checkpoints
    optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, encoder, decoder, optimizer, best_loss, scheduler
