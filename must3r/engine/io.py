# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
from pathlib import Path

import must3r.tools.path_to_dust3r  # noqa
import dust3r.utils.path_to_croco  # noqa
from croco.utils.misc import save_on_master


def save_model(args, epoch, encoder, decoder, optimizer, loss_scaler, fname=None):
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    optim_state_dict = optimizer.state_dict()
    to_save = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optim_state_dict,
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    print(f'>> Saving model to {checkpoint_path} ...')
    save_on_master(to_save, checkpoint_path)


def load_model(args, chkpt_path, encoder, decoder, optimizer, loss_scaler):
    args.start_epoch = 0
    if chkpt_path is not None:
        checkpoint = torch.load(chkpt_path, map_location='cpu', weights_only=False)

        print("Resume checkpoint %s" % chkpt_path)
        encoder.load_state_dict(checkpoint['encoder'], strict=False)
        decoder.load_state_dict(checkpoint['decoder'], strict=False)
        args.start_epoch = checkpoint['epoch'] + 1
        optim_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(optim_state_dict)
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        else:
            print("")
        print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')
