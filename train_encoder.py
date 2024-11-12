import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from src.dataset import CodeDataset, get_dataset_filelist
from src.encoder import Encoder
from src.code_predictor import CodePredictor
from src.utils import scan_checkpoint, load_checkpoint, count_parameters, \
    save_checkpoint, build_env, AttrDict


torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.dist_config['world_size'] * h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    encoder = Encoder(h).to(device)
    code_predictor = CodePredictor(h.upsample_initial_channel, h.num_codes).to(device)
    
    encoder_criterion = torch.nn.CrossEntropyLoss()

    if rank == 0:
        print(encoder)
        print("# Parameter : ", count_parameters(encoder))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    last_epoch = -1
    if cp_g is None or cp_do is None:
        state_dict_do = None
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        encoder.load_state_dict(state_dict_g['encoder'])
        code_predictor.load_state_dict(state_dict_do['code_predictor'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        code_predictor = DistributedDataParallel(code_predictor, device_ids=[rank]).to(device)

    optim_e = torch.optim.AdamW(itertools.chain(encoder.parameters(), code_predictor.parameters()), 
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_e.load_state_dict(state_dict_do['optim_e'])

    scheduler_e = torch.optim.lr_scheduler.ExponentialLR(optim_e, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, 
                           h.sampling_rate, device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, sampler=train_sampler,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, 
                               h.sampling_rate, False, device=device)
        validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    encoder.train()
    code_predictor.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _ = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            x = torch.autograd.Variable(x.to(device, non_blocking=False))
            x = x.unsqueeze(1)
            
            # Streaming Encoder
            if hasattr(encoder, 'module'):
                encoder_buf = encoder.module.init_buffers(x.shape[0], x.device)
                y_s_hat, encoder_buf = encoder.module(x, encoder_buf)
                code_logits = code_predictor.module(y_s_hat)
            else:
                encoder_buf = encoder.init_buffers(x.shape[0], x.device)
                y_s_hat, encoder_buf = encoder(x, encoder_buf)
                code_logits = code_predictor(y_s_hat)
            code = y
            y_hat = code_logits.argmax(dim=-1)

            encoder_loss = encoder_criterion(code_logits.transpose(1,2), code)
            encoder_loss.backward()

            if ((steps + 1) % h.encoder_accum_steps == 0) or (steps + 1 == len(train_loader)):
                # accumulate gradients for accum steps
                for param in encoder.parameters():
                    param.grad /= h.encoder_accum_steps
                optim_e.step()
                optim_e.zero_grad()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        accuracy = y_hat == y
                        accuracy = torch.mean(accuracy.float()).item()

                    print(
                        'Steps : {:d}, Enc.Loss: {:4.3f},  Accuracy : {:4.3f}, s/b :{:4.3f}'.format(
                            steps,
                            encoder_loss.item(),
                            accuracy,
                            time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'encoder': (encoder.module if h.num_gpus > 1 else encoder).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'code_predictor': (code_predictor.module if h.num_gpus > 1 else code_predictor).state_dict(),
                                                      'optim_e': optim_e.state_dict(),
                                                      'steps': steps, 'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/encoder_loss", encoder_loss, steps)
                    sw.add_scalar("training/accuracy", accuracy, steps)
                    
                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    encoder.eval()
                    code_predictor.eval()
                    torch.cuda.empty_cache()
                    accuracy_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _ = batch
                            y = y.to(device, non_blocking=False)
                            x = x.to(device, non_blocking=False)

                            if hasattr(encoder, 'module'):
                                encoder_buf = encoder.module.init_buffers(x.shape[0], x.device)
                                y_s_hat, encoder_buf = encoder.module(x.unsqueeze(1), encoder_buf)

                                code_logits = code_predictor.module(y_s_hat)
                            else:
                                encoder_buf = encoder.init_buffers(x.shape[0], x.device)
                                y_s_hat, encoder_buf = encoder(x.unsqueeze(1), encoder_buf)

                                code_logits = code_predictor(y_s_hat)

                            y_hat = code_logits.argmax(dim=-1)

                            accuracy = y_hat == y
                            accuracy_tot += torch.mean(accuracy.float()).item()

                        val_err = accuracy_tot / (j + 1)
                        sw.add_scalar("validation/accuracy", val_err, steps)
                    encoder.train()
                    code_predictor.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_e.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    if rank == 0:
        print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--checkpoint_path', default='experiments/base/encoder')
    parser.add_argument('-c', '--config', default='experiments/base/config.json')
    parser.add_argument('--training_epochs', default=6000, type=int)
    parser.add_argument('--training_steps', default=300000, type=int)
    parser.add_argument('--stdout_interval', default=20, type=int)
    parser.add_argument('--checkpoint_interval', default=50000, type=int)
    parser.add_argument('--summary_interval', default=50, type=int)
    parser.add_argument('--validation_interval', default=10, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    # build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
