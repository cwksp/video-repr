import argparse
import os
import os.path as osp
import logging

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

import datasets
import models
import utils


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def adjust_lr(optimizer, epoch, max_epoch, config):
    # lr = config['optimizer']['args']['lr']
    # for g in optimizer.param_groups:
    #     g['lr'] = lr
    pass


def train_step(model, data, gt, optimizer):
    pred = model(data)
    loss = F.mse_loss(pred, gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
    }


def eval_step(model, data, gt):
    with torch.no_grad():
        pred = model(data)
        loss = F.mse_loss(pred, gt)

    return {
        'loss': loss.item(),
    }


def sample_data_batch(dataset, n_sample=8):
    t = len(dataset) // n_sample
    ret = []
    for i in range(n_sample):
        ret.append(dataset[i * t])
    return torch.stack(ret)


def process_vis_batch(x, k_m=5, k_d=5):
    n = x.shape[0] // 2
    d = x[:n] - x[n:]
    ret = []
    for i in range(n):
        a, b = x[i], x[n + i]
        mask = (torch.abs(d[i]).mean(dim=0, keepdim=True) * k_m).clamp(0, 1)
        diif = (torch.abs(d[i]) * k_d).clamp(0, 1)
        ret.extend([a, b, a * mask, diif])
    ret = torch.stack(ret)
    return ret.detach().cpu()


def log_temp_scalar(k, v, t):
    writer.add_scalar(k, v, global_step=t)
    wandb.log({k: v}, step=t)


def log_temp_videos(tag, vids, step):
    vids = process_vis_batch(vids)
    vids = (vids.permute(0, 2, 1, 3, 4).numpy() * 255).astype('uint8')
    wandb.log({tag: [wandb.Video(x) for x in vids]}, step=step)


def log_temp_images(tag, imgs, step):
    imgs = process_vis_batch(imgs)
    wandb.log({tag: [wandb.Image(x) for x in imgs]}, step=step)


def main(config):
    # Environment setup
    save_dir = config['save_dir']
    utils.ensure_path(save_dir)
    with open(osp.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    global log, writer
    logger = set_logger(osp.join(save_dir, 'log.txt'))
    log = logger.info
    writer = SummaryWriter(osp.join(save_dir, 'tensorboard'))

    os.environ['WANDB_NAME'] = config['exp_name']
    os.environ['WANDB_DIR'] = config['save_dir']
    if not config.get('wandb_upload', False):
        os.environ['WANDB_MODE'] = 'dryrun'
    t = config['wandb']
    os.environ['WANDB_API_KEY'] = t['api_key']
    wandb.init(project=t['project'], entity=t['entity'], config=config)

    log('logging init done.')
    log(f'wandb id: {wandb.run.id}')

    # Dataset, model and optimizer
    train_dataset = datasets.make((config['train_dataset']))
    test_dataset = datasets.make((config['test_dataset']))

    model = models.make(config['model'], args=None).cuda()
    log(f'model #params: {utils.compute_num_params(model)}')

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.DataParallel(model)

    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, config['batch_size'],
                             num_workers=8, pin_memory=True)

    # Ready for training
    max_epoch = config['max_epoch']
    n_milestones = config.get('n_milestones', 1)
    milestone_epoch = max_epoch // n_milestones
    min_test_loss = 1e18

    sample_batch_train = sample_data_batch(train_dataset).cuda()
    sample_batch_test = sample_data_batch(test_dataset).cuda()

    epoch_timer = utils.EpochTimer(max_epoch)
    for epoch in range(1, max_epoch + 1):
        log_text = f'epoch {epoch}'

        # Train
        model.train()

        adjust_lr(optimizer, epoch, max_epoch, config)
        log_temp_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        ave_scalars = {k: utils.Averager() for k in ['loss']}

        pbar = tqdm(train_loader, desc='train', leave=False)
        for data in pbar:
            data = data.cuda()
            t = train_step(model, data, data, optimizer)
            for k, v in t.items():
                ave_scalars[k].add(v, len(data))
            pbar.set_description(desc=f"train loss:{t['loss']:.4f}")

        log_text += ', train:'
        for k, v in ave_scalars.items():
            v = v.item()
            log_text += f' {k}={v:.4f}'
            log_temp_scalar('train/' + k, v, epoch)

        # Test
        model.eval()

        ave_scalars = {k: utils.Averager() for k in ['loss']}

        pbar = tqdm(test_loader, desc='test', leave=False)
        for data in pbar:
            data = data.cuda()
            t = eval_step(model, data, data)
            for k, v in t.items():
                ave_scalars[k].add(v, len(data))
            pbar.set_description(desc=f"test loss:{t['loss']:.4f}")

        log_text += ', test:'
        for k, v in ave_scalars.items():
            v = v.item()
            log_text += f' {k}={v:.4f}'
            log_temp_scalar('test/' + k, v, epoch)

        test_loss = ave_scalars['loss'].item()

        if epoch % milestone_epoch == 0:
            with torch.no_grad():
                pred = model(sample_batch_train).clamp(0, 1)
                video_batch = torch.cat([sample_batch_train, pred], dim=0)
                log_temp_videos('train/videos', video_batch, epoch)
                img_batch = video_batch[:, :, 3, :, :]
                log_temp_images('train/images', img_batch, epoch)

                pred = model(sample_batch_test).clamp(0, 1)
                video_batch = torch.cat([sample_batch_test, pred], dim=0)
                log_temp_videos('test/videos', video_batch, epoch)
                img_batch = video_batch[:, :, 3, :, :]
                log_temp_images('test/images', img_batch, epoch)

        # Summary and save
        log_text += ', {} {}/{}'.format(*epoch_timer.step())
        log(log_text)

        model_ = model.module if n_gpus > 1 else model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        pth_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
        }

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            wandb.run.summary['min_test_loss'] = min_test_loss
            torch.save(pth_file, osp.join(save_dir, 'min-test-loss.pth'))

        torch.save(pth_file, osp.join(save_dir, 'epoch-last.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--load-root', default='/data/cyb/data')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', '-g', default='0')
    parser.add_argument('--wandb-upload', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def translate_load_root_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_load_root_(v)
            elif isinstance(v, str):
                d[k] = v.replace('${load_root}', args.load_root)
    translate_load_root_(config)

    if args.name is None:
        exp_name = '_' + osp.basename(args.config).split('.')[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag
    config['exp_name'] = exp_name
    save_dir = osp.join(args.save_root, exp_name)
    config['save_dir'] = save_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config['wandb_upload'] = args.wandb_upload

    main(config)
