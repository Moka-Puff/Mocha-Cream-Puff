import argparse
import csv
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model.model import SegFormer

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_arguments():
    p = argparse.ArgumentParser(description='Training script')
    p.add_argument('--data_dir', required=True, type=Path)
    p.add_argument('--out_dir', required=True, type=Path)
    p.add_argument('--max_iters', type=int, default=20000)
    p.add_argument('--save_every', type=int, default=2000)
    p.add_argument('--val_every', type=int, default=2000)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=6e-5)
    p.add_argument('--head_lr_mult', type=float, default=10.0)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_iters', type=int, default=1500)
    p.add_argument('--warmup_start_factor', type=float, default=1e-6)
    p.add_argument('--poly_power', type=float, default=1.0)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pretrained', required=True, type=Path, help='Converted backbone state_dict')
    p.add_argument('--crop_min', type=int, default=512)
    p.add_argument('--crop_max', type=int, default=2048)
    p.add_argument('--image_size', type=int, default=640)
    return p.parse_args()


class VOCCropDataset(Dataset):
    """Random foreground-aware crops for both training and validation.

    Training: random crop + independent horizontal/vertical flips (p=0.5).
    Validation: random crop only (no flips).
    Every returned resized mask is guaranteed to contain at least one pixel == 1.
    """
    def __init__(self, pairs, training, crop_min, crop_max, image_size):
        self.pairs = pairs
        self.training = training
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        with Image.open(mask_path) as m:
            mask = m.copy()
        if image.size != mask.size:
            raise ValueError(f'Image/mask size mismatch: {image_path} vs {mask_path}')

        w, h = image.size
        max_crop = min(self.crop_max, w, h)
        if max_crop < self.crop_min:
            raise ValueError(
                f'Image smaller than minimum crop {self.crop_min}: {image_path} ({w}x{h})'
            )

        mask_full = np.asarray(mask)
        values = np.unique(mask_full)
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f'Mask must contain only 0/1, found {values.tolist()} in {mask_path}')

        foreground_y, foreground_x = np.where(mask_full == 1)
        if len(foreground_x) == 0:
            raise ValueError(f'Mask has no foreground (value 1): {mask_path}')

        # Choose a foreground pixel first, then choose a crop containing it.
        # Retry because downsampling with nearest interpolation can still
        # eliminate an extremely thin foreground region.
        for _ in range(30):
            side = random.randint(self.crop_min, max_crop)
            k = random.randrange(len(foreground_x))
            fx, fy = int(foreground_x[k]), int(foreground_y[k])

            # Put the selected foreground pixel in the inner 50% of the crop
            # whenever image boundaries permit.
            x_low = max(0, fx - 3 * side // 4, fx - side + 1)
            x_high = min(w - side, fx - side // 4, fx)
            y_low = max(0, fy - 3 * side // 4, fy - side + 1)
            y_high = min(h - side, fy - side // 4, fy)

            # Near an image boundary, use any legal crop containing the pixel.
            if x_low > x_high:
                x_low, x_high = max(0, fx - side + 1), min(fx, w - side)
            if y_low > y_high:
                y_low, y_high = max(0, fy - side + 1), min(fy, h - side)

            x = random.randint(x_low, x_high)
            y = random.randint(y_low, y_high)
            box = (x, y, x + side, y + side)
            mask_crop = mask.crop(box).resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )
            mask_np = np.asarray(mask_crop, dtype=np.int64)
            if np.any(mask_np == 1):
                image_crop = image.crop(box).resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                break
        else:
            raise RuntimeError(
                f'Failed to retain foreground after 30 random crops: {mask_path}. '
                'Check whether the root is too thin for the requested output resolution.'
            )

        image_np = np.asarray(image_crop, dtype=np.float32) / 255.0

        if self.training:
            if random.random() < 0.5:
                image_np = np.flip(image_np, axis=1)
                mask_np = np.flip(mask_np, axis=1)
            if random.random() < 0.5:
                image_np = np.flip(image_np, axis=0)
                mask_np = np.flip(mask_np, axis=0)

        image_np = (image_np - MEAN) / STD
        return (
            torch.from_numpy(image_np.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask_np.copy()),
        )


def find_pairs(data_dir):
    images = data_dir / 'JPEGImages'
    masks = data_dir / 'SegmentationClass'
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f'Expected JPEGImages/ and SegmentationClass/ in {data_dir}')
    paths = sorted(images.glob('*.jpg'), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem))
    if not paths:
        raise ValueError('No JPG images found')
    pairs = []
    for image in paths:
        mask = masks / (image.stem + '.png')
        if not mask.is_file():
            raise FileNotFoundError(f'Missing mask for {image.name}: {mask}')
        pairs.append((image, mask))
    if len(list(masks.glob('*.png'))) != len(pairs):
        raise ValueError('Mask count differs from JPG count')
    return pairs


def make_optimizer(model, lr, weight_decay, head_lr_mult):
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    norm_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for param_name, _ in module.named_parameters(recurse=False):
                norm_names.add(f'{module_name}.{param_name}' if module_name else param_name)

    grouped = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = 'head' in name
        no_decay = name in norm_names or 'pos_block' in name or 'norm' in name
        # Matches the official custom_keys for head/norm/pos_block.
        key = (is_head, no_decay)
        grouped.setdefault(key, []).append(param)

    groups = []
    for (is_head, no_decay), params in grouped.items():
        group_lr = lr * (head_lr_mult if is_head else 1.0)
        groups.append(dict(params=params, lr=group_lr, initial_lr=group_lr,
                           weight_decay=0.0 if no_decay else weight_decay,
                           group_name=('head' if is_head else 'backbone') + ('_no_decay' if no_decay else '_decay')))
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def lr_factor(step, max_iters, warmup_iters, start_factor, power):
    # step is the 0-based index of the next optimizer update.
    if step < warmup_iters:
        return start_factor + (1.0 - start_factor) * step / warmup_iters
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    return max(0.0, 1.0 - progress) ** power


def apply_lr(optimizer, step, args):
    factor = lr_factor(step, args.max_iters, args.warmup_iters,
                       args.warmup_start_factor, args.poly_power)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * factor


def batch_metrics(logits, masks):
    pred = logits.detach().argmax(dim=1)
    tp = ((pred == 1) & (masks == 1)).sum().item()
    fp = ((pred == 1) & (masks == 0)).sum().item()
    fn = ((pred == 0) & (masks == 1)).sum().item()
    return tp, fp, fn


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, tp, fp, fn = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            total_loss += loss.item() * images.shape[0]
            a, b, c = batch_metrics(logits, masks)
            tp += a
            fp += b
            fn += c
    denom = tp + fp + fn
    iou = tp / denom if denom else 1.0
    dice_denom = 2 * tp + fp + fn
    dice = 2 * tp / dice_denom if dice_denom else 1.0
    return total_loss / len(loader.dataset), iou, dice


def main():
    args = parse_arguments()
    if not (0 < args.val_ratio < 1):
        raise ValueError('val_ratio must be between 0 and 1')
    if not (0 < args.crop_min <= args.crop_max):
        raise ValueError('Invalid crop bounds')
    if args.max_iters <= 0 or args.save_every <= 0 or args.val_every <= 0 or args.log_every <= 0:
        raise ValueError('Iteration counts must be positive')
    if not (0 <= args.warmup_iters < args.max_iters):
        raise ValueError('warmup_iters must be >= 0 and less than max_iters')
    if not (0 < args.warmup_start_factor <= 1):
        raise ValueError('warmup_start_factor must be in (0, 1]')
    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is False')

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(args.out_dir / 'train.log', encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)], force=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    pairs = find_pairs(args.data_dir)
    if len(pairs) < 2:
        raise ValueError('Need at least two image/mask pairs')
    random.shuffle(pairs)
    n_val = max(1, min(len(pairs) - 1, round(len(pairs) * args.val_ratio)))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    (args.out_dir / 'split.txt').write_text(
        'train:\n' + '\n'.join(p[0].name for p in train_pairs)
        + '\nval:\n' + '\n'.join(p[0].name for p in val_pairs), encoding='utf-8')

    train_ds = VOCCropDataset(train_pairs, True, args.crop_min, args.crop_max, args.image_size)
    val_ds = VOCCropDataset(val_pairs, False, args.crop_min, args.crop_max, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    model = SegFormer(num_classes=2)
    checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=True)
    model.backbone.load_state_dict(checkpoint, strict=True)
    logging.info('Loaded pretrained backbone: %s', args.pretrained)
    model = model.to(device)

    optimizer = make_optimizer(model, args.lr, args.weight_decay, args.head_lr_mult)
    criterion = nn.CrossEntropyLoss()
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    logging.info('train=%d val=%d device=%s batch_size=%d max_iters=%d',
                 len(train_ds), len(val_ds), device, args.batch_size, args.max_iters)
    logging.info('approx epochs=%.2f; warmup=%d; checkpoint every %d iterations',
                 args.max_iters / len(train_loader), args.warmup_iters, args.save_every)
    for group in optimizer.param_groups:
        logging.info('Optimizer group %s: tensors=%d base_lr=%.8g weight_decay=%g',
                     group['group_name'], len(group['params']), group['initial_lr'], group['weight_decay'])

    csv_path = args.out_dir / 'metrics.csv'
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_tp = running_fp = running_fn = 0
    running_samples = 0
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'epoch_approx', 'backbone_lr', 'head_lr',
                         'train_loss', 'train_iou', 'train_dice',
                         'val_loss', 'val_iou', 'val_dice'])
        for iteration in range(1, args.max_iters + 1):
            model.train()
            try:
                images, masks = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, masks = next(train_iter)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            apply_lr(optimizer, iteration - 1, args)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.shape[0]
            running_samples += images.shape[0]
            a, b, c = batch_metrics(logits, masks)
            running_tp += a
            running_fp += b
            running_fn += c

            do_log = iteration % args.log_every == 0 or iteration == args.max_iters
            do_val = iteration % args.val_every == 0 or iteration == args.max_iters
            if do_log or do_val:
                train_loss = running_loss / running_samples
                train_iou = (running_tp / (running_tp + running_fp + running_fn)
                             if running_tp + running_fp + running_fn else 1.0)
                train_dice = (2 * running_tp / (2 * running_tp + running_fp + running_fn)
                              if 2 * running_tp + running_fp + running_fn else 1.0)
                val_metrics = evaluate(model, val_loader, criterion, device, use_amp) if do_val else (float('nan'),) * 3
                backbone_lr = next(g['lr'] for g in optimizer.param_groups if g['group_name'].startswith('backbone'))
                head_lr = next(g['lr'] for g in optimizer.param_groups if g['group_name'].startswith('head'))
                writer.writerow([iteration, iteration / len(train_loader), backbone_lr, head_lr,
                                 train_loss, train_iou, train_dice, *val_metrics])
                f.flush()
                logging.info('iter %d/%d epoch≈%.1f lr=%.3g head_lr=%.3g train_loss=%.5f train_iou=%.4f train_dice=%.4f val_loss=%.5f val_iou=%.4f val_dice=%.4f',
                             iteration, args.max_iters, iteration / len(train_loader), backbone_lr, head_lr,
                             train_loss, train_iou, train_dice, *val_metrics)
                running_loss = 0.0
                running_tp = running_fp = running_fn = running_samples = 0

            if iteration % args.save_every == 0:
                path = args.out_dir / f'iter_{iteration:06d}.pth'
                torch.save(model.state_dict(), path)
                logging.info('Saved model: %s', path)

    final_path = args.out_dir / 'final.pth'
    torch.save(model.state_dict(), final_path)
    logging.info('Saved final model: %s', final_path)


if __name__ == '__main__':
    main()
