
import argparse
import re
from collections import Counter
from pathlib import Path
import torch
from model.model import MiTB5


def convert(source, target):
    result = {}
    consumed = set()
    rules = Counter()

    def assign(src, dst, mode='copy'):
        if src not in source:
            raise KeyError(f'Missing official key: {src}')
        if dst not in target:
            raise KeyError(f'Missing custom key: {dst}')
        value = source[src]
        if mode == 'conv_to_linear':
            if value.ndim != 4 or value.shape[-2:] != (1, 1):
                raise ValueError(f'Expected 1x1 Conv2d: {src} {tuple(value.shape)}')
            value = value.squeeze(-1).squeeze(-1)
        if tuple(value.shape) != tuple(target[dst].shape):
            raise ValueError(f'Shape mismatch: {src} {tuple(value.shape)} -> {dst} {tuple(target[dst].shape)}')
        if dst in result:
            raise ValueError(f'Duplicate destination: {dst}')
        result[dst] = value.detach().clone()
        consumed.add(src)
        rules[mode] += 1

    for stage, (channels, depth) in enumerate(zip((64, 128, 320, 512), (3, 6, 40, 3))):
        p = f'layers.{stage}'
        t = f'patch_embeds.{stage}'
        for suffix in ('weight', 'bias'):
            assign(f'{p}.0.projection.{suffix}', f'{t}.proj.{suffix}')
            assign(f'{p}.0.norm.{suffix}', f'{t}.norm.{suffix}')
            assign(f'{p}.2.{suffix}', f'norms.{stage}.{suffix}')

        for block in range(depth):
            p = f'layers.{stage}.1.{block}'
            t = f'blocks.{stage}.{block}'
            for suffix in ('weight', 'bias'):
                assign(f'{p}.norm1.{suffix}', f'{t}.norm1.{suffix}')
                assign(f'{p}.norm2.{suffix}', f'{t}.norm2.{suffix}')
                assign(f'{p}.attn.attn.out_proj.{suffix}', f'{t}.attn.proj.{suffix}')
                if stage < 3:
                    assign(f'{p}.attn.sr.{suffix}', f'{t}.attn.sr.{suffix}')
                    assign(f'{p}.attn.norm.{suffix}', f'{t}.attn.norm.{suffix}')
                assign(f'{p}.ffn.layers.0.{suffix}', f'{t}.mlp.fc1.{suffix}',
                       'conv_to_linear' if suffix == 'weight' else 'copy')
                assign(f'{p}.ffn.layers.1.{suffix}', f'{t}.mlp.dwconv.{suffix}')
                assign(f'{p}.ffn.layers.4.{suffix}', f'{t}.mlp.fc2.{suffix}',
                       'conv_to_linear' if suffix == 'weight' else 'copy')

            for suffix in ('weight', 'bias'):
                src = f'{p}.attn.attn.in_proj_{suffix}'
                if src not in source:
                    raise KeyError(f'Missing official QKV key: {src}')
                packed = source[src]
                if packed.shape[0] != 3 * channels:
                    raise ValueError(f'Unexpected QKV shape: {src} {tuple(packed.shape)}')
                q, kv = packed.split((channels, 2 * channels), dim=0)
                for label, tensor in (('q', q), ('kv', kv)):
                    dst = f'{t}.attn.{label}.{suffix}'
                    if dst not in target or tensor.shape != target[dst].shape:
                        raise ValueError(f'QKV mismatch: {src} -> {dst}')
                    if dst in result:
                        raise ValueError(f'Duplicate destination: {dst}')
                    result[dst] = tensor.detach().clone()
                consumed.add(src)
                rules['split_qkv'] += 1

    missing = sorted(set(target) - set(result))
    extras = sorted(set(source) - consumed)
    if missing:
        raise RuntimeError(f'{len(missing)} custom backbone keys were NOT mapped:\n' + '\n'.join(missing[:30]))
    # This is a backbone-only checkpoint; unrelated source keys are not silently discarded.
    if extras:
        raise RuntimeError(f'{len(extras)} official checkpoint keys were NOT consumed:\n' + '\n'.join(extras[:30]))
    return result, rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('mit_b5_backbone_converted.pth'))
    parser.add_argument('--smoke-test', action='store_true', help='Run a 128x128 forward pass on CPU')
    args = parser.parse_args()

    if not args.pretrained.is_file():
        parser.error(f'Checkpoint not found: {args.pretrained}')
    obj = torch.load(args.pretrained, map_location='cpu', weights_only=True)
    if isinstance(obj, dict) and 'state_dict' in obj:
        obj = obj['state_dict']
    if not isinstance(obj, dict):
        raise TypeError('Checkpoint is not a state_dict')
    # Support optional wrappers, but never remove arbitrary names.
    source = {}
    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Non-tensor checkpoint entry: {key}')
        clean = key
        for prefix in ('module.', 'backbone.'):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        if clean in source:
            raise RuntimeError(f'Duplicate checkpoint key after prefix removal: {clean}')
        source[clean] = value

    model = MiTB5()
    target = model.state_dict()
    converted, rules = convert(source, target)
    model.load_state_dict(converted, strict=True)
    print('Official tensors:', len(source))
    print('Converted tensors:', len(converted))
    print('Custom backbone tensors:', len(target))
    print('Conversion operations:', dict(rules))
    print('Missing / unexpected: 0 / 0')
    print('LayerNorm eps:', sorted({m.eps for m in model.modules() if isinstance(m, torch.nn.LayerNorm)}))
    if args.smoke_test:
        model.eval()
        with torch.inference_mode():
            outs = model(torch.randn(1, 3, 128, 128))
        print('Output shapes:', [tuple(o.shape) for o in outs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print('Saved:', args.output.resolve())
    print('Use: model.backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)')


if __name__ == '__main__':
    main()
