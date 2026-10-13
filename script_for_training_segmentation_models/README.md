Official Pretrained Weights

Source: OpenMMLab MMSegmentation

Download: mit_b5_20220624-658746d9.pth

Convert Pretrained Weights

The official checkpoint must be converted to match the custom backbone implementation.

python convert_pretrained_pth.py \
    --pretrained mit_b5_20220624-658746d9.pth \
    --output backbone_converted.pth

The converted checkpoint can then be loaded into the backbone using strict=True.

Model Configuration

Backbone: MiT-B5

Decoder: SegFormer Head

Decoder channels: 256

Number of classes: 2

Input resolution: 640 × 640

Optimizer: AdamW

Training iterations: 20,000

Batch size: 8

Pretrained backbone: OpenMMLab MiT-B5
