# Training

Scripts for training semantic segmentation models.

## Official Pretrained Weights

**Source:** OpenMMLab MMSegmentation

**Download:** [mit_b5_20220624-658746d9.pth](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth)

## Convert Pretrained Weights

The official checkpoint must be converted to match the custom backbone implementation.

```bash
python convert_pretrained_pth.py --pretrained mit_b5_20220624-658746d9.pth --output backbone_converted.pth
```

The converted checkpoint can be loaded into the backbone using `strict=True`.

## Model Configuration

| Parameter | Value |
|---|---|
| Backbone | MiT-B5 |
| Decoder | SegFormer Head |
| Decoder channels | 256 |
| Number of classes | 2 |
| Input resolution | 640 × 640 |
| Batch size | 8 |
| Training iterations | 20,000 |
| Optimizer | AdamW |
| Backbone learning rate | 6e-5 |
| Decoder learning rate | 6e-4 |
| Warmup iterations | 1,500 |

## Training

```bash
python train.py --data_dir /path/to/dataset --out_dir /path/to/output --pretrained backbone_converted.pth --batch_size 8
```

Both training and validation use random foreground-aware crops containing mask pixels labeled `1`. Only training applies random horizontal and vertical flips.
  
