# CVPR Assignment: CNN Development on a Custom Dataset

**Student:** Saadman Sakib Saad | **ID:** 22-49179-3 | **Course:** Computer Vision and Pattern Recognition (CVPR)

A custom CNN ("IntelNet") trained from scratch in PyTorch to classify natural-scene images into six classes: buildings, forest, glacier, mountain, sea, street.

## Dataset
Intel Image Classification (Kaggle: `puneet6060/intel-image-classification`), 150x150 RGB images. About 14,000 labelled training images and 3,000 test images. The test folder is held out until final evaluation, and a stratified 85/15 train/validation split is taken from the training folder.

## Model
4 convolutional blocks (2 x [Conv3x3, BatchNorm, ReLU], MaxPool, Dropout2d) with 32, 64, 128 and 256 channels, then Global Average Pooling and a small classifier head. 1,240,550 parameters (about 5 MB).

## Training
AdamW (lr 1e-3, weight decay 1e-4), CrossEntropyLoss, batch size 64, up to 30 epochs, ReduceLROnPlateau, best checkpoint by validation accuracy, augmentation (random resized crop, horizontal flip, rotation, colour jitter).

## Results (held-out test set)
| Metric | Value |
|---|---|
| Test accuracy | **90.33%** (2710 / 3000) |
| Macro F1 | 0.905 |
| Best validation accuracy | 90.60% (epoch 30) |
| Easiest class | forest (F1 0.985) |
| Hardest class | mountain (F1 0.848) |

The largest confusions are mountain predicted as glacier (73) and buildings predicted as street (43), which are visually similar scene types.

## Files
- `CNN_22-49179-3.ipynb`: full notebook with code, outputs and visualisations
- `intelnet_best.pth`: trained model weights (state_dict, about 5 MB)

## Load the model
```python
model = IntelNet(num_classes=6)
model.load_state_dict(torch.load("intelnet_best.pth", map_location="cpu", weights_only=True))
model.eval()
```
