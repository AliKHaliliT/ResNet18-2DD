# ResNet18-2DD
<div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px;">
    <img src="https://img.shields.io/github/license/AliKHaliliT/ResNet18-2DD" alt="License">
    <img src="https://github.com/AliKHaliliT/ResNet18-2DD/actions/workflows/tests.yml/badge.svg" alt="tests">
    <img src="https://img.shields.io/github/last-commit/AliKHaliliT/ResNet18-2DD" alt="Last Commit">
    <img src="https://img.shields.io/github/issues/AliKHaliliT/ResNet18-2DD" alt="Open Issues">
</div>
<br/>

A fully serializable 2D implementation of ResNet18, incorporating improvements from the paper ["Bag of Tricks for Image Classification with Convolutional Neural Networks"](https://arxiv.org/abs/1812.01187) along with additional personal optimizations and modifications.

This repository also includes implementations of the Hardswish and Mish activation functions:

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)

The codebase is fully integratable inside the TensorFlow and Keras code pipelines.

## Key Enhancements
- **Modified Stem:** Utilizes two convolutional layers instead of a single one.
- **ResNet-B Inspired Strides:** Moved the stride placement in the residual blocks from the first convolution to the second.
- **ResNet-D Inspired Shortcut:** Introduces an average pooling layer before the 1x1 convolution in the shortcut connection.
- **Reduced Downsampling:** Downsampling is now performed only twice (in the stem block) instead of the original five times.

<br/>
<br/>
<div align="center" style="display: flex; justify-content: center; align-items: center;">
    <img src="util_resources/readme/resnet_c.png" alt="ResNet-C image from the paper" style="width:300px; height:auto; margin-right: 16px;">
    <img src="util_resources/readme/shortcut.png" alt="Shortcut image by author" style="width:350px; height:auto;">
</div>
<br/>

*Note: The ResNet-C image is sourced from the referenced paper, while the shortcut image is created by the author.*

## Installation & Usage
This code is compatible with **Python 3.12.8** and **TensorFlow 2.18.0**.

```python
from ResNet182DD import ResNet182DD


model = ResNet182DD()
model.build((None, 256, 256, 3))
model.summary()
```

### Model Summary Example
```bash
Model: "res_net182dd"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_layer (Conv2DLayer)           │ (None, 128, 128, 32)        │             864 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_layer_1 (Conv2DLayer)         │ (None, 128, 128, 32)        │           9,216 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_layer_2 (Conv2DLayer)         │ (None, 128, 128, 64)        │          18,432 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 64, 64, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual2dd (Residual2DD)            │ (None, 64, 64, 64)          │          73,728 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual2dd_1 (Residual2DD)          │ (None, 32, 32, 128)         │         229,376 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual2dd_2 (Residual2DD)          │ (None, 32, 32, 128)         │         294,912 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual2dd_3 (Residual2DD)          │ (None, 16, 16, 256)         │         917,504 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual2dd_4 (Residual2DD)          │ (None, 16, 16, 256)         │       1,179,648 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual2dd_5 (Residual2DD)          │ (None, 8, 8, 512)           │       3,670,016 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ residual2dd_6 (Residual2DD)          │ (None, 8, 8, 512)           │       4,718,592 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 512)                 │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │         131,328 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 11,243,616 (42.89 MB)
 Trainable params: 11,243,616 (42.89 MB)
 Non-trainable params: 0 (0.00 B)
```

## License
This work is under an [MIT](https://choosealicense.com/licenses/mit/) License.