# grad-cam.tensorflow
Implementation of Grad CAM in tensorflow

Gradient class activation maps are a visualization technique for deep learning networks.

The original paper: https://arxiv.org/pdf/1610.02391v1.pdf

The original torch implementation: https://github.com/ramprs/grad-cam

## Setup

Clone the repository
```sh
git clone https://github.com/Ankush96/grad-cam.tensorflow/
```
Download the VGG16 weights from https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz

## Usage

```sh
python main.py --input laska.png --output laska_save.png --layer_name pool5
```

## Results

| Input | Output |
| ------|-----:|
| ![Original image][inp] | ![Original image + Visualization][out] |

[inp]: https://github.com/Ankush96/grad-cam.tensorflow/blob/master/laska.png
[out]: https://github.com/Ankush96/grad-cam.tensorflow/blob/master/laska_save.png

## Acknowledgement

Model weights (vgg16_weights.npz), TensorFlow model of vgg (vgg16.py), Class names (imagenet_classes.py) and example input (laska.png) were either copied or modified from this blog by Davi Frossard (https://www.cs.toronto.edu/~frossard/post/vgg16/)



