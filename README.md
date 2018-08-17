# tf-pvanet
Implementation of [PVANet](https://arxiv.org/abs/1611.08588) in Tensorflow.
* The backbone of the network was implemented by Deng Dan at [tf_pvanet](https://github.com/dengdan/tf-pvanet/blob/master/pvanet.py). Changes are made to make the network accommodate with CIFAR-10 dataset better. 

## Pre-requisite
* The `util` module come from [pylib](https://github.com/dengdan/pylib). Add the path of its `src` directory to `PYTHONPATH` when using.
* Please fetch the CIFAR-10 dataset from [official website](https://www.cs.toronto.edu/~kriz/cifar.html) and put the data in a folder named `cifar10` at the same level as `scripts` before running the model, as the following structure shows:
```
$ls
cifar10  scripts  pvanet.py  train_pvanet.py ...
```
```
$cd cifar10
$ls
data_batch_1.bin  data_batch_2.bin  data_batch_3.bin  data_batch_4.bin
data_batch_5.bin  test_batch.bin
```

## Network Modification
Several modifications are made to the original network:
* The scale layer is removed. Scale and shift is handled by batch_norm layer, where both `scale` and `center` arguments are set to True;
* The spatial size shrinkage is modified. Originally, `[conv1, pool1, conv2, conv3, conv4, conv5]` all use stride 2 at the starting convolution layer which shrinks input images by 64 (2^6) times. However, CIFAR-10 images is 32x32 by default and the shrinkage is not valid. Thus, the strides from these layers are adjusted to `[1, 2, 1, 2, 2, 2]`;
* The first convolution layer uses kernel size of 5 instead of 7;
* For a residual block (CReLU or Inception) where number of input filters and output filters are different, instead of using 1x1 convolution for projection in the paper, 0-padding is used as the default method;
* Concatenation is now added after conv5 using deconvolution.

## Network Configurations
The model trained with default settings should reach a best test accuracy of 91.1% on CIFAR-10 test dataset. The default setting is:
* Fatness is 2, adjustable in `config.py`
* Use concatenation at the end of conv5
* Use 0 padding for projection in residual blocks (CReLU and inception)
* Decay in slim.batch_norm is 0.9
* Learning rate starts at 0.1, and decay by a factor of 10 for every 40,000 steps. The minimum learning rate is 1e-4
* Use momentum optimizer with rate 0.9
* The weight decay rates on model weights is 0.0002

## How to Use
* Download CIFAR-10 data and put them in the right location as instructed.
* Adjust parameters in `config.py` and `./scripts/train_pvanet.sh` as desired.
* Train the model:
```
sh ./scripts/train_pvanet.sh ${GPU} ${MODEL_DIR}
```
* Evaluate the model:
```
sh ./scripts/test_pvanet.sh ${GPU} ${MODEL_DIR}
```
