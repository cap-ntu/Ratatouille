# Optimization of data communication for Mxnet

This project is modified from original mxnet.It includes a data filter with threshold equation and Snappy compression for distributed training for Mxnet python.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
	GCC 4.8 or later to compile C++ 11.
	GNU Make

### Installing

A step by step series of examples that tell you have to get a development env running
```
  sudo apt-get update
  sudo apt-get install -y build-essential git
  sudo apt-get install libsnappy-dev
  sudo apt-get install -y libopenblas-dev liblapack-dev
  sudo apt-get install -y libopencv-dev
  git clone https://github.com/cap-ntu/Mxnet.git
  cp make/config.mk ./
  make -j to compile
  cd python 
  sudo python setup.py install
```
Repeat the above installation on multiple machine

## Overall Structure

![alt text](https://github.com/cap-ntu/Mxnet/blob/master/overall%20structure.JPG)


## Threshold Equation

Threshold equation limits the data flow,the key update only occurs when key value is larger than threshold.
As the iteration increases,threshold decreases,ensure the key value converging.
The threshold equation is defined as initial_threshold/t^(1/a) , t and a denote  number of iteration and converging factor

To change initial threshold
```
  sudo nano /etc/environment
  add INIT_THRESHOLD=xx
```


## Result
For distributed training,command is
```
python ../../tools/launch.py --launcher ssh -H hosts prog --kv-store dist_async
```
hosts: the file contains the ip address of nodes




We have tested on image_classification/train_mnist.py and train_cifar10.py

```
python ../../tools/launch.py -n 2 --launcher ssh -H hosts python train_mnist.py --network mlp  --kv-store dist_async
```
| Initial Threshold | Data(GB) |  Sample Rate | Final Accuracy  | Time Cost |
| -------------| ------------- |------------- | -------------   | ----------|
|0.0 (No Compress) |  12.3  | 98  | 0.975 | 621|
|0.0  | 12.1  | 98  | 0.975 | 621 |
|0.2  | 7.58  | 102  | 0.978 | 584 |
|0.4  | 5.01  | 103  | 0.966 | 546 |
|0.6  | 3.02  | 113  | 0.969 | 526 |
|0.8  | 3.20  | 110  | 0.967 | 520 |

![alt text](https://github.com/cap-ntu/Mxnet/blob/master/train_mnist%20with%20different%20threshold.JPG)

Fig. 1 train_mnist with different initial threshold


