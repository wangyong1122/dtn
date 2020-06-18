### Introduction
This repository is the pytorch implementation for the paper: Go From the General to the Particular: Multi-Domain Translation with Domain Transformation Networks.

The project is based on the fairseq. (Please get familar with the fairseq project first)

### Requirements and Installation
* PyTorch version >= 1.0.0
* Python version 3.6

### Usage
First, preprocess your training corpus. Use BPE (byte-pair-encoding) to segment text into subword units and please follow <https://github.com/rsennrich/subword-nmt> for further details. In addition, please place the validation and test sets of respective domains to the directory valid_test_txt, which is included in the directory of binary files. For training and inference, please refer to our paper and scripts for more details.

### Contact
For any questions, please email to the [first author](mailto:wangyong@eee.hku.hk).
