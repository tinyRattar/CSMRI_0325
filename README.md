# CSMRI_0325
This is the offical code for the paper 
**Cascaded Dilated Dense Network with Two-step Data Consistency for MRI Reconstruction** published in NeruIPS 2019.

The link of paper will be available later.

# Requirements
- Python==3.6.5
- numpy==1.14.3
- opencv-python==3.4.1.15
- scipy==1.1.0
- pytorch==1.0.1.post2
- matplotlib==2.2.2

# How to Train
1. Prepare data.
2. Create an Initialization File in `config/`, named like `CONFIGNAME.ini`. You can also make a copy of `default.ini` and edit it.
3. run `python main.py CONFIGNAME`. Notice that `config/` and `.ini` will be added automatically.

More details can be found below.

# Prepare Data
The original data is established from the work Alexander et al. Details can be found in the paper. 
You can download the original data from [Here](http://www.cse.yorku.ca/~mridataset/).

We convert the data into `.png` format. The png-format can be found from [Here](https://github.com/tinyRattar/CDDNwithTDC_storage/tree/master/data/pngFormat), and the convert code is [Here](https://github.com/tinyRattar/CDDNwithTDC_storage/blob/master/data/saveAsPng.m).

Although there are 4480 frames, we only use 3300 frames(100 frames/patient). Preparing for training, you should:
1. Download the png-format data.
2. Put the data in `./data/cardiac_ktz/`.

In our training process, we pre-generate a quantity of random sampling masks in the `mask/`, named like `mask_rAMOUT_SAMPLINGRATE.mat`. These masks will be applied in the constructor of dataset. 

# How to Conifg
## General
1. NetType: The network for MRI reconstruction. All the options can be found in the function [`getNet`](https://github.com/tinyRattar/CSMRI_0325/blob/b5a8cec01b98a2be0c313dfe403488582c7fced2/network/__init__.py#L31). `CN_dOri_c5_complex_tr_trick2` is the proposed method.
2. UseCuda: use `True` for cuda
3. NeedParallel: use `True` if you want to train with multi gpu devices. We recommend to choose `True` even if only one devices is available.
4. Device: 1 for use and 0 for not. E.g., you want to use the 2nd and the 3rd gpu devices , you should write `0110` here. (more or less devices is acceptable)
5. LossType: The loss function. `mse` or `mae`. (Actually we found no difference in this work)
6. DataType: The part is implemented by "Keyword detection". Check the function [`getDataloader`](https://github.com/tinyRattar/CSMRI_0325/blob/b5a8cec01b98a2be0c313dfe403488582c7fced2/dataProcess.py#L31) for details. `1in1_complex_random` is the default choice in the paper.
7. CrossValid: It is only used for cross-valid. Fill in a integer in [0, 10]. Notice it is not available for fastMRI dataset.
8. Mode: Abandoned. Only `inNetDC` is acceptable here.
9. Path: The saving path for the record and the trained weights.

## Train
1. BatchSize: Batch-size.
2. LearningRate: Learning rate.
3. Epoch: Epoch. What am I doing.
4. Optimizer: Check [`getOptimizer`](https://github.com/tinyRattar/CSMRI_0325/blob/b5a8cec01b98a2be0c313dfe403488582c7fced2/network/__init__.py#L15).
5. WeightDecay: It only work if you use `Adam_wd` in Optimizer above. Remember `Adam_DC_DCNN` and `Adam_RDN` will use the pre-defined weight decay.

## Log
1. SaveEpoch: The result will be logged and saved per `SaveEpoch` epoches.
2. MaxSaved: Only last `MaxSaved` weights will be reversed. Earlier ones will be removed automatically.

## Check
Actucally, we implemented this part long time ago for training with trained record but never use it. So we DON'T promise it can work now.

Use the function [`loadCkpt`](https://github.com/tinyRattar/CSMRI_0325/blob/b5a8cec01b98a2be0c313dfe403488582c7fced2/core.py#L196) instead if you want to load the record. 
For example:
```python
c1 = core_ver2.core('PATH_TO_RESULT/config.ini',True)
c1.loadCkpt(1000, True)
```

Notice that the final result will be saved permanently with additional `CHECKED_` prefix, so set `True` in the second parameter of loadCkpt().
