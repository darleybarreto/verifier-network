# Generic Discriminator for pairs of images

Code based on [Facial Similarity with Siamese Networks in Pytorch](https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7) repository. You can find the related article [here](https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e).

# Pytorch Version
This project is updated to be compatible with Pytorch 1.0.1-1.1.0

# How to run
Example 1:
```bash
$ python3 main.py
```

Example 2:
```bash
$ python3 main.py --dataset cifar-10 --n_epochs 5000
```

Example 3:
```bash
$ python3 main.py --dataset mnist --b_size 512
```

Example 4:
```bash
$ python3 main.py --dataset faces --b_size 128 --n_epochs 120
```

# Usage

```
usage: main.py [-h] [--training_dir TRAINING_DIR] [--testing_dir TESTING_DIR]
               [--dataset DATASET] [--b_size B_SIZE] [--n_epochs N_EPOCHS]
               [--resume]

optional arguments:
  -h, --help            show this help message and exit
  --training_dir    TR  folder where data to train is
  --testing_dir     TS  folder where data to test is
  --dataset         D   dataset to run experiments (faces, cifar-10 or mnist)
  --b_size          B   batch size
  --epochs          N   number of total epochs to run, default is 500
  --resume              resume from saved in TRAINING_DIR/model folder