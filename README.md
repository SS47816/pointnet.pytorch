# PointNet.pytorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet/model.py`.

It is tested with pytorch-1.0.

# Download data and running

```
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
```

Download and build visualization tool
```
cd script
bash build.sh #build C++ code for visualization
bash download.sh #download dataset
```

Training 
```
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

Use `--feature_transform` to use feature transform.

# Performance

## Classification performance

On ModelNet40:

|  | Overall Acc |
| :---: | :---: |
| Original implementation | 89.2 |
| this implementation(w/o feature transform) | 86.4 |
| this implementation(w/ feature transform) | 87.0 |

On [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html)

|  | Overall Acc |
| :---: | :---: |
| Original implementation | N/A |
| this implementation(w/o feature transform) | 98.1 |
| this implementation(w/ feature transform) | 97.7 |

## Segmentation performance

Segmentation on  [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html).

| Class(mIOU) | Airplane | Bag| Cap|Car|Chair|Earphone|Guitar|Knife|Lamp|Laptop|Motorbike|Mug|Pistol|Rocket|Skateboard|Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Original implementation |  83.4 | 78.7 | 82.5| 74.9 |89.6| 73.0| 91.5| 85.9| 80.8| 95.3| 65.2| 93.0| 81.2| 57.9| 72.8| 80.6| 
| this implementation(w/o feature transform) | 73.5 | 71.3 | 64.3 | 61.1 | 87.2 | 69.5 | 86.1|81.6| 77.4|92.7|41.3|86.5|78.2|41.2|61.0|81.1|
| this implementation(w/ feature transform) |  |  |  |  | 87.6 |  | | | | | | | | | |81.0|

Note that this implementation trains each class separately, so classes with fewer data will have slightly lower performance than reference implementation.

Sample segmentation result:
![seg](https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/misc/show3d.png?token=AE638Oy51TL2HDCaeCF273X_-Bsy6-E2ks5Y_BUzwA%3D%3D)

# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
- [Tensorflow implementation](https://github.com/charlesq34/pointnet)



---

> by: SS47816




## Modifications

The following changes have been made to train on ModelNet40:
1. Fixed CUDA issue with GPU by adding the following in `train_classification.py`:
    ```python
    import torch
    torch.cuda.current_device()
    ```

2. Added `pointnet.yml` file which contains the conda env I used

3. Added  `off_to_ply.py` to convert ModelNet40 dataset from `.off` files to `.ply` files

4. Added a list of `.txt` files to split the dataset for training/testing under `utils/modelnet40_txts/` directory:

5. In `train_classification.py`, added records for train/test loss & accuracy during the training, and plotted the results

6. Lowered the Adam learning rate `lr` in `train_classification.py` from `1e-3` to `1e-4`



## Train on ModelNet40

### System Setup

* GTX 1070Ti
* Nvidia Driver 455.23.04
* CUDA 10.2



1. Create a new Conda env `pointnet` by:

   ```bash
   conda env create --file pointnet.yml
   
   conda activate pointnet
   ```

   **Note:** Though the original author built the model with Pytorch 1.0, it has been tested working in this Conda env with Pytorch 1.7

2. Download and Unzip the ModelNet40 Dataset to a location (denoted as `<path-to-ModelNet40>`)

3. Convert ModelNet40 Dataset from `.off` to `.ply` file format:

   ```bash
   cd utils
   python3 off_to_ply.py -i <path-to-ModelNet40> -o <path-to-ModelNet40>
   ```

4. Copy the 4 `.txt` files in `utils/modelnet40_txts/` to the `<path-to-ModelNet40>` directory

5. Start training (here are the params I am using):

   ```bash
   python3 train_classification.py --dataset <path-to-ModelNet40> --batchSize 32 --nepoch 300 --dataset_type modelnet40 --feature_transform
   ```

   

