# SHREC23-ANIMAR-BASELINE

A baseline for the SHREC23 ANIMAR challenge. See the [challenge website](https://aichallenge.hcmus.edu.vn/) for more information.
We already provide the baseline for [preliminary dataset](https://drive.google.com/file/d/1BYDCPkOa-s30aFUAr0h80N-t-Q7kDupQ/view). Other teams can use this baseline to test their methods on the official dataset.

## Download raw dataset

Download and extract `dataset.zip`. The example uses `gdown` to download the file from Google Drive. You can install it with `pip install gdown`. 

```bash
$ cd data/
$ gdown 1BYDCPkOa-s30aFUAr0h80N-t-Q7kDupQ
$ unzip ANIMAR_Preliminary_Data.zip 
```

The resulting directory tree should look like this:

```
./
├─ data/
│  ├─ ANIMAR_Preliminary_Data/
│  │  ├─ 3D_Models/
│  │  ├─ Sketch_Query/
│  │  ├─ Text_Query.xlsx
├─ ...
```

We provide the `Text_Query.xlsx` file to help you understand the dataset. The `Sketch_Query` directory contains the sketches of the queries. The `3D_Models` directory contains the 3D models of the reference obj models. 

Create your own train/test split, we provide a simple example in `data/csv/train_preliminary.csv` and `data/test_preliminary.csv`. 
The csv files contain the names of the queries and the reference models. 
```csv
id,obj_filename,sket_filename,tex
```

[Update] We provide the prepared notebook in `data/preparation.ipynb` to help you prepare the dataset. 


```
./
├─ data/
│  ├─ ANIMAR_Preliminary_Data/
│  │  ├─ 3D_Models/
│  │  │  ├─ References/
│  │  ├─ Text_Query.xlsx
|
│  ├─ TextANIMAR2023/
│  │  ├─ 3D_Model_References/
│  │  │  ├─ References/
│  │  ├─ Train/
│  │  │  ├─ *GT_Train.csv
│  │  │  ├─ *Train.csv
|
│  ├─ SketchANIMAR2023/
│  │  ├─ 3D_Model_References/
│  │  │  ├─ References/
│  │  ├─ Train/
│  │  │  ├─ SketchQuery_Train/
│  │  │  ├─ *GT_Train.csv
│  │  │  ├─ *Train.csv
├─ ...
```

## Capture ring views images

We provide a script to capture the ring views images. The script uses Blender to render the images.
This technique is described in [paper](https://diglib.eg.org/handle/10.2312/3dor20201163) to represent 3D models as images.

1. Download the blender tar file, version 2.79 (exclusive). Untar and check the `blender` executive inside.

```bash
$ wget https://download.blender.org/release/Blender2.79/blender-2.79-linux-glibc219-x86_64.tar.bz2
$ tar xjf blender-2.79-linux-glibc219-x86_64.tar.bz2
$ mv blender-2.79-linux-glibc219-x86_64 blender-2.79
```
2. Generate render images 

```bash
blender-2.79/blender -b -P generate_ring.py -- data/ANIMAR_Preliminary_Data/3D_Models
```
It will create a checkpoint file `save.txt` as it runs, this file stores the index of the last generated object to resume in case of errors occuring. This file needs to be deleted between generation of each phase (else it will keep resuming).

A directory `generated_models` is created next to the `3D_Models` directory. It contains the generated images for each ring view. The directory tree should look like this:

```
data/
├─ ANIMAR_Preliminary_Data/
│  ├─ 3D_Models/
│  │  ├─ *.obj
│  ├─ generated_models/
│  │  ├─ ring#                      
│  │  │  ├─ <type>/                 
│  │  │  │  ├─ Image####.png
```

where
- `rid`: a number (default: 0-6) as ring identifier
- `type`: `depth`, `mask`, or `render`
- `####`: a number (with leading 0s, default: 0001-0012) as view identifier

## Training

### Install dependencies
Before running the baseline, you may need to install the dependencies. We recommend using a virtual environment. 

Install conda/mamba according to the instructions on the homepage
Before installing the repo, we need to install the CUDA driver version >=11.6

```bash
$ conda env create -f animar.yml
$ conda activate animar
```

We provide 2 baselines corresponding to the 2 tasks of the challenge.
The first baseline is for the sketch query task and second one is for the text query task. For more details, see the implemented [models](models).

To train the baseline, run the following command:

```bash
$ python train_sketch_query.py
$ python train_prompt_query.py
```


