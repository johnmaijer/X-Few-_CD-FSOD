


# Methods
## Setup
An anaconda environment is suggested, take the name "cdfsod" as an example: 

```
git clone git@github.com:lovelyqian/CDFSOD-benchmark.git
conda create -n cdfsod python=3.9
conda activate cdfsod
pip install -r CDFSOD-benchmark/requirements.txt 
pip install -e ./CDFSOD-benchmark
cd CDFSOD-benchmark
```

## Run 
1. download weights:
  download pretrained model from [DE-ViT](https://github.com/mlzxy/devit/blob/main/Downloads.md).

2. download datasets：dataset1、dataset2、dataset3 . Put them in ./CDFSOD-benchmark/datasets/

3. run script: 
```
bash main_results.sh
```
