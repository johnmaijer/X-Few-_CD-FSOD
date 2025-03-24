#!/bin/bash
if [ ! -d "cache_models" ]; then
  mkdir "cache_models"
  echo "cache_models folder created."
else
  echo "cache_models folder already exists."
fi
data_list=(
# "dataset1"
"dataset2"
"dataset3"
# "ArTaxOr"
# "clipart1k"
# "DIOR"
# "UODD"
# "NEUDET"
# "FISH"
)
shot_list=(
1
5
# 10
)
model_list=(
"l"
)

for dataset in "${data_list[@]}"; do
    for shot in ${shot_list[@]}; do
        python ./tools/build_cache_model.py   --dataset ${dataset}_${shot}shot --out_dir cache_models
        echo "buils_cache_models with vit${model} for ${shot}shot ${dataset} done, save at cache_models dir."
    done
done
