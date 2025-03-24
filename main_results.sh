datalist=(
dataset1
dataset2
dataset3
)
shot_list=(
1
5
10
)
model_list=(
"l"
#"b"
#"s"
)
for model in "${model_list[@]}"; do
  for dataset in "${datalist[@]}"; do
    for shot in "${shot_list[@]}"; do
      python tools/train_net.py --num-gpus 2 --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml MODEL.WEIGHTS weights/trained/few-shot/vit${model}_0089999.pth DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR output/vit${model}/${dataset}_${shot}shot/
    done
  done
done
