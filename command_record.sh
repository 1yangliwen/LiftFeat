python tools/demo_match_video.py --img data/ill_template.png --video data/ill_change.mp4

python train.py --megadepth_root_path data --coco_root_path data/coco_20k --ckpt_save_path ckpts

python train.py \
    --name LiftFeat_train \
    --megadepth_root_path data \
    --use_megadepth \
    --coco_root_path data/coco_20k \
    --use_coco \
    --ckpt_save_path ckpts

python evaluation/MegaDepth1500_evaluation.py --gpu 6

python demo.py --img1=assert/query.jpg --img2=assert/ref.jpg