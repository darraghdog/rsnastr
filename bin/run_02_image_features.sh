for FOLD in 0 1 2 3
do
    python training/pipeline/train_image_mask_classifier.py --device 'cuda' \
            --fold $FOLD --batchsize 40 --logdir 'logs/zoo' --augextra False \
            --label-smoothing 0.0 --config 'configs/effnetb5_lr5e4_multi.json'
done