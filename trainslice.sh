WINDOW=1
for FOLD in 0 3 4 # 1 2 3 # 0
do
    python training/pipeline/train_image_slice_classifier.py --device 'cuda' --fold $FOLD --batchsize 32 --accum 4 \
            --window $WINDOW --logdir 'logs/zoo' --flip True --augextra False --label-smoothing 0.005 \
            --config 'configs/512/effnetb5_lr5e4_multi.json' 
done

: '
for FOLD in 0 1 2 3 4
do
    python training/pipeline/train_image_mask_classifier.py --device 'cuda' --fold $FOLD --batchsize 40 \
    --logdir 'logs/zoo' --augextra False --label-smoothing 0.0 --config 'configs/512/effnetb5_lr5e4_multi.json'
done
'
