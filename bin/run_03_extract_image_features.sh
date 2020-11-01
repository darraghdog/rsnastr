for FOLD in 0 1 2 # 3 4 
do
    WEIGHTS='classifier__RSNAClassifier_tf_efficientnet_b5_ns_04d_fold'$FOLD'_img512_accum1___best'
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 192 \
        --config 'configs/effnetb5_lr5e4_multi.json' --weights $WEIGHTS --label-smoothing 0.0 
done
