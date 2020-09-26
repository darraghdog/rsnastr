for FOLD in 0 1 2 3 4
do
    WEIGHTS="classifier_RSNAClassifier_resnext101_32x8d_"$FOLD"__fold"$FOLD"_epoch2*"
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 256 \
        --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json' \
        --weightsrgx $WEIGHTS --epochs '24' \
        --infer false --emb true
done
