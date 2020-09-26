python training/pipeline/infer_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/focal' \
        --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json' \
        --weightsrgx 'classifier_RSNAClassifier_resnext101_32x8d_0__fold0_epoch2*' --epochs '20|21|22|23|24'
        --infer true
