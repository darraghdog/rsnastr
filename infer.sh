FOLD=0
EPOCH=20
for FOLD in 0 1 2 3 4
do
    WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_"$FOLD"__nclasses10_size512_fold"$FOLD"_epoch"$EPOCH
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 192 \
        --config 'configs/512/effnetb5_lr5e4_multi.json' \
        --weightsrgx $WEIGHTS --epochs '' \
        --infer false --emb true
done



: '
# Best epoch - 0.177 LB
FOLD=0
EPOCH=20
for FOLD in 0 1 2 3 4
do
    WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b7_ns_"$FOLD"__nclasses1_size512_fold"$FOLD"_epoch"$EPOCH
    #WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_"$FOLD"__nclasses1_fold"$FOLD"_epoch"$EPOCH
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 32 \
        --config 'configs/512/effnetb7_lr5e4_binary.json' \
   	--weightsrgx $WEIGHTS --epochs '' \
        --infer false --emb true
done
'

: '
FOLD=0
for EPOCH in 10 30 # 20
do
    WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_"$FOLD"__nclasses1_fold"$FOLD"_epoch"$EPOCH
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 128 \
        --config 'configs/512/effnetb5_lr5e4_binary.json' \
        --weightsrgx $WEIGHTS --epochs '' \
        --infer false --emb true
done
'

: '
# Best epoch - 0.169 LB
for FOLD in 0 1 2 3 4
do
    WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_"$FOLD"__nclasses1_fold"$FOLD"_epoch20"
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 128 \
        --config 'configs/512/effnetb5_lr5e4_binary.json' \
        --weightsrgx $WEIGHTS --epochs '' \
        --infer false --emb true
done
'

: '
for FOLD in 0 1 2 3 4
do
    WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_"$FOLD"__fold"$FOLD"_best_dice_all"
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 128 \
        --config 'configs/512/effnetb5_lr5e4_binary.json' \
        --weightsrgx $WEIGHTS --epochs '' \
        --infer false --emb true
done
'

: '
for FOLD in 0 1 2 3 4 # 0 1
do
    WEIGHTS='exam_lstm_tf_efficientnet_b2_ns_epoch31_fold'$FOLD'.bin*'
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 128 \
        --config 'configs/effnetb2_lr5e4_multi.json' \
        --weightsrgx $WEIGHTS --epochs '' --type 'study' \
        --infer false --emb true
done
'

: '
for FOLD in 0 1 2 3 4 # 0 1
do
    WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_"$FOLD"__fold"$FOLD"_best_dice"
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 64 \
        --config 'configs/effnetb5_lr5e4_binary.json' \
        --weightsrgx $WEIGHTS --epochs '' \
        --infer false --emb true
done
'





: '
for FOLD in 0 1 2 3 4
do
    WEIGHTS="classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_"$FOLD"__fold"$FOLD"_epoch2*"
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 256 \
        --config 'configs/effnetb5_lr1e4_binary.json' \
        --weightsrgx $WEIGHTS --epochs '24' \
        --infer false --emb true
done
'

: '
for FOLD in 0 1 2 3 4
do
    WEIGHTS="classifier_RSNAClassifier_resnext101_32x8d_"$FOLD"__fold"$FOLD"_epoch2*"
    python training/pipeline/infer_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 256 \
        --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json' \
        --weightsrgx $WEIGHTS --epochs '24' \
        --infer false --emb true
done
'
