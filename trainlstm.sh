
EPOCH=20
FOLD=0
DO=0.0
for FOLD in 0 1 2 3 4 
do
    WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses1_fold'$FOLD'_epoch'$EPOCH'__all_size512.emb'
    python training/pipeline/train_study_classifier_v02.py \
        --lr 0.00005 --dropout $DO --device 'cuda' --fold $FOLD --batchsize 32 --epochs 25 --lrgamma 0.98 \
        --imgembrgx $WEIGHTS
done

: '
EPOCH=20
FOLD=0
DO=0.0
for EPOCH in 20 # 10 30
do
    WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__nclasses1_fold*_epoch'$EPOCH'__all_size512.emb'
    python training/pipeline/train_study_classifier_v02.py \
        --lr 0.00005 --dropout $DO --device 'cuda' --fold $FOLD --batchsize 32 --epochs 25 --lrgamma 0.98 \
        --imgembrgx $WEIGHTS
done
'
: ' 
for FOLD in 0 1 2
do
    for DO in 0.01 0.03 0.05
    do
        python training/pipeline/train_study_classifier_v02.py \
        --lr 0.00005 --dropout $DO --device 'cuda' --fold $FOLD --batchsize 32 --epochs 25 --lrgamma 0.98 \
        --imgembrgx 'weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_best_dice__all_size320.emb'
    done
done
'

#python training/pipeline/train_study_classifier.py --lr 0.0001   --label-smoothing 0.0 \
#        --device 'cuda' --fold 0 --batchsize 32 /
#        --embrgx 'weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_best_dice__all_size320.emb'

# python training/pipeline/train_study_classifier.py --lr 0.01   --label-smoothing 0.0  --device 'cuda' --fold 0 --batchsize 4 --embrgx 'weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_best_dice__hflip0_transpose0_size320.emb' 
