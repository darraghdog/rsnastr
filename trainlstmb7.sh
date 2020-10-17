
EPOCH=20
FOLD=0
DO=0.0
for FOLD in 0 1 2 3 4 
do
    WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b7_ns_'$FOLD'__nclasses1_size512_fold'$FOLD'_epoch'$EPOCH'__all_size512.emb'
    python training/pipeline/train_study_classifier_v02.py \
        --lr 0.00005 --dropout $DO --device 'cuda' --fold $FOLD --batchsize 32 --epochs 25 --lrgamma 0.98 \
        --lstm_units 512 --imgembrgx $WEIGHTS
done

