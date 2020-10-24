EPOCH=15
FOLD=0
DO=0.0
for FOLD in 0 # 1 2 3 # 4 
do
    for bsize in 64 
    do
        WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses10_size512_accum4_fold'$FOLD'_epoch'$EPOCH'__all_size512.emb'
        python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 15 --lrgamma 0.95 \
        --delta False --imgembrgx $WEIGHTS
    done
done


: '
# Best - 0.160
EPOCH=20
FOLD=0
DO=0.0
for FOLD in 2 3 # 0 1  4 
do
    for bsize in 64 
    do
        WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses10_size512_fold'$FOLD'_epoch'$EPOCH'__all_size512.emb'
        python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 14 --lrgamma 0.95 \
        --delta False --imgembrgx $WEIGHTS
    done
done

'