
: '
bsize=32
DO=0.0
for FOLD in 0 1 
do
    WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses10_size512_fold'$FOLD'_epoch20'
    python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 16  --lrgamma 0.98 \
        --hidden_size 2048 --nlayers 1 --delta False --imgembrgx $WEIGHTS

    python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 16  --lrgamma 0.98 \
        --hidden_size 1024 --nlayers 2 --delta False --imgembrgx $WEIGHTS
done
'
: '
# Still to be run
bsize=32
DO=0.0
for FOLD in 2 3
do
    WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses10_size512_accum4_fold'$FOLD'_epoch15'
    python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 16  --lrgamma 0.98 \
        --hidden_size 2048 --nlayers 1 --delta False --imgembrgx $WEIGHTS

    python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 16  --lrgamma 0.98 \
        --hidden_size 1024 --nlayers 2 --delta False --imgembrgx $WEIGHTS
done
'

EPOCH=12
FOLD=0
DO=0.0
bsize=48
for FOLD in 0 3 # 0 1 2 3
do

    WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses10_even_size512_accum8_fold'$FOLD'_epoch'$EPOCH'__all_size512.emb'
    python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 14  --lrgamma 0.98 \
        --hidden_size 2048 --nlayers 1 --delta False --imgembrgx $WEIGHTS

    WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses10_even_size512_accum8_fold'$FOLD'_epoch'$EPOCH'__all_size512.emb'
    python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 14  --lrgamma 0.98 \
        --hidden_size 1024 --nlayers 2 --delta False --imgembrgx $WEIGHTS
done

: '
# 0.157 together with lstm for accum
EPOCH=15
FOLD=0
DO=0.0
for FOLD in 0 1 2 3 # 4 
do
    for bsize in 32 
    do
        WEIGHTS='weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_'$FOLD'__nclasses10_size512_accum4_fold'$FOLD'_epoch'$EPOCH'__all_size512.emb'
        python training/pipeline/train_transformer_classifier_v02.py \
        --lr 0.00005 --lstm_unit 512 --dropout $DO --device 'cuda' --fold $FOLD --batchsize $bsize --epochs 25	--lrgamma 0.98 \
        --delta False --imgembrgx $WEIGHTS
    done
done
'
: '
EPOCH=20
FOLD=0
DO=0.0
for FOLD in 0 1 2 3 # 0 1  4
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
