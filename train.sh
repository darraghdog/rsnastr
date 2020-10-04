for FOLD in 0 1 2 3 4
do 
    python training/pipeline/train_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb5_lr5e4_binary.json'
done

: '
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --accum 4 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --accum 2 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --accum 1 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --accum 4 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --accum 2 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --accum 1 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
'



: '
for FOLD in 0 1 2 3 4
do 
    python training/pipeline/train_image_classifier.py --device 'cuda' --fold $FOLD --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/resnest200e_lr5e4_binary.json'
done

python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb3_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb3_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 2 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb3_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 3 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb3_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 4 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb3_lr5e4_binary.json'

python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb5_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb5_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 2 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb5_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 3 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb5_lr5e4_binary.json'
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 4 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.0 --config 'configs/effnetb5_lr5e4_binary.json'
'
: '
for FOLD in 0 1
do
    for SMOOTH in 0.0 0.005 0.01
    do
        python training/pipeline/train_image_classifier.py --device 'cuda' --fold $FOLD \
       	    --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing $SMOOTH \
       	    --config 'configs/effnetb5_lr5e4_binary.json'
    done
done
'

: '
Epoch: 24 bce: 0.32642, bce_best: 0.31309
Epoch: 24 bce: 0.32289, bce_best: 0.32159
Epoch: 24 bce: 0.31331, bce_best: 0.30927
Epoch: 24 bce: 0.31583, bce_best: 0.30183
Epoch: 24 bce: 0.32569, bce_best: 0.31173
'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 2 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 3 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 4 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/effnetb5_lr1e4_binary.json'

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config  'configs/densenet169_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config  'configs/densenet169_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 2 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config  'configs/densenet169_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 3 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config  'configs/densenet169_lr1e4_binary.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 4 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config  'configs/densenet169_lr1e4_binary.json'


#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/se101_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/effnetb5_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 24 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/effnetb7_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/mixnet_xl_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/densenet169_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/densenet201_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 --logdir 'logs/zoo' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/dpn92_lr1e4_binary_focal_pe0.25.json'



#Fold 0 Epoch: 24 bce: 0.31407, bce_best: 0.31407
#Fold 1 Epoch: 24 bce: 0.35100, bce_best: 0.32609
#Fold 2 Epoch: 24 bce: 0.32645, bce_best: 0.30949
#Fold 3 Epoch: 24 bce: 0.31574, bce_best: 0.31444
#Fold 4 Epoch: 24 bce: 0.33733, bce_best: 0.31755
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 2 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 3 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 4 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json'




# =======================
# python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/rnxt101_binary.json'
# python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25_ep50.json'
# python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config 'configs/_lr2308/rnxt101_lr1e4_binary_focal_pe0.25.json'
# python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --mixup_beta 0.3  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_lr1e4_binary_mixup.json


# Focal Loss
## New best @
#2020-09-24 10:09:01,506 - Train - INFO - Negimg PosStudy loss 0.2200 acc 0.9086; Posimg PosStudy loss 0.5284 acc 0.7400; Negimg NegStudy loss 0.1901 acc 0.9246; Avg 3 loss 0.3128 acc 0.8577
#Epoch 18 improved from 0.31339 to 0.31283
#Epoch: 18 bce: 0.31283, bce_best: 0.31283
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/focal' --augextra False  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_lr1e4_binary_focal.json


## Best @ 
#2020-09-23 18:37:20,842 - Train - INFO - Negimg PosStudy loss 0.2039 acc 0.9158; Posimg PosStudy loss 0.5814 acc 0.7278; Negimg NegStudy loss 0.1653 acc 0.9362; Avg 3 loss 0.3169 acc 0.8599
#Epoch: 18 bce: 0.31687, bce_best: 0.31687

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra False  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_lr1e4_binary.json







#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra True  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_lr1e4_binary.json

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra False  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_binary.json
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra True  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_binary.json

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra True  --label-smoothing 0.01 --config configs/rnxt101_binary.json 
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra False  --label-smoothing 0.01 --config configs/rnxt101_binary.json 


#for RATIO in '05' '02' '005'
#do 
#    CONFIG=configs/peratio/rnxt101_binary_pe$RATIO.json
#    LOGS=logs/rnxt101_peratio$RATIO
#    python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 \
#	    --logdir $LOGS  --label-smoothing 0.01 --config $CONFIG
#done

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 \
#    --logdir 'logs/nmin2_nmax4'  --label-smoothing 0.01 --config 'configs/nmin/rnxt101_binary_nmin2_nmax4.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 \
#    --logdir 'logs/nmin2_nmax6'  --label-smoothing 0.01 --config 'configs/nmin/rnxt101_binary_nmin2_nmax6.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 \
#    --logdir 'logs/nmin2_nmax8'  --label-smoothing 0.01 --config 'configs/nmin/rnxt101_binary_nmin2_nmax8.json'
#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 \
#    --logdir 'logs/nmin2_nmax16'  --label-smoothing 0.01 --config 'configs/nmin/rnxt101_binary_nmin2_nmax16.json'


#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 \
#    --logdir 'logs/chkme'  --label-smoothing 0.01 --config 'configs/nmin/rnxt101_binary.json'

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 1 --batchsize 32 \
#    --logdir 'logs/rnxt101_2109'  --label-smoothing 0.01 --config 'configs/rnxt101_binary.json'

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 32 \
#    --logdir 'logs/rnxt101_2109'  --label-smoothing 0.01 --config 'configs/rnxt101_binary.json'

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 96  \
#    --logdir 'logs/b2_2020'  --label-smoothing 0.01 --config 'configs/b2_binary.json'
