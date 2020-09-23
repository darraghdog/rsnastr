python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra False  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_lr1e4_binary.json
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 48 --logdir 'logs/chkme' --augextra True  --label-smoothing 0.01 --config configs/_lr2308/rnxt101_lr1e4_binary.json

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
