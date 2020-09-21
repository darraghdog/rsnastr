python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 96  \
    --logdir 'logs/rnxt101_2109'  --label-smoothing 0.01 --config 'configs/rnxt101_binary.json'

#python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 96  \
#    --logdir 'logs/b2_2020'  --label-smoothing 0.01 --config 'configs/b2_binary.json'
