
python training/pipeline/train_image_classifier.py --device 'cuda' --fold 0 --batchsize 64 --imgsize 384  \
    --logdir 'logs/b2_2020'  --label-smoothing 0.01 --config 'configs/b2_binary.json'
