python training/pipeline/train_study_classifier.py --lr 0.0001   --label-smoothing 0.0 \
        --device 'cuda' --fold 0 --batchsize 32 /
        --embrgx 'weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_best_dice__all_size320.emb'

# python training/pipeline/train_study_classifier.py --lr 0.01   --label-smoothing 0.0  --device 'cuda' --fold 0 --batchsize 4 --embrgx 'weights/classifier_RSNAClassifier_tf_efficientnet_b5_ns_04d_*__fold*_best_dice__hflip0_transpose0_size320.emb' 
