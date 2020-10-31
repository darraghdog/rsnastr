for FOLD in 0 1 2 3
do
    WEIGHTS='classifier__RSNAClassifier_tf_efficientnet_b5_ns_04d_fold'$FOLD'_img512_accum1___best__all_size512.emb'
    python training/pipeline/train_sequence_classifier.py --device 'cuda' --fold $FOLD --imgemb $WEIGHTS --config configs/b5_seq_lstm.json
    python training/pipeline/train_sequence_classifier.py --device 'cuda' --fold $FOLD --imgemb $WEIGHTS --config configs/b5_seq_transformer_1layer.json
    python training/pipeline/train_sequence_classifier.py --device 'cuda' --fold $FOLD --imgemb $WEIGHTS --config configs/b5_seq_transformer_2layer.json
done