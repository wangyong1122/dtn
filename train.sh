#!/bin/bash

if [ ! -d /path/to/data/valid_test_txt ]; then
    mkdir -p /path/to/ckpt/valid_test_txt
    cp -r /path/to/data/valid_test_txt /path/to/ckpt
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multidomain.py /path/to/data \
    --ddp-backend no_c10d \
    --task translation_multidomain -s $src -t $tgt \
    --restore-file checkpoint_last.pt \
    -a transformer --optimizer adam --lr 0.0005 \
    --domains name_of_domains\
    --train-domains $domain \
    --valid-domains name_of_domains\
    --valid-select $domain \
    --label-smoothing 0.1 --dropout 0.1 --max-tokens 4096 --update-freq 2 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --max-update 100000 \
    --criterion label_smoothed_cross_entropy \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --save-dir /path/to/ckpt \
    --max-sentences-valid 256 \
    --save-interval-updates 1000 --validate-interval-updates 1000 --save-interval 1 --validate-interval 1 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --quiet --beam 1 --max-len-a 2 --max-len-b 0 \
    --num-ref num_of_ref \
    --valid-decoding-path /path/to/ckpt/valid_test_txt \
    --multi-bleu-path ./scripts --remove-bpe | tee /path/to/ckpt/backup-log.txt