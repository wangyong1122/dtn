#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python generate_dtn.py /path/to/data \
    --task $task -s $src -t $tgt \
    --gen-subset test \
    --domains name_of_domains \
    --valid-domains $domain \
    --test-domains $domain \
    --path /path/to/ckpt \
    --batch-size 256 \
    --quiet --remove-bpe \
    --decoding-path /path/to/decoding \
    --num-ref num_of_ref \
    --valid-decoding-path /path/to/data/valid_test_txt \
    --multi-bleu-path ./scripts