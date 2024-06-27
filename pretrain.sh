nohup python ./main.py \
                --data ./dset/EEG/train_concat_data.npz \
                -b 80 \
                -wd 1e-4 \
                --warmup 0 \
                --gpu-index 6 \
                --device 'cuda' \
                --fp16precision \
                --n_method 1 \
                --pretrain_epochs 1 \
                --temperature 0.2 \
                --pretrain_lr 2e-5 \
                --in_channel 1 \
                --h_dim 256 \
                --vocab_size 5000 \
                --beta 0.5 \
                --n_labels 5 \
                --pretrain &>./logs/pretrain.log &
