nohup python ./main.py \
                --data ./dset/EEG/test_concat_data.npz \
                --ckpt ./runs/Jun21_09-06-06_yq01-inf-hic-k8s-a100-aa24-0494.yq01.baidu.com_finetune/checkpoint_finetuned.pth.tar \
                --gpu-index 4 \
                --device 'cuda' \
                --in_channel 1 \
                --h_dim 256 \
                --vocab_size 9000 \
                --beta 0.5 \
                --dropout 0.1 \
                --n_labels 5 \
                --eval &>>./logs/eval.log &
