export DATA_DIR=t1_s0

CUDA_VISIBLE_DEVICES=2 /usr/bin/python main.py \
 --topks [10] \
 --layer 4 \
 --dropout 1 \
 --keepprob 0.9 \
 --decay 6e-5 \
 --lr 1e-3 \
 --a_fold 100 \
 --load 0 \
 --inference 0 \
 --seed 792 \
 --model lgn
