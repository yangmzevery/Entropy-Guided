# AdaFocal
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--master_port 9560 \
--nproc_per_node=2 --use_env train.py \
--dataset 20_newsgroups \
--model bert-base-uncased \
--max-length 256 \
--epochs 20 \
--batch-size 64 \
--learning-rate 2e-5 \
--num-labels 20 \
--loss AdaFocal


