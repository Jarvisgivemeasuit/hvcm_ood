OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29502 ./ind_acc.py \
--arch resnet50 \
--num_labels 100 \
--out_dim 8192 \
--batch_size_per_gpu 128 \
--num_kernel 32 \
--data_path data \
--pretrained_weights results/checkpoint.pth