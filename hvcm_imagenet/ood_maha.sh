OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python ./ood_maha.py \
--arch resnet50 \
--num_labels 100 \
--out_dim 8192 \
--num_kernel 32 \
--batch_size_per_gpu 64 \
--ind_path /path/to/ImageNet/ \
--ind_path /path/to/ood_datasets/ \
--pretrained_weights /path/to/checkpoint.pth
