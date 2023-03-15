OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python ./get_gau.py \
--arch resnet50 \
--data_path data \
--output_dir results \
--batch_size_per_gpu 128 \
--num_labels 100 \
--num_kernel 32 \
--out_dim 8192
