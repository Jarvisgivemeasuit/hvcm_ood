OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3 ./get_gau.py \
--model ResNet18Gram \
--name cifar_hvcm \
--dataset cifar10 \
--dataroot data \
--saveroot results \
--batch_size 100 \
--attri_dim 1024 \
--num_kernel 32