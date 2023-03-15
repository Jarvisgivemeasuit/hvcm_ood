OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python3 ../ind_acc.py \
--model ResNet18Gram \
--dataset cifar10 \
--dataroot data \
--saveroot result \
--batch_size 128 \
--attri_dim 1024 \
--name cifar_hvcm \
--num_classes 10 \
--num_kernel 32 