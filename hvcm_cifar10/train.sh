OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python3 ./train.py \
--hvcm \
--sgpu 0 \
--lr 0.1 \
--epoch 200 \
--model ResNet18Gram \
--name cifar_hvcm \
--decay 1e-4 \
--dataset cifar10 \
--dataroot data/ \
--saveroot results/ \
--attri_dim 1024 \
--num_kernel 32 \
--alpha 1 \
--beta 0.1 \
--gamma 1e-4