import os
import random
import shutil

path = "hvcm_cifar10/data/places/test_256/"
sample_list = os.listdir(path)
samples = random.sample(sample_list, 10000)

save_path = 'hvcm_cifar10/data/places_sample/images'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for idx, sam in enumerate(samples):
    file_path = os.path.join(path, sam)
    shutil.copy(file_path, os.path.join(save_path, sam))
    if idx % 100 == 0:
        print(f"{idx} images copy complete.")