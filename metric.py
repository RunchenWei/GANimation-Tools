from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import skimage.io as io
import torch
from pytorch_fid import fid_score
import os

groundtruth_path = '/home/wrc/project/GANimation/test-set/groundtruth'
result_path = '/home/wrc/project/GANimation/test-set/results'

gt_list = os.listdir(groundtruth_path)
re_list = os.listdir(result_path)

#gt_list.sort(key=lambda x:int(x[2:-4]))
#re_list.sort(key=lambda x:int(x[2:-4]))

Sum_PNSR = 0
PNSR_Count = 0
for gt_image in gt_list:
    gt_name = gt_image.split('.')[0]
    gt_image_path = os.path.join(groundtruth_path, gt_image);
    re_image_path = os.path.join(result_path, gt_image);
    image1 = io.imread(gt_image_path) / 1.0
    image2 = io.imread(re_image_path) / 1.0

    image1 = image1 / 255.0
    image2 = image2 / 255.0
    psnr_val = peak_signal_noise_ratio(image1, image2)
    Sum_PNSR += psnr_val
    PNSR_Count += 1
    
print("PNSR Value:", Sum_PNSR/PNSR_Count)
    
fid_val = fid_score.calculate_fid_given_paths([result_path, groundtruth_path],
				    	     device = torch.device("cuda"),
					     batch_size = 32,
					     dims=2048)
					     
print("FID Value:",fid_val)
