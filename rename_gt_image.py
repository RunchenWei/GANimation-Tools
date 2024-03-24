import os

groundtruth_path = '/home/wrc/project/GANimation/test-set/groundtruth'

gt_list = os.listdir(groundtruth_path)

for gt_image in gt_list:
    gt_image_path = os.path.join(groundtruth_path, gt_image);
    split_list = gt_image_path.split('.')
    if '_22' not in split_list[0]:
        new_name = split_list[0] + '_22.' + split_list[1]
        os.rename(gt_image_path, new_name)
