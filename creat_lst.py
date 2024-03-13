import os
root = 'Sun520/train/'
img_paths = root + 'img_train'
gt_paths = root + 'gt_train'

f = open( 'Sun520_aug_train_pair.lst', 'w')

img_path = os.path.abspath(img_paths)
gt_path = os.path.abspath(gt_paths)

img_filenames = os.listdir(img_path)
gt_filenames = os.listdir(gt_path)
for i in range(len(img_filenames)):
    img = 'img_train' + '/' + img_filenames[i]
    gt = 'gt_train' + '/' + gt_filenames[i]
    print(img, gt)
    f.write(img + ' ' + gt + '\n')
f.close()



root2 = 'Sun520/test/'
img_paths = root2 + 'img_test'
gt_paths = root2 + 'gt_test'

f = open( 'Sun520_test.lst', 'w')

img_path2 = os.path.abspath(img_paths)
gt_path2 = os.path.abspath(gt_paths)

img_filenames2 = os.listdir(img_path2)
gt_filenames2 = os.listdir(gt_path2)

for i in range(len(img_filenames2)):
    img2 = 'img_train' + '/' + img_filenames2[i]
    gt2 = 'gt_train' + '/' + gt_filenames2[i]
    print(img2, gt2)
    f.write(img2 + ' ' + gt2 + '\n')
f.close()