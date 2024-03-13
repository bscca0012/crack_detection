import os, shutil, random
random.seed(0)
import numpy as np
from sklearn.model_selection import train_test_split

val_size = 0
test_size = 0.2
postfix = 'png'
imgpath1 = 'Sun520/img'
imgpath2 = 'Sun520/gt'


os.makedirs('Sun520/train/img_train', exist_ok=True)
os.makedirs('Sun520/test/img_test', exist_ok=True)
os.makedirs('Sun520/train/gt_train', exist_ok=True)
os.makedirs('Sun520/test/gt_test', exist_ok=True)
listdir = np.array([i for i in os.listdir(imgpath2) if 'png' in i])
random.shuffle(listdir)
train, val, test = listdir[:int(len(listdir) * (1 - val_size - test_size))], listdir[int(len(listdir) * (1 - val_size - test_size)):int(len(listdir) * (1 - test_size))], listdir[int(len(listdir) * (1 - test_size)):]
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    shutil.copy('{}/{}.{}'.format(imgpath1, i[:-4], postfix), 'Sun520/train/img_train/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}.{}'.format(imgpath2, i[:-4], postfix), 'Sun520/train/gt_train/{}.{}'.format(i[:-4], postfix))

for i in test:
    shutil.copy('{}/{}.{}'.format(imgpath1, i[:-4], postfix), 'Sun520/test/img_test/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}.{}'.format(imgpath2, i[:-4], postfix), 'Sun520/test/gt_test/{}.{}'.format(i[:-4], postfix))
