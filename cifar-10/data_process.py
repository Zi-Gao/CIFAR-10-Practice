import cv2
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from utils.trans_pic import to_tensor

df = pd.read_csv('data/trainLabels.csv')
NUM_TRAIN_PIC=len(df)
all_label=df['label'].unique()
all_label.sort()
label_id={all_label[i]:i for i in range(len(all_label))}
label=np.array([label_id[x] for x in df['label']],dtype=np.uint8)
print(all_label)
print(label)
print(df['label'])

np.save('data/train_label.npy',label)

PIC_CHAN=3
PIC_SIZE=32

def read_pic(n,path):
    pic=np.ndarray((n,PIC_CHAN,PIC_SIZE,PIC_SIZE),dtype=np.uint8)
    for i in tqdm.trange(1,n+1):
        pic[i-1]=to_tensor(np.array(Image.open(f'{path}/{i}.png')))
    return pic

np.save('data/train_pic.npy',read_pic(NUM_TRAIN_PIC,'data/train'))
np.save('data/test_pic.npy',read_pic(len(pd.read_csv('data/sampleSubmission.csv')),'data/test'))
