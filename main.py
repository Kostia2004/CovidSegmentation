import numpy as np 
import pandas as pd
import nibabel as nib
import cv2
from os import remove

data = pd.read_csv('metadata.csv')

def read_nii(filepath):
    filepath = filepath[26:]
    ct_scan = nib.load(filepath) #читаем с файла
    array   = ct_scan.get_fdata() #достаём numpy массив
    array   = np.rot90(np.array(array)) #поворачиваем я хз зачем
    return array

img_size = 512

metadata = pd.DataFrame(columns=['ct_img', 'lung_img'])



for i in range(len(data)):
  ct_name = data['ct_scan'][i]
  ct = read_nii(ct_name)
  ct_path = ct_name[25:]
  lung_name = data['lung_mask'][i]
  lung = read_nii(lung_name)
  lung_path = lung_name[25:]

  print(ct_name, lung_name)
  print(ct_path, lung_path)

  for ii in range(ct.shape[2]):
    ct_img = cv2.resize(ct[:,:,ii], dsize = (img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
    lung_img = cv2.resize(lung[:,:,ii],dsize=(img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
    ct_path = "lung_dataset/ct_imgs/"+ct_name.split('/')[-1][:-4]+"_"+str(ii)+".npy"
    lung_path = "lung_dataset/lung_imgs/"+lung_name.split('/')[-1][:-4]+"_"+str(ii)+".npy"
    metadata = metadata.append({'ct_img':ct_path, 'lung_img':lung_path}, ignore_index=True)
    print(ct_path, lung_path)
    np.save(ct_path, ct_img)
    np.save(lung_path, lung_img)

metadata.to_csv("resdf.csv", sep=',')
