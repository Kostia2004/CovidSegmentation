import numpy as np 
import pandas as pd
import nibabel as nib
import cv2
from os import remove
import gzip
import shutil

data = pd.read_excel('COVID19_1110/dataset_registry.xlsx').iloc[254:304, :]

def read_nii(filepath):
    filepath = 'COVID19_1110'+filepath
    with gzip.open(filepath, 'rb') as f_in:
      with open(filepath[:-3], 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    ct_scan = nib.load(filepath[:-3]) #читаем с файла
    array   = ct_scan.get_fdata() #достаём numpy массив
    array   = np.rot90(np.array(array)) #поворачиваем я хз зачем
    return array

img_size = 512

metadata = pd.read_csv('resdf.csv')



for i in range(len(data)):
  ct_name = data['study_file'][i+254]
  ct = read_nii(ct_name)
  ct_path = ct_name[25:]
  infect_name = data['mask_file'][i+254]
  infect = read_nii(infect_name)
  infect_path = infect_name[25:]
  for ii in range(ct.shape[2]):
    lung_img = cv2.resize(ct[:,:,ii], dsize = (img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
    infect_img = cv2.resize(infect[:,:,ii],dsize=(img_size, img_size),interpolation = cv2.INTER_AREA).astype('uint8')
    ct_path = "ct_imgs/"+ct_name.split('/')[-1][:-4]+"_"+str(ii)+".npy"
    infect_path = "infection_imgs/"+infect_name.split('/')[-1][:-4]+"_"+str(ii)+".npy"
    metadata = metadata.append({'ct_img':ct_path, 'infect_img':infect_path}, ignore_index=True)
    print(ct_path)
    np.save(ct_path, lung_img)
    np.save(infect_path, infect_img)

metadata.to_csv("resdf.csv", sep=',')
