import os
import sys
import torch
import monai
import random
import fnmatch
import logging
import torch.nn as nn
from monai.networks.nets import ViTAutoEnc
from monai.data import DataLoader, CacheDataset, CSVSaver
from monai.transforms import (
    LoadImaged,
    Compose,
    Resized,
    ScaleIntensityd
)

from Models import Classifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Select the GPU

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    def get_data(path):
        #normal && tumor  meningioma && glioma  IDH-wt && IDH-mut
        data_0_dir = os.path.join(path,'') #Name of the class labeled 0, e.g.: normal
        data_1_dir = os.path.join(path,'') #Name of the class labeled 1, e.g.: tumor
        
        data_0=list()    
        for path,dirs,files in os.walk(data_0_dir):
            for f in fnmatch.filter(files,'*.nii.gz'):
                data_0.append(os.path.join(path,f))             
                 
        data_1=list()
        for path,dirs,files in os.walk(data_1_dir):
            for f in fnmatch.filter(files,'*.nii.gz'):
                data_1.append(os.path.join(path,f)) 
           
        # 2 binary labels for classification: tumor and normal
        files_0 = [{"img": img[0], "label": 0} for img in zip(data_0)]
        files_1 = [{"img": img[0], "label": 1} for img in zip(data_1)]
        data = files_0 + files_1
        random.shuffle(data)
        print('Total Number of training Data Samples:{}'.format(len(data)))
        return(data)
 
    test_files = get_data('/path/to/test_data')
    
    # Define transforms for image
    transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96))
        ]
    )
  
    #Define the classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Classifier()
    net = net.to(device)

    net.load_state_dict(torch.load("best_metric_classifier.pth"))
    net.eval()
 
    batch_size = 24
 
    # create a test data loader
    test_ds = CacheDataset(data=test_files, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())

    # test classifier
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        saver = CSVSaver(output_dir="./output")
        for test_data in test_loader:
            test_images, test_labels = test_data["img"].to(device), test_data["label"].to(device)
            test_outputs = net(test_images).argmax(dim=1)
            value = torch.eq(test_outputs, test_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            saver.save_batch(test_outputs, test_data["img"].meta)
        metric = num_correct / metric_count
        print("evaluation metric:", metric)
        saver.finalize()  

if __name__=="__main__":
    main()
