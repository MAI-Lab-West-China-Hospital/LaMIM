import os
import sys
import torch
import monai
import random
import fnmatch
import logging
import torch.nn as nn
from monai.metrics import ROCAUCMetric
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import ViTAutoEnc
from torch.utils.tensorboard import SummaryWriter
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    LoadImaged,
    Compose,
    Resized,
    Activations, 
    AsDiscrete,
    ScaleIntensityd
)

from Models import Classifier


os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Select the GPU, default 0

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
    
    train_files = get_data('/path/to/train_data')
    val_files = get_data('/path/to/validation_data')

    # Define transforms for image
    transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96))
        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])   
  
    #Define the classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    net = Classifier()
    net = net.to(device)    
 
    # Define Hyper-paramters for training loop
    max_epoch = 200
    val_interval = 1
    batch_size = 36
    lr = 1e-5

    #Define loss & optimizer & learning rate scheduler
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=lr)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=max_epoch, eta_min=1e-6)
    auc_metric = ROCAUCMetric()
    
    # create a training data loader
    train_ds =  CacheDataset(data=train_files, transform=transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds =  CacheDataset(data=val_files, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())

    learning_rate = []
    train_loss_values = []
    accuracy_values = []
    auc_values = []

    # start a typical PyTorch training and validation
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()
    for epoch in range(max_epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epoch}")
        net.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            learning_rate.append(current_lr)           
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        train_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, net(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                accuracy_values.append(acc_metric)
                auc_values.append(auc_result)
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(net.state_dict(), "best_metric_classifier.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()  
    
if __name__=="__main__":
    main()
