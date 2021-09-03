import os,csv,random
from time import time
import pathlib, argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch, torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import lightbridge.data as dataset
import lightbridge.layers as layers
import lightbridge.utils as utils
import lightbridge.models as models
device="cuda:0"

def train(model,train_dataloader, val_dataloader, args):        
    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    print('training starts.')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.5)
    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.epochs):
        log = [epoch]
        model.train()
        train_len, train_running_counter, train_running_loss = 0.0, 0.0, 0.0
        tk0 = tqdm(train_dataloader, ncols=125, total=int(len(train_dataloader)))
        for train_iter, train_data_batch in enumerate(tk0):
            train_images, train_labels = utils.data_to_cplex(train_data_batch, device=device)
            train_outputs = model(train_images)
            train_loss_ = criterion(train_outputs, train_labels)
            train_counter_ = torch.eq(torch.argmax(train_labels, dim=1), torch.argmax(train_outputs, dim=1)).float().sum()
           
            optimizer.zero_grad()
            train_loss_.backward(retain_graph=True)
            optimizer.step()
            train_len += len(train_labels)
            train_running_loss += train_loss_.item()
            train_running_counter += train_counter_

            train_loss = train_running_loss / train_len
            train_accuracy = train_running_counter / train_len

            tk0.set_description_str('Epoch {}/{} : Training'.format(epoch, args.start_epoch + 1 + args.epochs - 1))
            tk0.set_postfix({'Train_Loss': '{:.2f}'.format(train_loss), 'Train_Accuracy': '{:.5f}'.format(train_accuracy)})
        scheduler.step()
        log.append(train_loss)
        log.append(train_accuracy)
        torch.save(model.state_dict(), (args.model_save_path + str(epoch) + args.model_name))
        print('Model : "' + args.model_save_path + str(epoch) + args.model_name + '" saved.')

        with open(args.result_record_path, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log)
        val_loss, val_accuracy = eval(model, val_dataloader, epoch, args)
        log.append(val_loss)
        log.append(val_accuracy)
    return train_loss, train_accuracy, val_loss, val_accuracy, log 

def eval(model, val_dataloader, args):
    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    with torch.no_grad():
        model.eval()
        val_len,val_running_counter,val_running_loss = 0.0, 0.0, 0.0
        tk1 = tqdm(val_dataloader, ncols=100, total=int(len(val_dataloader)))
        for val_iter, val_data_batch in enumerate(tk1):
            val_images, val_labels = utils.data_to_cplex(val_data_batch,device=device)
            val_outputs = model(val_images)

            val_loss_ = criterion(val_outputs, val_labels)
            val_counter_ = torch.eq(torch.argmax(val_labels, dim=1), torch.argmax(val_outputs, dim=1)).float().sum()

            val_len += len(val_labels)
            val_running_loss += val_loss_.item()
            val_running_counter += val_counter_

            val_loss = val_running_loss / val_len
            val_accuracy = val_running_counter / val_len

            tk1.set_description_str('Epoch {}/{} : Validating'.format(args.start_epoch, args.start_epoch + 1 + args.epochs - 1 ))
            tk1.set_postfix({'Val_Loss': '{:.5f}'.format(val_loss), 'Val_Accuarcy': '{:.5f}'.format(val_accuracy)})
    return val_loss, val_accuracy
   
def main(args):
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    if args.dataset == "mnist":
        print("training and testing on MNIST10 dataset")
        load_dataset = dataset.load_dataset(batch_size = args.batch_size, system_size = args.sys_size, datapath = "./data")
        train_dataloader, val_dataloader = load_dataset.MNIST()
    elif args.dataset == "Fmnist":
        print("training and testing on FashionMNIST10 dataset")
        load_dataset = dataset.load_dataset(batch_size = args.batch_size, system_size = args.sys_size, datapath = "./Fdata")
        train_dataloader, val_dataloader = load_dataset.FMNIST()
    else:
        assert(0), "current version only supports MNIST10 and FashionMNIST10"

    model = models.DiffractiveClassifier_Raw(wavelength=args.wavelength, pixel_size=args.pixel_size, 
			sys_size=args.sys_size, distance=args.distance, pad = args.pad, num_layers=args.depth, 
			amp_factor=args.amp_factor,  Fresnel = args.Fresnel,  Fraunhofer = args.Fraunhofer)
    model.to(device)
    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) +  args.model_name))
        print('Model1 : "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
    else:
        if os.path.exists(args.result_record_path):
            os.remove(args.result_record_path)
        else:
            with open(args.result_record_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ['Epoch', 'Train_Loss', "Train_Acc", 'Val_Loss', "Val_Acc", "LR"])

    if args.evaluation:
        eval(model, val_dataloader, args)
        return
    else:
        start_time = time()
        train(model, train_dataloader, val_dataloader, args)
        print('run time', time()-start_time)
        return
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default="mnist", help='define train/test dataset (mnist, cifar10, cifar100)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--depth', type=int, default=5, help='number of fourier optic transformations/num of layers')
    parser.add_argument('--whether-load-model', type=bool, default=False, help="load pre-train model")
    parser.add_argument('--evaluation', type=bool, default=False, help="Evaluation only")
    parser.add_argument('--start-epoch', type=int, default=0, help='load pre-train model at which epoch')
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="save training result.")
    parser.add_argument('--sys-size', type=int, default=200, help='system size (dim of each diffractive layer)')
    parser.add_argument('--distance', type=float, default=0.66, help='layer distance (default=0.1 meter)')
    parser.add_argument('--amp-factor', type=float, default=2, help='regularization factors to balance phase-amplitude where they share same downstream graidents')
    parser.add_argument('--pixel-size', type=float, default=0.000036, help='the size of pixel in diffractive layers')
    parser.add_argument('--wavelength', type=float, default=5.32e-7, help='wavelength')
    parser.add_argument('--pad', type=int, default=100, help='the padding size ')
    parser.add_argument('--Fresnel', type=bool, default=False, help="Use Fresnel Approximation, otherwise Sommerfeld Approximation.")
    parser.add_argument('--Fraunhofer', type=bool, default=False, help="Use Fraunhofer Approximation, otherwise Sommerfeld Approximation.")
 
    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)
