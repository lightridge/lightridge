import os
import csv
from time import time
import random
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import lightbridge.layers as layers
import lightbridge.utils as utils
import lightbridge.models as models
import lightbridge.data as dataset

device='cuda:0'

def prop_vis(model, val_dataloader, epoch, args):
    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    with torch.no_grad():
        model.eval()
        tk1 = tqdm(val_dataloader, ncols=100, total=int(len(val_dataloader)))
        for val_iter, val_data_batch in enumerate(tk1):
            val_images, val_labels = utils.data_to_cplex(val_data_batch,device=device)
            val_outputs = model.prop_view(val_images)
            return 

def train(model,train_dataloader, val_dataloader,lambda1, args):        
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    print('training starts.')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.5)
    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.epochs):
        log = [epoch]
        model.train()
        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0
        tk0 = tqdm(train_dataloader, ncols=150, total=int(len(train_dataloader)))
        for train_iter, train_data_batch in enumerate(tk0):
            train_images, train_labels = utils.data_to_cplex(train_data_batch, device='cuda:0')
            train_outputs = model(train_images)
            train_loss_ = lambda1 * criterion(train_outputs, train_labels)
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

def eval(model, val_dataloader, epoch, args):
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    with torch.no_grad():
        model.eval()
        val_len = 0.0
        val_running_counter = 0.0
        val_running_loss = 0.0

        tk1 = tqdm(val_dataloader, ncols=100, total=int(len(val_dataloader)))
        for val_iter, val_data_batch in enumerate(tk1):
            val_images, val_labels = utils.data_to_cplex(val_data_batch,device='cuda:0')
            val_outputs = model(val_images)

            val_loss_ = criterion(val_outputs, val_labels)
            val_counter_ = torch.eq(torch.argmax(val_labels, dim=1), torch.argmax(val_outputs, dim=1)).float().sum()

            val_len += len(val_labels)
            val_running_loss += val_loss_.item()
            val_running_counter += val_counter_

            val_loss = val_running_loss / val_len
            val_accuracy = val_running_counter / val_len

            tk1.set_description_str('Epoch {}/{} : Validating'.format(epoch, args.start_epoch + 1 + args.epochs - 1 ))
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

    phase_file =  args.phase_file
    phase_function = utils.phase_func(phase_file,  i_k=2**args.precision)
    with open('phase_file.npy', 'wb') as f_phase:
        np.save(f_phase, phase_function.cpu().numpy())
    intensity_file = args.intensity_file
    intensity_function = utils.intensity_func(intensity_file,  i_k=2**args.precision)
    with open('intensity_file.npy', 'wb') as f_amp:
        np.save(f_amp, intensity_function.cpu().numpy())
        
    model = models.DiffractiveClassifier_CoDesign(num_layers=args.depth, batch_norm =args.use_batch_norm,device=device, 
			wavelength=args.wavelength, pixel_size = args.pixel_size, sys_size=args.sys_size, 
			pad = args.pad, distance=args.distance,phase_func=phase_function, intensity_func=intensity_function, 
			precision=2**args.precision, amp_factor=args.amp_factor, Fresnel= args.Fresnel, Fraunhofer = args.Fraunhofer)
    model.to(device)
    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) +  args.model_name))
        print('Model1 : "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
        if args.get_phase:
            utils.get_phase(model, args)
        if args.save_w:
            eval(model, val_dataloader, 0, args)
            model.save_weights_numpy(fname="SLM_MNIST")
            return
    if args.vis:
        prop_vis(model, val_dataloader, 0, args)
        return
    else:
        if os.path.exists(args.result_record_path):
            os.remove(args.result_record_path)
        else:
            with open(args.result_record_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ['Epoch', 'Train_Loss', "Train_Acc", 'Val_Loss', "Val_Acc", "LR"])
    lambda1= args.lambda1
    
    if args.evaluation:
        eval(model, val_dataloader, 0, args)
        return
    else:
        start_time = time()
        train(model, train_dataloader, val_dataloader,  lambda1, args)
        print('run time', time()-start_time)
        return
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=350)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default="mnist", help='define train/test dataset (mnist, cifar10, cifar100)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--depth', type=int, default=4, help='number of fourier optic transformations/num of layers')
    parser.add_argument('--whether-load-model', type=bool, default=False, help="load pre-train model")
    parser.add_argument('--get-phase', type=bool, default=False, help="load pre-train model and extra phase parameters")
    parser.add_argument('--save-w', type=bool, default=False, help="save voltage parameters for SLM deployment")
    parser.add_argument('--evaluation', type=bool, default=False, help="Evaluation only")
    parser.add_argument('--start-epoch', type=int, default=0, help='load pre-train model at which epoch')
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="save training result.")
    parser.add_argument('--lambda1', type=float, default=1, help="loss weight for the model.")
    parser.add_argument('--phase-file', type=str, default='./device_parameters/phase.csv', help="the experimental data collected for phase function.")
    parser.add_argument('--intensity-file', type=str, default='./device_parameters/intensity.csv', help="the experimental data collected for phase function.")
    parser.add_argument('--use-batch-norm', type=bool, default=False, help="use BN layer in modulation")
    parser.add_argument('--vis', type=bool, default=False, help="")
    parser.add_argument('--sys-size', type=int, default=200, help='system size (dim of each diffractive layer)')
    parser.add_argument('--distance', type=float, default=0.6604, help='layer distance (default=0.1 meter)')
    parser.add_argument('--precision', type=int, default=8, help='precision (# bits) of the phase/intensity of given HW (e.g., 2**8 intervals)')
    parser.add_argument('--amp-factor', type=float, default=6, help='regularization factors to balance phase-amplitude where they share same downstream graidents')
    parser.add_argument('--pixel-size', type=float, default=0.000036, help='the size of pixel in diffractive layers')
    parser.add_argument('--pad', type=int, default=100, help='the padding size ')
    parser.add_argument('--Fresnel', type=bool, default=False, help="Use Fresnel Approximation, otherwise Sommerfeld Approximation.")
    parser.add_argument('--Fraunhofer', type=bool, default=False, help="Use Fraunhofer Approximation, otherwise Sommerfeld Approximation.")
    parser.add_argument('--wavelength', type=float, default=5.32e-7, help='wavelength')

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)



"""
# training example
CUDA_VISIBLE_DEVICES=0 python tutorial_02_codesign.py --sys-size 100 --depth 2 --amp-factor 50 --lr 0.5 --pad 50 --epochs 15 --batch-size 800

# plot prop of each layer for the first 32 image in the dataset
CUDA_VISIBLE_DEVICES=0 python tutorial_02_codesign.py --sys-size 100 --depth 2 --whether-load-model True --start-epoch 15 --vis True --pad 50 --batch-size 32
"""
