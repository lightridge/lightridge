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
## reflection-aware diffractive neural networks design
## layer API : layers.DiffractiveLayerRaw_Reflect 
device='cpu'
class Net(torch.nn.Module):
    def __init__(self, sys_size = 200, distance=0.04, num_layers=2,relist=[0.0,1/3], 
			pixel_size = 0.0008, wavelength = 7.5e-4, amp_factor=40.0):
        super(Net, self).__init__()
        self.amp_factor = amp_factor
        self.size = sys_size
        self.distance = distance
        self.wavelength= wavelength
        self.pixel_size= pixel_size
        #pixel_size = self.pixel_size
        print(relist)
        assert(num_layers == len(relist)), "Number of D2NN layers has to match the reflection index list"
        self.diffractive_layers = torch.nn.ModuleList(
            [layers.DiffractiveLayerRaw_Reflect(pixel_size = self.pixel_size, size=self.size, distance=self.distance,
                                              rE=relist[i],amplitude_factor=1, phase_mod=True, wavelength=self.wavelength) for i in range(num_layers)])
        self.last_diffraction = layers.DiffractiveLayerRaw_Reflect(size=self.size, distance=self.distance, rE=0.0,
                                                                   amplitude_factor=self.amp_factor,phase_mod=False)
        # 200 by 200 system siz det designe
        #self.detector = layers.Detector(start_x = [46,46,46], start_y = [46,46,46], det_size = 20,
        #                                gap_x = [19,20], gap_y = [27, 12, 27])
        ratio = sys_size/200.0
        x1 = int(46*ratio)
        y1 = x1
        ds = int(20*ratio)
        gapx1 = int(20*ratio)
        gapy1 = int(27*ratio)
        gapy2 = int(12*ratio)
        self.detector = layers.Detector_10(start_x=[x1, x1, x1], start_y=[y1, y1, y1], det_size=ds,
                                        gap_x=[gapx1, gapx1], gap_y=[gapy1, gapy2, gapy1])
        #self.detector = layers.Detector(x_loc=[92, 92, 92, 170, 170, 170, 170, 250, 250, 250],
	#				y_loc=[92, 186, 280, 92, 156, 220, 284, 92, 186, 280], size=sys_size, det_size=20*ratio)
    def forward(self, x):
        x = x * self.amp_factor
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
        x = self.last_diffraction(x)
        output = self.detector(x)
        return output
    def phase_view(self):
        phase_list = []
        for index, layer in enumerate(self.diffractive_layers):
            phase_list.append(layer.phase)
        utils.phase_visualization(phase_list,size=self.size, cmap="hsv", fname="thz_phase.pdf")
def train(model,train_dataloader, val_dataloader, input_padding, lambda1, args):        
    criterion = torch.nn.MSELoss(reduction='sum')#.cuda()
    print('training starts.')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.5)
    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.epochs):
        log = [epoch]
        model.train()
        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0
        tk0 = tqdm(train_dataloader, ncols=100, total=int(len(train_dataloader)))
        for train_iter, train_data_batch in enumerate(tk0):
            train_images, train_labels = utils.data_to_cplex(train_data_batch, device=device)
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
    criterion = torch.nn.MSELoss(reduction='sum')#.cuda()
    with torch.no_grad():
        model.eval()
        val_len = 0.0
        val_running_counter = 0.0
        val_running_loss = 0.0

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

            tk1.set_description_str('Epoch {}/{} : Validating'.format(epoch, args.start_epoch + 1 + args.epochs - 1 ))
            tk1.set_postfix({'Val_Loss': '{:.5f}'.format(val_loss), 'Val_Accuarcy': '{:.5f}'.format(val_accuracy)})
    return val_loss, val_accuracy
   
def main(args):
    torch.autograd.set_detect_anomaly(True)
    #if args.inference:
        #eval(model, args)
        #return
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.Resize((args.sys_size),interpolation=2),transforms.ToTensor()])
        print("training and testing on MNIST10 dataset")
        train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
        val_dataset = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, pin_memory=True)
        input_padding = 0
    elif args.dataset == "Fmnist":
        transform = transforms.Compose([transforms.Resize((args.sys_size),interpolation=2),transforms.ToTensor()])
        print("training and testing on FashionMNIST10 dataset")
        train_dataset = torchvision.datasets.FashionMNIST("./Fdata", train=True, transform=transform, download=True)
        val_dataset = torchvision.datasets.FashionMNIST("./Fdata", train=False, transform=transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        input_padding = 0
    elif args.dataset == "Kmnist":
        transform = transforms.Compose([transforms.Resize((args.sys_size),interpolation=2),transforms.ToTensor()])
        print("training and testing on FashionMNIST10 dataset")
        train_dataset = torchvision.datasets.FashionMNIST("./Kdata", train=True, transform=transform, download=True)
        val_dataset = torchvision.datasets.FashionMNIST("./Kdata", train=False, transform=transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
        input_padding = 0
    ## initialization of reflection list to match the depth
    relist = [0] + [args.re]*(args.depth-1)
    model = Net(sys_size=args.sys_size, distance=args.distance, relist=relist, 
			num_layers=args.depth, pixel_size = args.pixel_size, wavelength = args.wl, amp_factor=args.amp_factor)
    model.to(device)
    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) +  args.model_name))
        print('Model1 : "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
        model.phase_view()
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
        train(model, train_dataloader, val_dataloader, input_padding, lambda1, args)
        print('run time', time()-start_time)
        return
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default="mnist", help='define train/test dataset (mnist, cifar10, cifar100)')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--depth', type=int, default=2, help='number of fourier optic transformations/num of layers')
    parser.add_argument('--whether-load-model', type=bool, default=False, help="load pre-train model")
    parser.add_argument('--evaluation', type=bool, default=False, help="Evaluation only")
    parser.add_argument('--start-epoch', type=int, default=0, help='load pre-train model at which epoch')
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="save training result.")
    parser.add_argument('--lambda1', type=float, default=1, help="loss weight for the model.")
    parser.add_argument('--sys-size', type=int, default=400, help='system size (dim of each diffractive layer)')
    parser.add_argument('--distance', type=float, default=0.08, help='layer distance (default=0.1 meter)')
    parser.add_argument('--precision', type=int, default=20, help='precision (# bits) of the phase/intensity of given HW (e.g., 2**8 intervals)')
    parser.add_argument('--amp-factor', type=float, default=60.0, help='regularization factors to balance phase-amplitude where they share same downstream graidents')
    parser.add_argument('--re', type=float, default=1/3, help='reflection index')
    parser.add_argument('--wl', type=float, default=7.5e-4, help='wavelength of the laser source')
    parser.add_argument('--pixel-size', type=float, default=0.0004, help='reflection index')

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)
