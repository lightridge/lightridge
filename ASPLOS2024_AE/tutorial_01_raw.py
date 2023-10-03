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
import lightridge.layers as layers
import lightridge.utils as utils
import lightridge.data as dataset
from lightridge.get_h import _field_Fresnel
device='cuda:0'

class DiffractiveClassifier_Raw(torch.nn.Module):
    def __init__(self, device, det_x_loc, det_y_loc, det_size, 
                 wavelength=15.32e-7, pixel_size=0.000036, 
                 batch_norm=False, sys_size = 200, pad = 100, 
                 distance=0.1, num_layers=2, amp_factor=6, 
                 approx="Fresnel3"):
        super(DiffractiveClassifier_Raw, self).__init__()
        self.amp_factor = amp_factor
        self.size = sys_size
        self.distance = distance
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.pad = pad
        self.approx=approx
        self.diffractive_layers = torch.nn.ModuleList([layers.DiffractLayer_Raw(wavelength=self.wavelength, pixel_size=self.pixel_size,
                                                                                    size=self.size, pad = self.pad, distance=self.distance, 
                                                                                    amplitude_factor = amp_factor, approx=self.approx, 
                                                                                    phase_mod=True) for _ in range(num_layers)])
        self.last_diffraction = layers.DiffractLayer_Raw(wavelength=self.wavelength, pixel_size=self.pixel_size,
                                                            size=self.size, pad = self.pad, distance=self.distance, 
                                                            approx=self.approx, phase_mod=False)
        self.detector = layers.Detector(x_loc=det_x_loc, y_loc=det_y_loc, det_size=det_size, size=self.size)

    def forward(self, x):
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
        x = self.last_diffraction(x)
        output = self.detector(x)
        return output

    def prop_view(self, x):
        prop_list = []
        prop_list.append(x)
        x = x #* self.amp_factor
        for index, layer in enumerate(self.diffractive_layers):
            x = layer(x)
            prop_list.append(x)
        x = self.last_diffraction(x)
        prop_list.append(x)
        for i in range(x.shape[0]):
            print(i)
            utils.forward_func_visualization(prop_list, self.size, fname="mnist_%s.pdf" % i, idx=i, intensity_plot=False)
        output = self.detector(x)
        return

    def phase_view(self, cmap="hsv"):
        phase_list = []
        for index, layer in enumerate(self.diffractive_layers):
            phase_list.append(layer.phase)
        print(phase_list[0].shape)
        utils.phase_visualization(phase_list,size=self.size, cmap=cmap, fname="phase_view_raw.pdf")
        return

def prop_vis(model, val_dataset, vis_list):
    img_list = []
    with torch.no_grad():
        model.eval()
        for i in range(len(vis_list)):
            val_images, val_labels = val_dataset[vis_list[i]]
            img_list.append(val_images.to(device))
        img_set = torch.stack(img_list)
        model.prop_view(img_set)
    return 

def train(model,train_dataloader, val_dataloader, epochs, lr, args):        
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    print('training starts.')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.5)
    
    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + epochs):
        log = []
        model.train()
        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0
        tk0 = tqdm(train_dataloader, ncols=150, total=int(len(train_dataloader)))
        for train_iter, train_data_batch in enumerate(tk0):
            train_images, train_labels = utils.data_to_cplex(train_data_batch, device='cuda:0')
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

            tk0.set_description_str('Epoch {}/{} : Training'.format(epoch, args.start_epoch + 1 + epochs - 1))
            tk0.set_postfix({'Train_Loss': '{:.2f}'.format(train_loss), 'Train_Accuracy': '{:.5f}'.format(train_accuracy)})
        scheduler.step()
        log.append(train_loss)
        log.append(train_accuracy.cpu())
        torch.save(model.state_dict(), (args.model_save_path + str(epoch) + args.model_name))
        print('Model : "' + args.model_save_path + str(epoch) + args.model_name + '" saved.')

        val_loss, val_accuracy = eval(model, val_dataloader)
        log.append(val_loss)
        log.append(val_accuracy.cpu())
        log_arr = np.array(log).reshape(1, 4)
        #f = open('raw_train_' + str(args.depth) + 'layer_' + str(args.amp_factor) + '_ampfactor' + '.csv', 'ab')
        f = open(args.result_record_path, 'ab')
        np.savetxt(f, log_arr, fmt='%.4f')
        f.close()

    return train_loss, train_accuracy, val_loss, val_accuracy, log 

def eval(model, val_dataloader):
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

            tk1.set_description_str('Validating')
            tk1.set_postfix({'Val_Loss': '{:.5f}'.format(val_loss), 'Val_Accuarcy': '{:.5f}'.format(val_accuracy)})
    return val_loss, val_accuracy
   
def main(args):
    torch.autograd.set_detect_anomaly(True)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    if args.dataset == "mnist":
        print("training and testing on MNIST10 dataset")
        #load_dataset = dataset.load_dataset(batch_size = args.batch_size, system_size = args.sys_size, datapath = "./data")
        #train_dataloader, val_dataloader = load_dataset.MNIST()
        transform = transforms.Compose([transforms.Resize((args.sys_size),interpolation=2),transforms.ToTensor()])
        train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
        val_dataset = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    elif args.dataset == "Fmnist":
        print("training and testing on FashionMNIST10 dataset")
        #load_dataset = dataset.load_dataset(batch_size = args.batch_size, system_size = args.sys_size, datapath = "./Fdata")
        #train_dataloader, val_dataloader = load_dataset.FMNIST()
        transform = transforms.Compose([transforms.Resize((sys_size),interpolation=2),transforms.ToTensor()])
        train_dataset = torchvision.datasets.FashionMNIST("./Fdata", train=True, transform=transform, download=True)
        val_dataset = torchvision.datasets.FashionMNIST("./Fdata", train=False, transform=transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    else:
        assert(0), "current version only supports MNIST10 and FashionMNIST10"

        
    model = DiffractiveClassifier_Raw(num_layers=args.depth, batch_norm =args.use_batch_norm,device=device, 
			#det_y_loc = [105, 125, 145, 95, 115, 135, 155, 105, 125, 145], #det_y_loc = [175,195,215,165,185,205,225,175,195,215],
                        #det_x_loc = [105, 105, 105, 125, 125, 125, 125, 145, 145, 145], #, det_x_loc = [175,175,175,195,195,195,195,215,215,215],
                        #det_size = 10,
                        #det_x_loc = [40, 40, 40, 90, 90, 90, 90, 140, 140, 140],
                        #det_y_loc = [40, 90, 140, 30, 70, 110, 150, 40, 90, 140],
                        #det_size = 20,
                        det_x_loc = args.det_x_loc, det_y_loc = args.det_y_loc, det_size = args.det_size,
                        wavelength=args.wavelength, pixel_size = args.pixel_size, sys_size=args.sys_size, pad = args.pad, 
                        distance=args.distance,amp_factor=args.amp_factor, approx=args.approx)
    model.to(device)

    
    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) +  args.model_name))
        print('Model1 : "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
    if args.prop_vis:
        prop_vis(model, val_dataset, args.vis_list)
        return
    if args.phase_vis:
        model.phase_view(cmap="hsv")
        return
    if os.path.exists(args.result_record_path):
        os.remove(args.result_record_path)
    
    if args.evaluation:
        print('evaluation only!')
        eval(model, val_dataloader)
        return
    else:
        print('training!')
        start_time = time()
        train(model, train_dataloader, val_dataloader, args.epochs, args.lr, args)
        print('run time', time()-start_time)
        return
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=350)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default="mnist", help='define train/test dataset (mnist, Fmnist).')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate.')
    parser.add_argument('--depth', type=int, default=5, help='number of fourier optic transformations/num of layers.')
    parser.add_argument('--whether-load-model', type=bool, default=False, help="load pre-train model.")
    parser.add_argument('--evaluation', type=bool, default=False, help="Evaluation only.")
    parser.add_argument('--start-epoch', type=int, default=0, help='load pre-train model at which epoch.')
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./save_model_raw/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="save training result.")
    parser.add_argument('--use-batch-norm', type=bool, default=False, help="use BN layer in modulation.")
    parser.add_argument('--prop_vis', type=bool, default=False, help="the visualization of the propagation in the model.")
    parser.add_argument('--phase_vis', type=bool, default=False, help="the visualization of the phase in the model.")
    parser.add_argument('--sys-size', type=int, default=200, help='system size (dim of each diffractive layer).')
    parser.add_argument('--distance', type=float, default=0.3, help='layer distance (default=0.3 meter).')
    parser.add_argument('--amp-factor', type=float, default=2, help='regularization factors to balance phase-amplitude where they share same downstream graidents.')
    parser.add_argument('--pixel-size', type=float, default=0.000036, help='the size of pixel in diffractive layers.')
    parser.add_argument('--pad', type=int, default=50, help='the padding size.')
    parser.add_argument('--approx', type=str, default='Fresnel', help="Use which Approximation, Sommerfeld, fresnel or fraunhofer.")
    parser.add_argument('--wavelength', type=float, default=5.32e-7, help='wavelength.')
    parser.add_argument('--det-size', type=int, default=20, help='The detector size.')
    parser.add_argument('--det-x-loc', nargs='+', type=int, default=[40, 40, 40, 90, 90, 90, 90, 140, 140, 140], help='the x-axis location of the detector region.')
    parser.add_argument('--det-y-loc', nargs='+', type=int, default=[40, 90, 140, 30, 70, 110, 150, 40, 90, 140], help='the y-axis location of the detector region.')
    parser.add_argument('--vis-list', nargs='+', type=int, default=[400, 600, 900, 1100], help='the input image sample for propagation visualization.')

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)


