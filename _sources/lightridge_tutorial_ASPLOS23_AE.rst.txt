Welcome to LightRidge — An Open-Source Hardware Project for Optical AI
======================================================================

The following tutorial is a LightRidge tutorial of building a basic
diffractive optical neural networks (DONNs)

NOTE: This colab code is tested with free colab runtime for simple Artifact Evaluation (ASPLOS'24) . For better runtime, please use your own GPU server.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: 

    !pip install lightridge


.. parsed-literal::

    Requirement already satisfied: lightridge in /usr/local/lib/python3.10/dist-packages (0.2.1)
    Requirement already satisfied: torch>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from lightridge) (2.0.1+cu118)
    Requirement already satisfied: torchvision>=0.13.0 in /usr/local/lib/python3.10/dist-packages (from lightridge) (0.15.2+cu118)
    Requirement already satisfied: setuptools>=42 in /usr/local/lib/python3.10/dist-packages (from lightridge) (67.7.2)
    Requirement already satisfied: lightpipes in /usr/local/lib/python3.10/dist-packages (from lightridge) (2.1.4)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->lightridge) (3.12.2)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->lightridge) (4.5.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->lightridge) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->lightridge) (3.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->lightridge) (3.1.2)
    Requirement already satisfied: triton==2.0.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.12.0->lightridge) (2.0.0)
    Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.12.0->lightridge) (3.27.4.1)
    Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch>=1.12.0->lightridge) (16.0.6)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision>=0.13.0->lightridge) (1.23.5)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision>=0.13.0->lightridge) (2.31.0)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision>=0.13.0->lightridge) (9.4.0)
    Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from lightpipes->lightridge) (1.11.2)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from lightpipes->lightridge) (3.7.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.12.0->lightridge) (2.1.3)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->lightpipes->lightridge) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->lightpipes->lightridge) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->lightpipes->lightridge) (4.42.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->lightpipes->lightridge) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->lightpipes->lightridge) (23.1)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->lightpipes->lightridge) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->lightpipes->lightridge) (2.8.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.13.0->lightridge) (3.2.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.13.0->lightridge) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.13.0->lightridge) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.13.0->lightridge) (2023.7.22)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.12.0->lightridge) (1.3.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib->lightpipes->lightridge) (1.16.0)


Step 1: LightRidge installation
-------------------------------

.. raw:: html

   <h1>

LightRidge Design Flow

.. raw:: html

   </h1>


   <p>

.. raw:: html

   </p>

More can be found at https://lightridge.github.io/lightridge/index.html#

Step 2: Check LightRidge installation
-------------------------------------

.. code:: 

    import lightridge
    import lightridge.layers as layers
    import lightridge.utils as utils
    import lightridge.data as dataset
    from   lightridge.get_h import _field_Fresnel


.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/lightridge/get_h.py:27: UserWarning: 
    **************************** WARNING ***********************
    LightPipes: Cannot import pyFFTW, falling back to numpy.fft.
    (Try to) install pyFFTW on your computer for faster performance.
    Enter at a terminal prompt: python -m pip install pyfftw.
    Or reinstall LightPipes with the option pyfftw
    Enter: python -m pip install lightpipes[pyfftw]
    
    You can suppress warnings by using the -Wignore option:
    Enter: python _Wignore *****.py
    *************************************************************
      warnings.warn(_WARNING)


GPU Details
~~~~~~~~~~~

The GPU details can be accessed by ``!nvidia-smi``.

.. code:: 

    !nvidia-smi


.. parsed-literal::

    Mon Oct  2 19:31:19 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   68C    P8    11W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


Step 3: Load additional packages and configure your training device
-------------------------------------------------------------------

.. code:: 

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
    device='cuda:0'

.. code:: 

    from platform import python_version
    print("Python version", python_version())
    print("Pytorch - version", torch.__version__)
    print("Pytorch - cuDNN version :", torch.backends.cudnn.version())


.. parsed-literal::

    Python version 3.10.12
    Pytorch - version 2.0.1+cu118
    Pytorch - cuDNN version : 8700


Step 4: Constructing DONNs
--------------------------

The DONN model is constructed here. With defined diffractive layers and
parameters, the model class works as a sequential container that stacks
arbitrary numbers of customized diffractive layers in the order of light
propagation in the DONN system and a detector plane. As a result, we
construct a complete DONN system just like constructing a conventional
neural network.

.. figure:: https://lightridge.github.io/lightridge/_images/sci18.jpg
   :alt: DONN Overview

   DONN Overview

1. Set the hardware information for the emulation

-  Input laser source information: *wavelength* (532e-9)
-  System size: *sys\_size* (200)
-  Pixel size: *pixel\_size* (3.6e-5)
-  Padding size for emulations: *pad* (100)

2. Define the system parameters for model construction

-  Diffractive layers: inlcudes diffraction approximation and phase
   modulation in *layers.DiffractLayer\_Raw/layers.DiffractLayer*:
-  Mathematical approximation for light diffraction: *approx*
-  Diffraction distance: *distance*
-  System depth: *num\_layers*
-  Training regularization: *amp\_factor*

-  Detector design: defined in *layers.Detector*:
-  Location coordinate for detector regions: *det\_loc\_x, det\_loc\_y*,
-  Size for each sub-detector: *det\_size*

3. Visualization functions

-  Propogation pattern visualization: *prop\_view*
-  Trainable parameter, phase modulation visualization: *phase\_view*

.. code:: 

    class DiffractiveClassifier_Raw(torch.nn.Module):
        def __init__(self, device, det_x_loc, det_y_loc, det_size, wavelength=5.32e-7, pixel_size=0.000036,
                     batch_norm=False, sys_size = 200, pad = 100, distance=0.1, num_layers=2, amp_factor=6, approx="Fresnel3"):
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
            utils.phase_visualization(phase_list,size=self.size, cmap=cmap, fname="prop_view_reflection.pdf")
            return

Step 5: Training DONNs
----------------------

The fully differentiable DONN system can use conventional
backpropagation engine to optimize.

.. code:: 

    def train(model,train_dataloader, val_dataloader, epochs, lr):
        criterion = torch.nn.MSELoss(reduction='sum').to(device)
        print('training starts.')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20, gamma=0.5)
    
        for epoch in range(epochs):
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
    
                tk0.set_description_str('Epoch {}/{} : Training'.format(epoch, epochs))
                tk0.set_postfix({'Train_Loss': '{:.2f}'.format(train_loss), 'Train_Accuracy': '{:.5f}'.format(train_accuracy)})
            scheduler.step()
            val_loss, val_accuracy = eval(model, val_dataloader, epoch)
    
    
        return train_loss, train_accuracy, val_loss, val_accuracy, log

.. code:: 

    def eval(model, val_dataloader, epoch):
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

.. code:: 

    # Parameters define
    batch_size = 500
    sys_size = 200
    distance = 0.3
    pixel_size = 3.6e-5
    pad = 100
    wavelength = 5.32e-7
    approx = 'Fresnel'
    amp_factor = 1.5
    depth = 5
    device = "cuda:0"
    epochs = 10
    lr = 0.1
    det_x_loc = [40, 40, 40, 90, 90, 90, 90, 140, 140, 140]
    det_y_loc = [40, 90, 140, 30, 70, 110, 150, 40, 90, 140]
    det_size = 20

.. code:: 

    # dataset loader
    load_dataset = dataset.load_dataset(batch_size = batch_size, system_size = sys_size, datapath = "./data")
    train_dataloader, val_dataloader = load_dataset.MNIST()
    #train_dataloader, val_dataloader = load_dataset.FMNIST()
    
    # model construction
    model = DiffractiveClassifier_Raw(num_layers=depth, batch_norm =False,device=device,
                            #det_y_loc = [105, 125, 145, 95, 115, 135, 155, 105, 125, 145], #det_y_loc = [175,195,215,165,185,205,225,175,195,215],
                            #det_x_loc = [105, 105, 105, 125, 125, 125, 125, 145, 145, 145], #, det_x_loc = [175,175,175,195,195,195,195,215,215,215],
                            #det_size = 10,
                            det_x_loc = det_x_loc,
                            det_y_loc = det_y_loc,
                            det_size = det_size,
                            wavelength=wavelength, pixel_size = pixel_size, sys_size=sys_size, pad = pad,
                            distance=distance,amp_factor=amp_factor, approx=approx)
    model.to(device)
    
    # mode training
    train(model, train_dataloader, val_dataloader, epochs, lr)


.. parsed-literal::

    Network is constructed using Fresnel approximation
    Network is constructed using Fresnel approximation
    Network is constructed using Fresnel approximation


.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(


.. parsed-literal::

    Network is constructed using Fresnel approximation
    Network is constructed using Fresnel approximation
    Network is constructed using Fresnel approximation
    training starts.


.. parsed-literal::

    Epoch 0/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:18<00:00,  1.52it/s, Train_Loss=0.27, Train_Accuracy=0.85233]
    Validating: 100%|███████████| 20/20 [00:10<00:00,  1.90it/s, Val_Loss=0.17748, Val_Accuarcy=0.91400]
    Epoch 1/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:15<00:00,  1.59it/s, Train_Loss=0.18, Train_Accuracy=0.91192]
    Validating: 100%|███████████| 20/20 [00:09<00:00,  2.12it/s, Val_Loss=0.16505, Val_Accuarcy=0.91790]
    Epoch 2/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:15<00:00,  1.59it/s, Train_Loss=0.17, Train_Accuracy=0.91912]
    Validating: 100%|███████████| 20/20 [00:09<00:00,  2.06it/s, Val_Loss=0.15756, Val_Accuarcy=0.92510]
    Epoch 3/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:14<00:00,  1.61it/s, Train_Loss=0.16, Train_Accuracy=0.92135]
    Validating: 100%|███████████| 20/20 [00:09<00:00,  2.09it/s, Val_Loss=0.15753, Val_Accuarcy=0.92060]
    Epoch 4/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:15<00:00,  1.59it/s, Train_Loss=0.16, Train_Accuracy=0.92245]
    Validating: 100%|███████████| 20/20 [00:08<00:00,  2.25it/s, Val_Loss=0.15468, Val_Accuarcy=0.92590]
    Epoch 5/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:15<00:00,  1.59it/s, Train_Loss=0.16, Train_Accuracy=0.92373]
    Validating: 100%|███████████| 20/20 [00:09<00:00,  2.09it/s, Val_Loss=0.15384, Val_Accuarcy=0.92400]
    Epoch 6/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:14<00:00,  1.61it/s, Train_Loss=0.16, Train_Accuracy=0.92553]
    Validating: 100%|███████████| 20/20 [00:09<00:00,  2.02it/s, Val_Loss=0.14986, Val_Accuarcy=0.93020]
    Epoch 7/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:17<00:00,  1.55it/s, Train_Loss=0.16, Train_Accuracy=0.92615]
    Validating: 100%|███████████| 20/20 [00:09<00:00,  2.13it/s, Val_Loss=0.15055, Val_Accuarcy=0.92710]
    Epoch 8/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:15<00:00,  1.60it/s, Train_Loss=0.16, Train_Accuracy=0.92498]
    Validating: 100%|███████████| 20/20 [00:10<00:00,  1.95it/s, Val_Loss=0.15229, Val_Accuarcy=0.92690]
    Epoch 9/10 : Training: 100%|███████████████████████████████████████████████| 120/120 [01:14<00:00,  1.61it/s, Train_Loss=0.16, Train_Accuracy=0.92597]
    Validating: 100%|███████████| 20/20 [00:09<00:00,  2.21it/s, Val_Loss=0.15106, Val_Accuarcy=0.92790]




.. parsed-literal::

    (0.1554327927907308,
     tensor(0.9260, device='cuda:0'),
     0.1510582004547119,
     tensor(0.9279, device='cuda:0'),
     [])



**Note: We only showcase 10 epochs in this tutorial example to save the
time. To reproduce the full results, you will need 100 epochs as
reported in our paper.**

Step 6: Visualizations
----------------------

.. code:: 

    # Visualization for phase modulation
    model.phase_view(cmap="twilight")


.. parsed-literal::

    torch.Size([200, 200])



.. image:: lightridge_tutorial_ASPLOS23_AE_files/lightridge_tutorial_ASPLOS23_AE_20_1.png


.. code:: 

    # Visualization for propagation
    transform = transforms.Compose([transforms.Resize((sys_size),interpolation=2),transforms.ToTensor()])
    val_dataset = torchvision.datasets.MNIST("./data/", train=False, transform=transform, download=True)
    
    with torch.no_grad():
      model.eval()
      # feel free to add more round of inference test to see the magic of DONNs in Tasks!
      # You just need to replace the index of the val_dataset vector.
      # example round 1
      val_img, val_label =val_dataset[0]
      model.prop_view(val_img.to(device))
      # example round 2
      val_img, val_label =val_dataset[100]
      model.prop_view(val_img.to(device))


.. parsed-literal::

    0
    7 torch.Size([200, 200])
    0
    7 torch.Size([200, 200])



.. image:: lightridge_tutorial_ASPLOS23_AE_files/lightridge_tutorial_ASPLOS23_AE_21_1.png



.. image:: lightridge_tutorial_ASPLOS23_AE_files/lightridge_tutorial_ASPLOS23_AE_21_2.png



.. image:: lightridge_tutorial_ASPLOS23_AE_files/lightridge_tutorial_ASPLOS23_AE_21_3.png



.. image:: lightridge_tutorial_ASPLOS23_AE_files/lightridge_tutorial_ASPLOS23_AE_21_4.png


Step7: Change the DONN model with codesign information
------------------------------------------------------

The measured quantization vector w.r.t the SLM is stored in folder
*device\_parameters* including the phase measurements, i.e., phase
modulation vs applied voltage stage, in *phase.csv*, and the intensity
measurements, i.e., intensity modulation vs applied voltage stage, in
*intensity.csv*.

.. code:: 

    class DiffractiveClassifier_Codesign(torch.nn.Module):
        def __init__(self, phase_func, intensity_func, device, det_x_loc, det_y_loc, det_size, wavelength=5.32e-7, pixel_size=0.000036,
                     batch_norm=False, sys_size = 200, pad = 100, distance=0.1, num_layers=2, precision=256, amp_factor=6, approx="Fresnel3"):
            super(DiffractiveClassifier_Codesign, self).__init__()
            self.amp_factor = amp_factor
            self.size = sys_size
            self.distance = distance
            self.wavelength = wavelength
            self.pixel_size = pixel_size
            self.pad = pad
            self.approx=approx
            self.phase_func = phase_func.to(device)
            self.intensity_func = intensity_func.to(device)
            self.precision = precision
            self.diffractive_layers = torch.nn.ModuleList([layers.DiffractLayer(self.phase_func, self.intensity_func, wavelength=self.wavelength, pixel_size=self.pixel_size,
                                                                                size=self.size, pad = self.pad, distance=self.distance, precision=self.precision,
                                                                                amplitude_factor=amp_factor, approx=self.approx, phase_mod=True) for _ in range(num_layers)])
            self.last_diffraction = layers.DiffractLayer(self.phase_func, self.intensity_func, wavelength=self.wavelength, pixel_size=self.pixel_size,
                                                                size=self.size, pad = self.pad, distance=self.distance, precision=self.precision,
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
                phase_list.append(torch.argmax(torch.nn.functional.gumbel_softmax(layer.voltage,tau=1,hard=True).cpu(), dim=-1))
            print(phase_list[0].shape)
            utils.phase_visualization(phase_list,size=self.size, cmap=cmap, fname="prop_view_reflection.pdf")
            return

Step 8: Add the device parameters.
----------------------------------

Download the SLM *device\_parameters* measured from our own setups. You
can just replace this part with your own hardware systems.

.. code:: 

    # Parameters define
    batch_size = 100
    sys_size = 200
    distance = 0.3
    pixel_size = 3.6e-5
    pad = 100
    approx = 'Fresnel3'
    amp_factor = 7
    depth = 5
    device = "cuda:0"
    epochs = 3
    lr = 0.3
    precision = 8


.. code:: 

    !mkdir device_parameters
    !wget https://lightridge.github.io/lightridge/ASPLOS23_AE/intensity.csv
    !wget https://lightridge.github.io/lightridge/ASPLOS23_AE/phase.csv
    !mv intensity.csv device_parameters/
    !mv phase.csv device_parameters/
    
    phase_file =  "./device_parameters/phase.csv"
    phase_function = utils.phase_func(phase_file,  i_k=precision)
    #with open('phase_file.npy', 'wb') as f_phase:
    #    np.save(f_phase, phase_function.cpu().numpy())
    
    intensity_file =  "./device_parameters/intensity.csv"
    intensity_function = utils.intensity_func(intensity_file,  i_k=precision)
    #with open('intensity_file.npy', 'wb') as f_amp:
    #    np.save(f_amp, intensity_function.cpu().numpy())



.. parsed-literal::

    mkdir: cannot create directory ‘device_parameters’: File exists
    --2023-10-02 19:58:08--  https://lightridge.github.io/lightridge/ASPLOS23_AE/intensity.csv
    Resolving lightridge.github.io (lightridge.github.io)... 185.199.108.153, 185.199.109.153, 185.199.110.153, ...
    Connecting to lightridge.github.io (lightridge.github.io)|185.199.108.153|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1660 (1.6K) [text/csv]
    Saving to: ‘intensity.csv’
    
    intensity.csv       100%[===================>]   1.62K  --.-KB/s    in 0s      
    
    2023-10-02 19:58:08 (29.2 MB/s) - ‘intensity.csv’ saved [1660/1660]
    
    --2023-10-02 19:58:09--  https://lightridge.github.io/lightridge/ASPLOS23_AE/phase.csv
    Resolving lightridge.github.io (lightridge.github.io)... 185.199.108.153, 185.199.109.153, 185.199.110.153, ...
    Connecting to lightridge.github.io (lightridge.github.io)|185.199.108.153|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1596 (1.6K) [text/csv]
    Saving to: ‘phase.csv’
    
    phase.csv           100%[===================>]   1.56K  --.-KB/s    in 0s      
    
    2023-10-02 19:58:09 (31.7 MB/s) - ‘phase.csv’ saved [1596/1596]
    


.. code:: 

    # dataset loader
    load_dataset = dataset.load_dataset(batch_size = batch_size, system_size = sys_size, datapath = "./data")
    train_dataloader, val_dataloader = load_dataset.MNIST()
    #train_dataloader, val_dataloader = load_dataset.FMNIST()
    
    # model construction
    model = DiffractiveClassifier_Codesign(num_layers=depth, batch_norm =False,device=device,
                            #det_y_loc = [105, 125, 145, 95, 115, 135, 155, 105, 125, 145], #det_y_loc = [175,195,215,165,185,205,225,175,195,215],
                            #det_x_loc = [105, 105, 105, 125, 125, 125, 125, 145, 145, 145], #, det_x_loc = [175,175,175,195,195,195,195,215,215,215],
                            #det_size = 10,
                            det_x_loc = [40, 40, 40, 90, 90, 90, 90, 140, 140, 140],
                            det_y_loc = [40, 90, 140, 30, 70, 110, 150, 40, 90, 140],
                            det_size = 20, precision=precision, phase_func=phase_function, intensity_func=intensity_function,
                            wavelength=15.5e-7, pixel_size = pixel_size, sys_size=sys_size, pad = pad,
                            distance=distance,amp_factor=amp_factor, approx=approx)
    model.to(device)
    
    # mode training
    train(model, train_dataloader, val_dataloader, epochs, lr)


.. parsed-literal::

    /usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:560: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(_create_warning_msg(
    /usr/local/lib/python3.10/dist-packages/lightridge/layers.py:314: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ../torch/csrc/utils/tensor_new.cpp:245.)
      return_in_outK = torch.complex(torch.tensor([return_in_outK.real.astype('float32')]), torch.tensor([return_in_outK.imag.astype('float32')]))


.. parsed-literal::

    torch.complex64
    torch.complex64
    torch.complex64
    torch.complex64
    torch.complex64
    torch.complex64
    training starts.


.. parsed-literal::

    Epoch 0/3 : Training: 100%|████████████████████████████████████████████████| 600/600 [32:29<00:00,  3.25s/it, Train_Loss=0.55, Train_Accuracy=0.62268]
    Validating: 100%|█████████| 100/100 [03:52<00:00,  2.32s/it, Val_Loss=0.15074, Val_Accuarcy=0.90020]
    Epoch 1/3 : Training: 100%|████████████████████████████████████████████████| 600/600 [32:21<00:00,  3.24s/it, Train_Loss=0.13, Train_Accuracy=0.91645]
    Validating: 100%|█████████| 100/100 [03:51<00:00,  2.32s/it, Val_Loss=0.10463, Val_Accuarcy=0.93270]
    Epoch 2/3 : Training:  44%|█████████████████████▎                          | 266/600 [14:27<18:11,  3.27s/it, Train_Loss=0.10, Train_Accuracy=0.93714]

.. code:: 

    # Visualization for phase modulation
    model.phase_view(cmap="hsv")

.. code:: 

    # Visualization for propagation
    transform = transforms.Compose([transforms.Resize((sys_size),interpolation=2),transforms.ToTensor()])
    val_dataset = torchvision.datasets.MNIST("./data/", train=False, transform=transform, download=True)
    
    with torch.no_grad():
      model.eval()
      val_img, val_label =val_dataset[900]
      model.prop_view(val_img.to(device))
