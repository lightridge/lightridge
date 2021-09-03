import torch
import numpy as np
import math
import torch.nn.functional as F	

class Detector(torch.nn.Module):
    """ Implementation of detector plane for multi-task classification 

    The outputs are collected with specific defined detector regions over the entire light propogation.
    The outputs are (optional) normlized using functions such as softmax to enable effcient training of D2NNs.

    Args:
        >> detector plane design <<
        x: the x-axis location for your detector region (left-top)
        y: the y-axis location for your detector region (left-top)
        det_size: the size of the detector region
        size: the system size
        activation: activation function for training. default=torch.nn.Softmax(dim=-1)
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference: 
    """
    def __init__(self, x_loc, y_loc, det_size=20, size=200, activation = torch.nn.Softmax(dim=-1)):
        super(Detector, self).__init__()
        self.size = size
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.det_size = det_size
        self.activation = activation
    def forward(self, x):
        x = x.abs()
        assert len(self.x_loc)==len(self.y_loc) and len(self.x_loc) > 1, 'the input location information is wrong!'

        detectors = torch.cat((x[:, self.x_loc[0] : self.x_loc[0] + self.det_size, self.y_loc[0] : self.y_loc[0] + self.det_size].mean(dim=(1, 2)).unsqueeze(-1),
                               x[:, self.x_loc[1] : self.x_loc[1] + self.det_size, self.y_loc[1] : self.y_loc[1] + self.det_size].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)
        for i in range(2, len(self.x_loc)):
            detectors = torch.cat((detectors, x[:, self.x_loc[i] : self.x_loc[i] + self.det_size, self.y_loc[i] : self.y_loc[i] + self.det_size].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)

        assert self.x_loc[-1] + self.det_size < self.size and self.y_loc[-1] + self.det_size < self.size, 'the region is out of detector!'
        if self.activation == None:
            return detectors
        else:
            return self.activation(detectors)

	
class Detector_10(torch.nn.Module):
    """ Implementation of detector plane for multi-task classification 

    The outputs are collected with specific defined detector regions over the entire light propogation.
    The outputs are (optional) normlized using functions such as softmax to enable effcient training of D2NNs.

    Args:
        >> detector plane design <<
	start_x:
        start_y:
        det_size:
        gap_x:
        gap_y:
        size:
        activation: activation function for training. default=torch.nn.Softmax(dim=-1)
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference: 
    """
    def __init__(self, start_x, start_y, det_size, gap_x, gap_y, size=200, activation = torch.nn.Softmax(dim=-1)):
        super(Detector_10, self).__init__()
        self.size = size      
        self.start_x = start_x 
        self.start_y = start_y
        self.det_size = det_size
        self.gap_x = gap_x
        self.gap_y = gap_y
        self.activation = activation
    def forward(self, x):
        x = x.abs()
        detectors = torch.cat((
        x[:, self.start_x[0] : self.start_x[0]+self.det_size,                                 self.start_y[0] : self.start_y[0]+self.det_size].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[0] : self.start_x[0]+self.det_size,                                 self.start_y[0]+self.det_size+self.gap_y[0] : self.start_y[0]+self.gap_y[0]+self.det_size*2].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[0] : self.start_x[0]+self.det_size,                                 self.start_y[0]+self.det_size*2+self.gap_y[0]*2 : self.start_y[0]+self.det_size*3+self.gap_y[0]*2].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[1]+self.det_size+self.gap_x[0] : self.start_x[1]+self.det_size+self.gap_x[0]+self.det_size,     self.start_y[1] : self.start_y[1]+self.det_size].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[1]+self.det_size+self.gap_x[0] : self.start_x[1]+self.det_size+self.gap_x[0]+self.det_size,     self.start_y[1]+self.det_size+self.gap_y[1] : self.start_y[1]+self.gap_y[1]+self.det_size*2].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[1]+self.det_size+self.gap_x[0] : self.start_x[1]+self.det_size+self.gap_x[0]+self.det_size,     self.start_y[1]+self.det_size*2+self.gap_y[1]*2 : self.start_y[1]+self.det_size*3+self.gap_y[1]*2].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[1]+self.det_size+self.gap_x[0] : self.start_x[1]+self.det_size+self.gap_x[0]+self.det_size,     self.start_y[1]+self.det_size*3+self.gap_y[1]*3 : self.start_y[1]+self.det_size*4+self.gap_y[1]*3].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[2]+self.det_size*2+self.gap_x[0]+self.gap_x[1] : self.start_x[2]+self.det_size*3+self.gap_x[0]+self.gap_x[1],  self.start_y[2] : self.start_y[2]+self.det_size].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[2]+self.det_size*2+self.gap_x[0]+self.gap_x[1] : self.start_x[2]+self.det_size*3+self.gap_x[0]+self.gap_x[1],  self.start_y[2]+self.det_size+self.gap_y[2] : self.start_y[2]+self.gap_y[2]+self.det_size*2].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, self.start_x[2]+self.det_size*2+self.gap_x[0]+self.gap_x[1] : self.start_x[2]+self.det_size*3+self.gap_x[0]+self.gap_x[1],  self.start_y[2]+self.det_size*2+self.gap_y[2]*2 : self.start_y[2]+self.det_size*3+self.gap_y[2]*2].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)
        if self.activation == None:
            return detectors
        else:
            return self.activation(detectors)


class Diffraction(torch.nn.Module):
    """ Implementation of diffraction 

    Args:
	size: system size 
	distance: diffraction distance
	name: name of the layer
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference: 
    """
 
    def __init__(self, wavelength=5.32e-7, pixel_size=0.000036, size=200, distance=0.1, name="diffraction"): 
        super(DiffractiveLayer, self).__init__()
        self.size = size                         # 200 * 200 neurons in one layer
        self.distance = distance                    # distance bewteen two layers (3cm)
        self.ll = pixel_size * size                          # layer length (8cm)
        self.wl = wavelength                  # wave length
        self.fi = 1 / self.ll                   # frequency interval
        self.wn = 2 * 3.1415926 / self.wl       # wave number
        # self.phi (syssize, syssize)
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size), dtype=np.complex64)
        # h (syssize, syssize)
        h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
        # self.h (syssize, syssize, 2)
        self.h = torch.nn.Parameter(torch.view_as_complex(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1)), requires_grad=False)
        #self.h = torch.nn.Parameter(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1), requires_grad=False)
        # initialization with gumbel softmax (random one-hot encoding for voltage)
    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        x = torch.fft.ifft2( torch.fft.fft2(waves) * self.h )
        return x

class DiffractiveLayerRaw(torch.nn.Module):
    """ Implementation of diffractive layer without hardware constraints 

    Args:
	size: system size 
	distance: diffraction distance
	name: name of the layer
	amplitude_factor: training regularization factor w.r.t amplitude vs phase in backpropogation
	phase_mod: enable phase modulation or just diffraction. default: True
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference: 
    """
 
    def __init__(self, wavelength=5.32e-7, pixel_size=0.000036, size=200, distance=0.1, name="diffractive_layer_raw", amplitude_factor = 6, phase_mod=True):
        super(DiffractiveLayerRaw, self).__init__()
        self.size = size                         # 200 * 200 neurons in one layer
        self.distance = distance                    # distance bewteen two layers (3cm)
        self.ll = pixel_size * size                          # layer length (8cm)
        self.wl = wavelength                  # wave length
        self.fi = 1 / self.ll                   # frequency interval
        self.wn = 2 * 3.1415926 / self.wl       # wave number
        # self.phi (syssize, syssize)
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size), dtype=np.complex64)
        # h (syssize, syssize)
        h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
        # self.h (syssize, syssize, 2)
        self.h = torch.nn.Parameter(torch.view_as_complex(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1)), requires_grad=False)
        # phase parameter init
        self.phase = torch.nn.Parameter(torch.from_numpy( 2 * np.pi * torch.nn.init.xavier_uniform_(torch.empty(self.size,self.size)).numpy() ), requires_grad=True)
        self.register_parameter(name, self.phase)
        self.phase_model = phase_mod
        self.amplitude_factor = amplitude_factor
    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        temp = torch.fft.ifft2( torch.fft.fft2(waves) * self.h )
        if not self.phase_model:
            return temp
        exp_j_phase = torch.view_as_complex(torch.stack((self.amplitude_factor*torch.cos(self.phase), 
				self.amplitude_factor*torch.sin(self.phase)), dim=-1))
        x = temp * exp_j_phase
        return x

class DiffractiveLayer(torch.nn.Module):
    """ Implementation of diffractive layer that enables device-quantization aware training using Gumble-Softmax 

    Args:
	phase_func: phase space designed in a given hardware device, where the index represents the applying voltage/grayvalue (e.g., SLM)
	intensity_func: intensity space designed in a given hardware device, where the index represents the applying voltage/grayvalue (e.g., SLM)
	size: system size 
	distance: diffraction distance
	name: name of the layer
	precision: hardware precision encoded in number of possible values of phase or amplitude-phase. default: 256 (8-bit)
	amplitude_factor: training regularization factor w.r.t amplitude vs phase in backpropogation
	phase_mod: enable phase modulation or just diffraction. default: True
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference: 
    """
 
    def __init__(self, phase_func, intensity_func, wavelength=5.32e-7, pixel_size=0.000036, size=200, distance=0.1, name="diffractive_layer", 
				precision=256, amplitude_factor = 6, phase_mod=True):
        super(DiffractiveLayer, self).__init__()
        self.size = size                         # 200 * 200 neurons in one layer
        self.distance = distance                    # distance bewteen two layers (3cm)
        self.ll = pixel_size * size                          # layer length (8cm)
        self.wl = wavelength                  # wave length
        self.fi = 1 / self.ll                   # frequency interval
        self.wn = 2 * 3.1415926 / self.wl       # wave number
        # self.phi (syssize, syssize)
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size), dtype=np.complex64)
        # h (syssize, syssize)
        h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
        # self.h (syssize, syssize, 2)
        self.h = torch.nn.Parameter(torch.view_as_complex(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1)), requires_grad=False)
        # initialization with gumbel softmax (random one-hot encoding for voltage)
        #self.voltage = torch.nn.Parameter(torch.nn.functional.gumbel_softmax(torch.from_numpy(np.random.uniform(low=0,high=1,size=(self.size, self.size,precision)).astype('float32')),tau=10, hard=True)) 
        self.voltage = torch.nn.Parameter(torch.nn.functional.gumbel_softmax(
				torch.from_numpy(np.random.uniform(low=0,high=1,
					size=(self.size, self.size, precision)).astype('float32')),tau=10, hard=True)) 
        self.register_parameter(name, self.voltage)
        self.phase_func = phase_func
        self.intensity_func = intensity_func 
        self.phase_model = phase_mod
        self.amplitude_factor = amplitude_factor
    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        temp = torch.fft.ifft2(torch.fft.fft2(waves) * self.h)
        if not self.phase_model:
            return temp
        exp_j_phase = torch.matmul(torch.nn.functional.gumbel_softmax(self.voltage,tau=10, hard=True), self.phase_func)
        # mimic look-up-table matching for amplitude vectors
        amplitude = torch.matmul(torch.nn.functional.gumbel_softmax(self.voltage,tau=10, hard=True), 
				self.intensity_func) * self.amplitude_factor # amplitude_factor is a training regularization term
        phase_trig_form = torch.view_as_complex(torch.stack((torch.mul(amplitude,torch.cos(exp_j_phase)), torch.mul(amplitude,torch.sin(exp_j_phase))), dim=-1))
        #temp = torch.view_as_complex(temp)
        #print(temp.shape)
        #print(phase_trig_form.shape)
        x = temp * phase_trig_form
        return x


class DiffractiveLayerRaw_Reflect(torch.nn.Module):
    """ Implementation of diffractive layer without hardware constraints

    Args:
	size: system size
	distance: diffraction distance
	name: name of the layer
	amplitude_factor: training regularization factor w.r.t amplitude vs phase in backpropogation
	phase_mod: enable phase modulation or just diffraction. default: True
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference:
    """

    def __init__(self, pixel_size = 0.0004, size=400, distance=0.08, name="diffractive_layer_raw_reflect", rE=0.0, amplitude_factor=1.0,
                 phase_mod=True, wavelength=7.5e-4):
        super(DiffractiveLayerRaw_Reflect, self).__init__()
        self.size = size  # 200 * 200 neurons in one layer
        self.multi = 2
        self.pixel_size = pixel_size
        self.distance = distance  # distance bewteen two layers (3cm)
        self.ll = self.size * self.pixel_size  # layer length (8cm)
        self.wl = wavelength  # wave length
        self.refE = rE
        self.fi = 1 / self.ll  # frequency interval
        self.dd = self.ll / size
        self.ddi = 1 / self.dd
        self.wn = 2 * np.pi / self.wl  # wave number
        self.h = 0
        self.seth(rE=rE)
        # phase parameter init
        #self.phase = torch.nn.Parameter(
        #    torch.from_numpy(2 * np.pi * torch.nn.init.xavier_uniform_(torch.empty(self.size, self.size)).numpy()),
        #    requires_grad=True)
        self.phase = torch.nn.Parameter(
            2 * np.pi * torch.nn.init.xavier_uniform_(torch.empty(self.size, self.size)),requires_grad=True)
        self.register_parameter(name, self.phase)
        self.phase_model = phase_mod
        self.amplitude_factor = amplitude_factor

    def seth(self, rE=0.0, order=4):
        self.refE = rE
        r = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2))) + np.square((y - (self.size // 2))) + np.square(
                self.distance * self.ddi),
            shape=(self.size, self.size), dtype=np.float)
        r = torch.from_numpy(r)
        h = 1 / (2 * np.pi) * self.distance * self.ddi / r
        r = np.sqrt(r)
        #temp = torch.remainder((self.dd/self.wl) * r, 1.0) * 2 * np.pi
        temp = (self.dd * self.wn) * r
        temp = torch.view_as_complex(torch.stack((torch.cos(temp), torch.sin(temp)), dim=-1))
        #h = h * (1 / r - 1.0j * self.wn * self.dd) * torch.exp((1.0j * 2 * np.pi) * temp)
        h = h * (1 / r - 1.0j * self.wn * self.dd) * temp
        h = torch.fft.fftshift(h)
        h = torch.fft.fft2(h.to(torch.complex64))
        if rE > 0:
            temp = rE * rE * h * h
            temp2 = h
            for _ in range(order):
                temp2 = temp * temp2
                h += temp2
        self.h = torch.nn.Parameter(h, requires_grad=False)

    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        temp = torch.fft.ifft2(torch.fft.fft2(waves) * self.h)
        if not self.phase_model:
            return temp
        temp2 = torch.remainder(self.phase, 2*np.pi)
        #exp_j_phase = torch.view_as_complex(torch.stack((self.amplitude_factor * torch.cos(self.phase),
        #                                                 self.amplitude_factor * torch.sin(self.phase)), dim=-1))
        exp_j_phase = torch.view_as_complex(torch.stack((self.amplitude_factor * torch.cos(temp2),
                                                         self.amplitude_factor * torch.sin(temp2)), dim=-1))
        x = temp * exp_j_phase
        return x


class DiffractiveLayer_pixel(torch.nn.Module):
    """ Implementation of diffractive layer that enables device-quantization aware training using Gumble-Softmax 

    Args:
	phase_func: phase space designed in a given hardware device, where the index represents the applying voltage/grayvalue (e.g., SLM)
	intensity_func: intensity space designed in a given hardware device, where the index represents the applying voltage/grayvalue (e.g., SLM)
	size: system size 
	distance: diffraction distance
	name: name of the layer
	precision: hardware precision encoded in number of possible values of phase or amplitude-phase. default: 256 (8-bit)
	amplitude_factor: training regularization factor w.r.t amplitude vs phase in backpropogation
	phase_mod: enable phase modulation or just diffraction. default: True
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

    Reference: 
    """
 
    def __init__(self, phase_func, intensity_func, wavelength=5.32e-7, pixel_size=0.000036,size=200, distance=0.1, name="diffractive_layer", 
				precision=256, amplitude_factor = 6, pixel=True,phase_mod=True):
        super(DiffractiveLayer_pixel, self).__init__()
        self.size = size                         # 200 * 200 neurons in one layer
        self.distance = distance                    # distance bewteen two layers (3cm)
        self.ll = pixel_size * size                          # layer length (8cm)
        self.wl = wavelength                  # wave length
        self.fi = 1 / self.ll                   # frequency interval
        self.wn = 2 * 3.1415926 / self.wl       # wave number
        # self.phi (syssize, syssize)
        self.phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size), dtype=np.complex64)
        # h (syssize, syssize)
        h = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * self.phi))
        # self.h (syssize, syssize, 2)
        self.h = torch.nn.Parameter(torch.view_as_complex(torch.stack((torch.from_numpy(h.real), torch.from_numpy(h.imag)), dim=-1)), requires_grad=False)
        # initialization with gumbel softmax (random one-hot encoding for voltage)
        self.voltage = torch.nn.Parameter(torch.nn.functional.gumbel_softmax(
				torch.from_numpy(np.random.uniform(low=0,high=1,
					size=(self.size, self.size, precision)).astype('float32')),tau=10, hard=True)) 
        self.register_parameter(name, self.voltage)
        self.phase_func = phase_func
        self.intensity_func = intensity_func 
        self.phase_model = phase_mod
        self.pixel = pixel
        self.amplitude_factor = amplitude_factor
    '''
    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        temp = torch.fft.ifft2(torch.fft.fft2(waves) * self.h)
        if not self.phase_model:
            return temp
        exp_j_phase = torch.matmul(torch.nn.functional.gumbel_softmax(self.voltage,tau=10, hard=True), self.phase_func)
        # mimic look-up-table matching for amplitude vectors
        amplitude = torch.matmul(torch.nn.functional.gumbel_softmax(self.voltage,tau=10, hard=True), 
				self.intensity_func) * self.amplitude_factor # amplitude_factor is a training regularization term
        phase_trig_form = torch.view_as_complex(torch.stack((torch.mul(amplitude,torch.cos(exp_j_phase)), torch.mul(amplitude,torch.sin(exp_j_phase))), dim=-1))

        x = temp * phase_trig_form
        return x
    '''
    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        temp = torch.fft.ifft2(torch.fft.fft2(waves) * self.h)
        if not self.phase_model:
            temp_flat = torch.flatten(temp)
            temp_10 = torch.ones((64),dtype=torch.cfloat).cuda()
            temp_grid, temp_10_grid = torch.meshgrid(temp_flat, temp_10)
            w_grid = temp_grid.reshape(temp.shape[0], temp.shape[1], 8, 8)
            w_grid = F.pad(input=w_grid, pad=(2, 2, 2, 2), mode='constant', value=0)
            w_list = []
            for i in range(temp.shape[0]):
                for j in range(temp.shape[1]):
                    if j==0:
                        w_grid_2 = w_grid[i][0]
                    else:
                        w_grid_2 = torch.cat((w_grid_2, w_grid[i][j]), dim=1)
                w_list.append(w_grid_2) 
            for k in range(temp.shape[0]):
                temp_expand = torch.cat((w_grid_2[k], w_grid_2[k+1]), dim=0)
                k = k+2
            return temp_expand

        exp_j_phase = torch.matmul(torch.nn.functional.gumbel_softmax(self.voltage,tau=10, hard=True), self.phase_func)
        # mimic look-up-table matching for amplitude vectors
        amplitude = torch.matmul(torch.nn.functional.gumbel_softmax(self.voltage,tau=10, hard=True), 
				self.intensity_func) * self.amplitude_factor # amplitude_factor is a training regularization term
        phase_trig_form = torch.view_as_complex(torch.stack((torch.mul(amplitude,torch.cos(exp_j_phase)), torch.mul(amplitude,torch.sin(exp_j_phase))), dim=-1))
        x = temp * phase_trig_form
        x_expand = []
        for m in range(x.shape[0]):
          x_flat = torch.flatten(x[m])
          x_10 = torch.ones((64), dtype=torch.cfloat).cuda()
          x_grid, x_10_grid = torch.meshgrid(x_flat, x_10)
          w_grid = x_grid.reshape(x.shape[1],x.shape[2], 8, 8)
          w_grid = F.pad(input=w_grid, pad=(1, 1, 1, 1), mode='constant', value=0)
          w_list = []
          for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if j==0:
                    w_grid_2 = w_grid[i][0]
                else:
                    w_grid_2 = torch.cat((w_grid_2, w_grid[i][j]), dim=1)
            w_list.append(w_grid_2) 
          for k in range(x.shape[1]-1):
            if k ==0:
                x_one_exp = torch.cat((w_list[0], w_list[1]), dim=0)
            else:
                x_one_exp = torch.cat((x_one_exp, w_list[k+1]), dim=0)      
          print(x_one_exp.shape)
          x_one_exp = torch.unsqueeze(x_one_exp, 0)
          x_expand.append(x_one_exp)
        for n in range(x.shape[0]-1):
          if n==0:
              x_expand_2 = torch.cat((x_expand[0], x_expand[1]), dim=0)
          else:
              x_expand_2 = torch.cat((x_expand_2, x_expand[n+1]),dim=0)
        print(x_expand_2.shape)
        return x_expand_2

