[![License: GPL 
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# Welcome to LightRidge — An Open-Source Hardware Project for Optical AI!
[**Documentation**](https://lightridge.github.io/lightridge/index.html#) 

![image](https://lightridge.github.io/lightridge/_images/lightridge_flow.png)

LightRidge is an open-source framework for end-to-end optical machine learning (ML) 
compilation, which connects physics to system. It is specifically designed for 
diffractive optical computing, offering a comprehensive set of features:

- Precise and differentiable optical physics kernels: LightRidge empowers researchers 
and developers to explore and optimize diffractive optical neural network (DONN) 
architectures. With built-in, accurate, and differentiable optical physics kernels, 
users can achieve complete and detailed analyses of DONN performance.
- Accelerated optical physics computation kernel: LightRidge incorporates 
high-performance computation kernels, resulting in significant runtime reductions 
during training, emulation, and deployment of DONNs. This acceleration streamlines the 
development process and boosts the efficiency of optical ML workflows.
- Versatile and flexible optical system modeling: LightRidge provides a rich set of 
tools for modeling and simulating optical systems. Researchers can create complex 
optical setups, simulate light propagation, and analyze system behavior using 
LightRidge’s versatile capabilities.
- User-friendly domain-specific language (DSL): LightRidge includes a user-friendly 
DSL, enabling users to describe and configure diffractive optical networks easily. The 
DSL simplifies the implementation process and facilitates rapid prototyping of novel 
optical ML models.


## Getting Started

### Installation -- Install from PYPI

```bash
pip install lightridge
```
 

Google Colab Tutorial: [![Open In 
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JzUGxbnxCUJwU_IC70VEMEC8Yw6F7uNJ?usp=sharing]

Tutorial Webpage: 
[**Tutorial**](https://lightridge.github.io/lightridge/lightridge_tutorial_ASPLOS24_AE.html)


### Usage
Please see the [examples](examples/) folder for more details. Documentations will be 
released soon.
```python
import lightridge

# define your DONNs model
```bash
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

    def phase_view(self,x, cmap="hsv"):
        phase_list = []
        for index, layer in enumerate(self.diffractive_layers):
            phase_list.append(self.phase)
        print(phase_list[0].shape)
        utils.phase_visualization(phase_list,size=self.size, cmap="gray", fname="prop_view_reflection.pdf")
        return
```


## Optical neural architecture configurations

1. Set the hardware information for the emulation

*   Input laser source information: *wavelength* (532e-9)
*   System size: *sys_size* (200)
*   Pixel size: *pixel_size* (3.6e-5)
*   Padding size for emulations: *pad* (100)

2.   Define the system parameters for model construction

*   Diffractive layers: inlcudes diffraction approximation and phase modulation in 
*layers.DiffractLayer_Raw/layers.DiffractLayer*:
  - Mathematical approximation for light diffraction: *approx*
  - Diffraction distance: *distance*
  - System depth: *num_layers*
  - Training regularization: *amp_factor*


*   Detector design: defined in *layers.Detector*:
  - Location coordinate for detector regions: *det_loc_x,
det_loc_y*,
  - Size for each sub-detector: *det_size*





3.   Visualization functions
*    Propogation pattern visualization: *prop_view*
*    Trainable parameter, phase modulation visualization: *phase_view*

```bash
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
```

## Visulization Example

MNIST-10 "7" Prediction and Light Propogation --
![image](https://lightridge.github.io/lightridge/_images/lightridge_tutorial_ASPLOS24_21_1.png)


## Publication
```bash
@article{li2023lightridge,
  title={LightRidge: An End-to-end Agile Design Framework for Diffractive Optical 
Neural Networks},
  author={Li, Yingjie and Chen, Ruiyang and Lou, Minhan and Sensale-Rodriguez, Berardi 
and Gao, Weilu and Yu, Cunxi},
  journal={International Conference on Architectural Support for Programming Languages 
and Operating Systems (ASPLOS)},
  year={2023}
}
```

## License
LightRidge is released under the [![License: GPL 
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
