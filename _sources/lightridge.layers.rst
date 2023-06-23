==========
LightRidge Python API (lr.layers)
==========

.. toctree::
   hidden



lightridge.layers
=====================

**Author**: Cunxi Yu (ycunxi@github)

.. code-block:: default


    import lightridge as lr 


.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <layers.py>` to download the full example code





``lr.Detector(torch.nn.Module):``
---------------
This API returns a prefixed detector implementation of detector plane for multi-task classification 

.. code-block:: default

    class Detector(torch.nn.Module):
    """Implementation of detector plane for multi-task classification 
    The outputs are collected with specific defined detector regions over the entire light propogation.
    The outputs are (optional) normlized using functions such as softmax to enable effcient training of D2NNs.
    
    Args:
        >> detector plane design <<
	x_loc: the x-axis location for your detector region (left-top)
        y_loc: the y-axis location for your detector region (left-top)
        det_size: the size of the detector region
        size: the system size
        activation: activation function for training. default=torch.nn.Softmax(dim=-1)
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input
    Examples::
    Reference: 
    """


``lr.Detector_10(torch.nn.Module):``
---------------
This API returns a prefixed detector implementation of detector plane for multi-task classification 

.. code-block:: default

    class Detector(torch.nn.Module):
    """Implementation of detector plane for multi-task classification 
    The outputs are collected with specific defined detector regions over the entire light propogation.
    The outputs are (optional) normlized using functions such as softmax to enable effcient training of D2NNs.
    
    Args:
        >> detector plane design <<
	x_loc: the x-axis location for your detector region (left-top)
        y_loc: the y-axis location for your detector region (left-top)
        det_size: the size of the detector region
        size: the system size
        activation: activation function for training. default=torch.nn.Softmax(dim=-1)
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input
    Examples::
    Reference: 
    """





``lr.DiffractiveLayer(torch.nn.Module):``
---------------
This API supports diffractive layer that enables device-quantization aware training using Gumble-Softmax 

.. code-block:: default

    class DiffractiveLayer(torch.nn.Module):
    """ Implementation of diffractive layer that enables device-quantization aware training using Gumble-Softmax 
    Args:
	    phase_func: phase space designed in a given hardware device, where the index represents the applying voltage/grayvalue (e.g., SLM)
	    intensity_func: intensity space designed in a given hardware device, where the index represents the applying voltage/grayvalue (e.g., SLM)
	    wavelength: the wavelength of your laser source
            pixel-size: the size of pixels in your diffractive layers
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







``lr.Diffraction(torch.nn.Module):``
---------------

Implementation of diffraction

.. code-block:: default

    class Diffraction(torch.nn.Module):
    """ Implementation of diffraction
    Args:
            wavelength: the wavelength of your laser source
            pixel-size: the size of pixels in your diffractive layers
	    size: system size
	    distance: diffraction distance
	    name: name of the layer
    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input
    Examples::
    Reference:
    """




``lr.DiffractiveLayerRaw(torch.nn.Module):``
---------------
This API supports diffractive layer that with phase and amplitude are not limited to specific hardware. Phase and
amplitude has full range [0,2pi], [0,1], respectively (float32). 

.. code-block:: default

    class DiffractiveLayerRaw(torch.nn.Module):
    """ Implementation of diffractive layer without hardware constraints
    Args:
            wavelength: the wavelength of your laser source
            pixel-size: the size of pixels in your diffractive layers
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


``lr.DiffractiveLayerRaw_Reflect(torch.nn.Module):``
---------------
This API implements diffractive layer, in which the phase and amplitude are not limited to specific hardware (float32 precision), with reflection considered.

.. code-block:: default

    class DiffractiveLayerRaw_Reflect(torch.nn.Module):
        """ Implementation of diffractive layer without hardware constraints
        Args:
            wavelength: the wavelength of your laser source
            pixel-size: the size of pixels in your diffractive layers
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
