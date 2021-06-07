# ActiveLearningProject

## Clone with Submodules: 
```
git clone --recurse-submodules -j8 https://github.com/s0tt/ActiveLearningProject.git
```
## Update submodules to latest commit 
```
git submodule update --remote --merge
```

# Requirements 

**Python >= 3.73** should be used. The framework/ the extensions were explicitly developed and tested with **Python 3.73**.
## Install dependencies 
We have to install dependencies for modAL as well as label-studio. 

For **modAL** execute the command from the root directory
```
<python3-interpreter> -m pip install -r modAL/rtd_requirements.txt
```
For **label-studio** execute the command from the root directory 
```
<python3-interpreter> label-studio/setup.py install
```

To use the examples torchvision and matplotlib are additionally required. Install: 
```
<python3-interpreter> -m pip install torchvision matplotlib
```

# Examples

Code examples can be found under *ActiveLearningProject/Examples/*. For caching and datasets the folders *ActiveLearningProject/cache/* and *ActiveLearningProject/datasets/* can be used.

## Basic PyTorch model usage in modAL 

For explanation on how to use PyTorch models in the modAL work flow check the comments in: `ActiveLearningProject/Examples/modAL_example/basic_PyTorch_example.py`

## label-studio with PyTorch and modAL 

For explanation on how to use label-studio and modAL together with a PyTorch check the comments in: `ActiveLearningProject/Examples/modAL_example/multi_metric_PyTorch_label_example.py`


# Documentation 

The full **modAL** documentation can be found on <https://modal-python.readthedocs.io/en/latest/>.

The documentation for **label-studio** can be found here <https://labelstud.io/guide/>.




