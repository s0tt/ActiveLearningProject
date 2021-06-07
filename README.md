# ActiveLearningProject

## Clone with Submodules: 
```
git clone --recurse-submodules -j8 https://github.com/s0tt/ActiveLearningProject.git
```
After the repository is cloned change into the project root directory *ActiveLearningProject/* and run the following command
```
git submodule update --remote --merge
```
to update the submodules to the latest commit. 
# Requirements 

**Python >= 3.73** should be used. The framework/ the extensions were explicitly developed and tested with **Python 3.73**.
## Install dependencies 
We have to install dependencies for modAL as well as label-studio. 

For **modAL** execute the command from the root directory
```
<python3-interpreter> -m pip install -r modAL/rtd_requirements.txt
```
For **label-studio** execute the following command from the root directory 
```
<python3-interpreter> -m pip install -e label-studio/
```

To use the examples additional python libraries are required. Install them with: 
```
<python3-interpreter> -m pip install torchvision matplotlib jsonlines psutil tensorboard transformers==2.9.0
```

# Examples

Code examples can be found under *ActiveLearningProject/Examples/*. For caching and datasets the folders *ActiveLearningProject/cache/* and *ActiveLearningProject/datasets/* are provided. All examples can directly be executed as python scrips without the need to pass any parameters.

## Basic PyTorch model usage in modAL 

For explanation on how to use PyTorch models in the modAL work flow check the comments in: `ActiveLearningProject/Examples/modAL_example/basic_PyTorch_example.py`

## label-studio with PyTorch and modAL 

For explanation on how to use label-studio and modAL together with a PyTorch check the comments in: `ActiveLearningProject/Examples/modAL_labelStudio_example/multi_metric_PyTorch_label_example.py`

## BERT-QA with modAL and label-studio

For explanation on how to use a natural language processing model like BERT with label-studio (Question-Answering) and modAL check the comments in.: `ActiveLearningProject/Examples/bert_modAL_labelStudio/bert_example.py`



# Documentation 

The full **modAL** documentation can be found on [here](https://modal-python.readthedocs.io/en/latest/).

The documentation for **label-studio** can be found  [here](https://labelstud.io/guide/).




