# How to run 

# 1. Clone with Submodules: 
git clone --recurse-submodules -j8 https://github.com/s0tt/ActiveLearningProject.git

# 2. Install dependencies: 

## 2.1 Python version: 
You need Python 3.5 or higher. The code was explicitly tested and developed with Python 3.7. 
## 2.2 Python packages: 

Install dependencies for ModAL ("Active Learning Framework")
```
<python3-interpreter> -m pip install -r modAL/rtd_requirements.txt 
```
Install dependencies for label-studio ("Labeling-Framework")
```
<python3-interpreter> -m pip install -r label-studio/requirements.txt
```

## 3. Run examples

## How to use the basic_pytorch_active_learner.py 
* install libraries: active learner, pytorch ... 
* Replace the Python/3.7/site-packages/modAL/models/base.py file in the modAL library with the uploaded base.py file  


## How to use the Bert_modAl part
* set the variables: model, cache_dir, data_dir in the modAl file 
* take care that all files from mrqa-baseline which get imported are available ... (Sry that it is not so clear )


