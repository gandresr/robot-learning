# Preliminary Investigation Into Active Sample Selection in Imitation Learning
### Kun Qian, Gerardo Riano
___


## Instructions

First download the datasets executing `./download_datasets.sh`. You need to `pip install gdown` in order to execute the script. The *main.ipynb* jupyter notebook contains the necessary functions to execute ensemble BC with *'variance'*, *'max_error'*, and *'min_error'* acquisition functions. Within the jupyter notebook define the `method: {'variance', 'max_error', 'min_error'}` and the `envname: {'Lift', 'PickPlace'}`.