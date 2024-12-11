# GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers

---------

## Installation
1. Create a conda environment 
    ```
    conda create -n GauHamer python=3.10
    conda activate GauHamer
    ```
2. Install Pytorch and numpy

    ```
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git
    pip install git+https://gitlab.inria.fr/bkerbl/simple-knn.git
    pip install 'numpy<2.0'
    #pip install requirements.txt
    
    ```

## Global Constants

1. Check paths in 
```
configs/default_config.yaml
scene/mirage.py
scene/hamer_extension.py
HameR/hamer/configs/__init__.py

```

## Training
```
python train_network.py
```
