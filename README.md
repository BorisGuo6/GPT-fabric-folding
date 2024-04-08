# GPT-Fabric: Folding and Smoothing Fabric by Leveraging Pre-Trained Foundation Models
**Vedant Raval*, Enyu Zhao*, Hejia Zhang, Stefanos Nikolaidis, Daniel Seita**

**University of Southern California**

This repository is a python implementation of the paper "GPT-Fabric: Folding and Smoothing Fabric by Leveraging Pre-Trained Foundation Models", submitted to IROS 2024. This repository contains the code used to run the GPT-fabric simulation experiments for fabric folding. The code for performing fabric smoothing can be found in the repo [Coming soon]()

[Website](https://sites.google.com/usc.edu/gpt-fabrics/home) | [ArXiv: Coming soon]()

## Table of Contents
* [Installation](#Installation)
* [GPT-Fabric, zero-shot](#evaluating-gpt-fabric-in-zero-shot-setting)
* [GPT-Fabric, in-context](#evaluating-gpt-fabric-while-performing-in-context-learning)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## Installation
This simulation environment is based on SoftGym. You can follow the instructions in [SoftGym](https://github.com/Xingyu-Lin/softgym) to setup the simulator.

1. Clone this repository.

2. Follow the [SoftGym](https://github.com/Xingyu-Lin/softgym) to create a conda environment and install PyFlex. [A nice blog](https://danieltakeshi.github.io/2021/02/20/softgym/) written by Daniel Seita may help you get started on SoftGym.

3. Install the following packages in the created conda environment:
    
    * pytorch and torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
    * einops: `pip install einops`
    * tqdm: `pip install tqdm`
    * yaml: `pip install PyYaml`


4. Before you use the code, you should make sure the conda environment activated(`conda activate softgym`) and set up the paths appropriately: 
   ~~~
   export PYFLEXROOT=${PWD}/PyFlex
   export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
   export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
   ~~~
   The provided script `prepare_1.0.sh` includes these commands above.

## Evaluating GPT-Fabric in zero-shot setting

Coming soon

## Evaluating GPT-Fabric while performing in-context learning

Coming soon

## License

Coming soon

## Acknowledgements

Coming soon

## Contact

For any additional questions, feel free to email [ravalv@usc.edu](ravalv@usc.edu)