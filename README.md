# GPT-Fabric: Folding and Smoothing Fabric by Leveraging Pre-Trained Foundation Models
**Vedant Raval\*, Enyu Zhao\*, Hejia Zhang, Stefanos Nikolaidis, Daniel Seita**

**University of Southern California**

This repository is a python implementation of the paper "GPT-Fabric: Folding and Smoothing Fabric by Leveraging Pre-Trained Foundation Models", submitted to IROS 2024. This repository contains the code used to run the GPT-fabric simulation experiments for fabric folding. The code for performing fabric smoothing can be found in the repo [GPT-Fabric-Smoothing](https://github.com/slurm-lab-usc/GPT-Fabric-Smoothing)

[Website](https://sites.google.com/usc.edu/gpt-fabrics/home) | [ArXiv: Coming soon]()

## Table of Contents
* [Installation](#installation)
* [Pre-requisites](#pre-requisites)
* [GPT-Fabric, zero-shot](#evaluating-gpt-fabric-in-zero-shot-setting)
* [GPT-Fabric, in-context](#evaluating-gpt-fabric-while-performing-in-context-learning)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## Installation
This simulation environment is based on SoftGym. You can follow the instructions in [SoftGym](https://github.com/Xingyu-Lin/softgym) to setup the simulator.

1. Clone this repository.

2. Follow the [SoftGym](https://github.com/Xingyu-Lin/softgym) to create a conda environment and install PyFlex. [A nice blog](https://danieltakeshi.github.io/2021/02/20/softgym/) written by Daniel Seita on using Docker may help you get started on SoftGym.

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

## Pre-requisites

### Getting the evaluation configurations and demonstration sub-goals
- To get started with GPT-Fabric for folding, you will need initial cloth configurations for evaluating the system as well as demonstration sub-goals for the folding tasks which would be used by the LLM to generate instructions.

- To be consistent with prior work, we use the initial evaluation configurations for square and rectangular shaped fabric used by [Foldsformer](https://github.com/Murkey8895/foldsformer/tree/main?tab=readme-ov-file#evaluate-foldsformer). You can find these in `cached configs/`.

- You can also generate configurations yourself by running
    ~~~
    python generate_configs.py --num_cached 100 --cloth_type square
    ~~~
    where `--num_cached` specifies the number of configurations to be generated, and `--cloth_type` specifies the cloth type (square | rectangle | random). These generated initial configurations will be saved in `cached configs/`

- To get the demonstration sub-goals to be used by GPT in zero-shot setting, we used the same demonstration sub-goal images as provided by [Foldsformer](https://github.com/Murkey8895/foldsformer/tree/main?tab=readme-ov-file#evaluate-foldsformer). You can find these in `data/demo`.

### Getting the ground truth demonstrations for evaluation

- In order to evaluate the folds achieved by GPT-fabric, we need to compare the results with folds obtained by some expert system. This expert system can be found in the `Demonstrator` directory and can be ran using
    ~~~
    python generate_demonstrations.py --gui --task DoubleTriangle --img_size 128 --cached square
    python generate_demonstrations.py --gui --task DoubleStraight --img_size 128 --cached rectangle
    python generate_demonstrations.py --gui --task AllCornersInward --img_size 128 --cached square
    python generate_demonstrations.py --gui --task CornersEdgesInward --img_size 128 --cached square
    ~~~
    where `--task` specifies the task name, `--img_size` specifies the image size captured by the camera in the simulator, and `--cached` specifies the filename of the cached configurations. You can remove `--gui` to run headless. These generated demonstrations will be saved in `data/demonstrations`.

- Note that since the same folding task could be achieved in various ways for the same cloth configuration, we consider all the different possible final cloth configurations corresponding to such a successful cloth fold as per the expert driven heuristic (aka the `Demonstrator`).

- For each cloth configuration, `0.png` is the top-down image corresponding to the initial state. `{step#}-{fold#}.png` is the top-down image corresponding to the given step number `{step#}` for the given specifgic way of achieving the successful fold represented as `{fold#}`. The final cloth configuration will be saved as a pickle file given as `info-{fold#}.pkl`. TO compute the mean particle position error (in mm) for evaluation, we consider the distances for all the possible final cloth configurations from the acheived final cloth configuration by GPT-Fabric and take the minimum of those.

- These are the total number of possible folding steps and total possible ways of successfully folding the same initial cloth for the folding types considered by us
    - DoubleTriangle: 8 possible ways of folding, each taking two steps
    - DoubleStraight: 16 possible ways of folding, each taking three steps
    - AllCornersInward: 1 possible ways of folding, each taking four steps
    - CornersEdgesInward: 16 possible ways of folding, each taking four steps

## Evaluating GPT-Fabric in zero-shot setting

- To reproduce the results obtained by GPT-Fabric (GPT-4, zero-shot):
    ~~~
    python eval.py --task DoubleTriangle --img_size 128 --gpt_model gpt-4-1106-preview --cached square --total_runs 5 --eval_type zero-shot
    python eval.py --task DoubleStraight --img_size 128 --gpt_model gpt-4-1106-preview --cached rectangle --total_runs 5 --eval_type zero-shot
    python eval.py --task AllCornersInward --img_size 128 --gpt_model gpt-4-1106-preview --cached square --total_runs 5 --eval_type zero-shot
    python eval.py --task CornersEdgesInward --img_size 128 --gpt_model gpt-4-1106-preview --cached square --total_runs 5 --eval_type zero-shot
    ~~~

- To reproduce the results obtained by GPT-Fabric (GPT-3.5, zero-shot):
    ~~~
    python eval.py --task DoubleTriangle --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached square --total_runs 5 --eval_type zero-shot
    python eval.py --task DoubleStraight --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached rectangle --total_runs 5 --eval_type zero-shot
    python eval.py --task AllCornersInward --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached square --total_runs 5 --eval_type zero-shot
    python eval.py --task CornersEdgesInward --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached square --total_runs 5 --eval_type zero-shot
    ~~~

- The above script would run GPT-fabric for fabric folding corresponding to the specified `--task` for each initial cloth configuration in the saved `--cached` configurations. The choice of which GPT model do we wanna use is specified by `--gpt_model` and the `--eval_type` corresponding to *zero-shot* would correspond to the zero-shot version of GPT-Fabric for fabric folding. In our reported results, we ran our system for each initial cloth configuration for a total of `--total_runs`. If you just wish to test how the system performs without a specific need to reproduce our results then you can simply set `--total_runs` as 1. We run our system for five times to account for the randomness in the LLM's responses. Log files will be generated corresponding to each test run in the `logs` directory. The evaluation results for each test runs are saved in `eval result/`. For the sake of convenience, we organised the saved directories based on the date of the program execution, configuration type etc. The results can be organised in a different directory structure by trivial changes. The mean particle position errors for all the cloth configurations across all the total runs will be saved as a 2D numpy array in `position errors/`.

- In order to save the videos of the generated simulations, you can run the script with `--save_vid` as:
    ~~~
    python eval.py --task DoubleTriangle --img_size 128 --gpt_model gpt-4-1106-preview --cached square --total_runs 5 --eval_type zero-shot --save_vid True
    ~~~
    The directory for the saved simulation videos can be changed via `--save_video_dir`

## Evaluating GPT-Fabric while performing in-context learning

### Getting a few expert demonstrations to perform in-context learning

- In order to perform in-context learning for GPT-4V before generating instructions, we need to generate some expert demonstrations consisting of the demonstration sub-goal images
    ~~~
    python generate_configs.py --num_cached 100 --cloth_type square
    python generate_configs.py --num_cached 100 --cloth_type rectangle
    ~~~
    The above script will generate new configurations `square100` and `rectangle100`, different from the evaluation configurations of `square` and `rectangle`.

    ~~~
    python training-examples/generate_demonstrations.py --gui --task DoubleTriangle --img_size 224 --cached square100
    python training-examples/generate_demonstrations.py --gui --task DoubleStraight --img_size 224 --cached rectangle100
    python training-examples/generate_demonstrations.py --gui --task AllCornersInward --img_size 224 --cached square100
    python training-examples/generate_demonstrations.py --gui --task CornersEdgesInward --img_size 224 --cached square100
    ~~~
    The above script will generate expert demonstrations corresponding to the previously generated cloth configurations.

- In order to perform in-context learning for GPT-4 or GPT-3.5 before generating actions, we need to [coming soon]

### Performing in-context learning on GPT-Fabric

- To reproduce the results obtained by GPT-Fabric (GPT-4, in-context):
    ~~~
    python eval.py --task DoubleTriangle --img_size 128 --gpt_model gpt-4-1106-preview --cached square --total_runs 5 --eval_type in-context
    python eval.py --task DoubleStraight --img_size 128 --gpt_model gpt-4-1106-preview --cached rectangle --total_runs 5 --eval_type in-context
    python eval.py --task AllCornersInward --img_size 128 --gpt_model gpt-4-1106-preview --cached square --total_runs 5 --eval_type in-context
    python eval.py --task CornersEdgesInward --img_size 128 --gpt_model gpt-4-1106-preview --cached square --total_runs 5 --eval_type in-context
    ~~~

- To reproduce the results obtained by GPT-Fabric (GPT-3.5, in-context):
    ~~~
    python eval.py --task DoubleTriangle --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached square --total_runs 5 --eval_type in-context
    python eval.py --task DoubleStraight --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached rectangle --total_runs 5 --eval_type in-context
    python eval.py --task AllCornersInward --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached square --total_runs 5 --eval_type in-context
    python eval.py --task CornersEdgesInward --img_size 128 --gpt_model gpt-3.5-turbo-0125 --cached square --total_runs 5 --eval_type in-context
    ~~~

## License

Coming soon

## Acknowledgements

A lot of this code has been adapted from the repository used for [Foldsformer](https://github.com/Murkey8895/foldsformer). Feel free to check that out!

## Contact

For any additional questions, feel free to email [ravalv@usc.edu](ravalv@usc.edu)
