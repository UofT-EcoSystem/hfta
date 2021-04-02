# Horizontally Fused Training Array

![Logo](/docs/images/intro.gif "Horizontally Fused Training Array")

--------------------------------------------------------------------------------

_**H**orizontally **F**used **T**raining **A**rray_ (HFTA) is a
[PyTorch](https://pytorch.org/) extension library that helps machine learning
and deep learning researchers and practitioners to develop **horizontally 
fused** models. Each fused model is functionally/mathematically equivalent to 
**an array of models** with **similar/same operators**.

Why developing horizontally fused models at all, you ask? This is because
sometimes training a certain class of models can **under-utilize** the 
underlying accelerators. Such hardware under-utilization could then be **greatly 
amplified** if you train this class of models **repetitively** (e.g., when you 
tune its hyper-parameters). Fortunately, in such use cases, the models under 
repetitive training often have the **same types** of operators with the **same 
shapes** (e.g., think about what happens to the operators when you adjust the 
learning rate). Therefore, with HFTA, you can improve the hardware utilization 
by training an array of models (as a single fused model) on the same accelerator 
at the same time.

HFTA is **device-agnostic**. So far, we tested HFTA and observed significant
training performance and hardware utilization improvements on NVIDIA
[V100](https://www.nvidia.com/en-us/data-center/v100/),
[RTX6000](https://www.nvidia.com/en-us/design-visualization/quadro/rtx-6000/)
and [A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs and Google
[Cloud TPU](https://cloud.google.com/tpu) v3.

## Installation

### From Source

```bash
# NVIDIA GPUs:
$ pip install git+https://github.com/UofT-EcoSystem/hfta

# Google Cloud TPU v3:
$ pip install git+https://github.com/UofT-EcoSystem/hfta[xla]
```

### From PyPI

TODO

### Testing the Installation

1. Clone the HFTA's repo.

    ```bash
    # Clone the repo
    $ git clone https://github.com/UofT-EcoSystem/hfta
    ```

2. Run the MobileNet-V2 example without HFTA.

    ```bash
    # NVIDIA GPUs:
    $ python hfta/examples/mobilenet/main.py --version v2 --epochs 5 --amp --eval --dataset cifar10 --device cuda --lr 0.01

    # Google Cloud TPU v3:
    $ python hfta/examples/mobilenet/main.py --version v2 --epochs 5 --amp --eval --dataset cifar10 --device xla --lr 0.01

    # The following output is captured on V100:
    Enable cuDNN heuristics!
    Files already downloaded and verified
    Files already downloaded and verified
    Epoch 0 took 7.802547454833984 s!
    Epoch 1 took 5.990707635879517 s!
    Epoch 2 took 6.000213623046875 s!
    Epoch 3 took 6.0167365074157715 s!
    Epoch 4 took 6.071732521057129 s!
    Running validation loop ...
    ```

3. Run the same MobileNet-V2 example with HFTA, testing three learning rates on
   the same accelerator simultaneously.

    ```bash
    # NVIDIA GPUs:
    $ python hfta/examples/mobilenet/main.py --version v2 --epochs 5 --amp --eval --dataset cifar10 --device cuda --lr 0.01 0.03 0.1 --hfta

    # Google Cloud TPU v3:
    $ python hfta/examples/mobilenet/main.py --version v2 --epochs 5 --amp --eval --dataset cifar10 --device xla --lr 0.01 0.03 0.1 --hfta

    # The following output is captured on V100:
    Enable cuDNN heuristics!
    Files already downloaded and verified
    Files already downloaded and verified
    Epoch 0 took 13.595093727111816 s!
    Epoch 1 took 7.609431743621826 s!
    Epoch 2 took 7.635211229324341 s!
    Epoch 3 took 7.6383607387542725 s!
    Epoch 4 took 7.7035486698150635 s!
    ```

In the above example, ideally, the end-to-end training time for MobileNet-V2
with HFTA should be much less than three times the end-to-end training time
without HFTA.

## Getting Started

TODO

## Publication

- Fourth Conference on Machine Learning and Systems
  ([MLSys'21](https://mlsys.org/))
  - [Horizontally Fused Training Array: An Effective Hardware Utilization
    Squeezer for Training Novel Deep Learning Models](https://mlsys.org/virtual/2021/oral/1610)
    - [Proceedings](https://proceedings.mlsys.org/paper/2021/hash/a97da629b098b75c294dffdc3e463904-Abstract.html)
    - [Talk](https://youtu.be/zJ5UUb0J9tI)
    - [arXiv](https://arxiv.org/abs/2102.02344)
    - Please refer to the [MLSys'21 Artifact Reproduction Guide](docs/mlsys21/README.md)
      on how to reproduce the reported experiments.

## Citation

If you use HFTA in your work, please cite our MLSys'21 publication using the
following BibTeX:

```BibTeX
% TODO: Update the BibTex after pre-proceeding -> proceeding.
@inproceedings{MLSYS2021_HFTA,
 author = {Shang Wang and Peiming Yang and Yuxuan Zheng and Xin Li and Gennady Pekhimenko},
 booktitle = {Proceedings of Machine Learning and Systems},
 title = {Horizontally Fused Training Array: An Effective Hardware Utilization Squeezer for Training Novel Deep Learning Models},
 url = {https://proceedings.mlsys.org/paper/2021/file/a97da629b098b75c294dffdc3e463904-Paper.pdf},
 volume = {3},
 year = {2021}
}
```

## Contributing

We sincerely appreciate contributions! We are currently working on the
contributor guidelines. For now, just send us a PR for review!

## License

HFTA itself has a [MIT License](LICENSE). When collecting the [examples](examples/)
and [benchmarks](benchmarks/), we leverage other open-sourced projects, and we
include their licenses in their corresponding directories.

## Authors

HFTA is developed and maintained by Shang Wang ([@wangshangsam](https://github.com/wangshangsam)),
Peiming Yang ([@ypm1999](https://github.com/ypm1999)), Yuxuan (Eric) Zheng
([@eric-zheng](https://github.com/eric-zheng)), Xin Li ([@nixli](https://github.com/nixli)).

HFTA is one of the research projects from the [EcoSystem](https://www.cs.toronto.edu/ecosystem/)
group at the [University of Toronto](https://www.utoronto.ca/), [Department of
Computer Science](https://web.cs.toronto.edu/).
