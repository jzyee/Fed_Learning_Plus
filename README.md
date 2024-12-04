# Fed_Learning_Plus

## Introduction

This repository implements Federated Learning for Incremental Learning with an extensible factory-style architecture, enabling seamless integration and ablation of diverse components to replicate and extend the results of the GLFC framework with a focus on running a resource-constrained experiments to identify the effectiveness of FCIL methods in a resource constrained environment.

## Table of Contents

<!-- TOC -->

- [Fed\_Learning\_Plus](#fed_learning_plus)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Video Demo](#video-demo)
  - [How to get script running](#how-to-get-script-running)
    - [For Linux/Ubuntu](#for-linuxubuntu)
  - [Hardware tested on](#hardware-tested-on)
  - [File structure](#file-structure)
  - [Relevant Method Papers](#relevant-method-papers)
    - [2024](#2024)
    - [2023](#2023)
    - [2022](#2022)
  - [Relevant Survey Papers](#relevant-survey-papers)
    - [2024](#2024-1)
- [References](#references)

<!-- /TOC -->

## Video Demo

[![Watch the video](https://img.youtube.com/vi/flcQrno9D9g/hqdefault.jpg)](https://youtu.be/flcQrno9D9g)

Video demo of the how to download and use this repository.

## How to get script running

Process has only been test on a Linux machine.


### For Linux/Ubuntu 

1. Please install the pre-requirements under hardware tested on section
   1. You will need to have installed the following:
      - CUDA toolkit

        - [CUDA Toolkit Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#linux)
      - cuDNN

        - [cuDNN Quick Start Guide](https://docs.nvidia.com/deeplearning/cudnn/support-resources/index.html)
      - a variant of anaconda or miniconda

        - [Miniconda Installation Guide](https://docs.anaconda.com/miniconda/miniconda-install/)
2. Create a virtual environment
   ```
   conda create -n fed_learning python=3.10.12
   ```
3. Activate the virtual environment
   ```
   conda activate fed_learning
   ```
4. Then install the requirements
   ```
   pip install -r requirements.txt
   ```
5. Run any bash file to implement the FCIL method 
   1. the following command will run the bash file and save the output of the bash file into a .txt so that the output is not lost.
   2. If you want to want more detail about how to run the bash file, please see notebook/demo.ipynb. Or this video: [link](https://youtu.be/flcQrno9D9g)
   ```
   bash run_glfc_cifar10.sh > output_glfc_cifar10.txt
   ```

## Hardware tested on

- 1x A6000 48GB
- Linux
- Python 3.10.12

## File structure

```
.
├── dataset/        # Dataset management and data loading utilities
├── encoder/         # Data encoding and preprocessing modules
├── eval/           # Evaluation metrics and testing frameworks
├── model/         # Neural network architectures and model definitions
├── notebooks/      # Jupyter notebooks for demonstrations and tutorials
├── output/         # Storage for experiment outputs, logs, and results
├── scripts/        # Shell scripts and experiment runners
├── training/       # Training loops and optimization procedures
├── utils/          # Helper functions and common utilities
└── weight_agg/     # Weight aggregation strategies for federated learning
```





## Relevant Method Papers

### 2024

---


| model          | paper name                                                                                                        | link                                                                                                                                                                                | year | published                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------- |
| IOFL           | IOFL: Intelligent-Optimization-Based Federated Learning for Non-IID Data                                          | [link](https://ieeexplore.ieee.org/abstract/document/10400794?casa_token=Cc6el4Ty9RIAAAAA:UJ-Z7a0CkS9QrLOHvOYkosvioBWM3xZV4kWxbcmv6S7FOFVEC7jB4FNq5iexWQXMJ0hNKvgj9Zx5)             | 2024 | [IEEE Internet of Things Journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6488907)                       |
| auditable PPFL | A Robust Privacy-Preserving Federated Learning Model Against Model Poisoning Attacks                              | [link](https://ieeexplore.ieee.org/abstract/document/10574838?casa_token=HmEpfwp55EAAAAAA:MWDRa0eG0JIoNr9asCy0Yn5jRojXdW7MM3rVIlUVb8ZANt-vcBN772rdtV8bFAF3-LzOBMiFitmh)             | 2024 | [IEEE Transactions on Information Forensics and Security](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10206) |
| Fedstellar     | Fedstellar: A Platform for Decentralized Federated Learning                                                       | [link](https://)                                                                                                                                                                    | 2024 | [Expert Systems with Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications)                |
| MMVFL          | MMVFL: A simple vertical federated learning framework for multi-participant scenarios                             | [link](https://www.mdpi.com/1424-8220/24/2/619)                                                                                                                                     | 2024 | [Sensors](https://www.mdpi.com/journal/sensors/sections/sensornetworks)                                                   |
| Fedisp         | Fedisp: an incremental subgradient proximal-based ring-type architecture for decentralized federated learning     | [link](https://link.springer.com/article/10.1007/s40747-023-01272-4)                                                                                                                | 2024 | [Complex & Intelligent Systems](https://link.springer.com/journal/40747)                                                  |
| PI-Fed         | Continual Federated Learning With Parameter-Level Importance Aggregation                                          | [link](https://ieeexplore.ieee.org/abstract/document/10628095?casa_token=hNCao18hEsMAAAAA:snOdv4MtYydq5MKlptgTF5amoSLrZk4KiMtmaXG9HTclO4N1nU5XQnwsBYTAJ4ooPwVHot50jPHO)             | 2024 | [IEEE Internet of Things Journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6488907)                       |
| SIBLS          | Self-balancing Incremental Broad Learning System with privacy protection                                          | [link](https://www.sciencedirect.com/science/article/pii/S0893608024003605?casa_token=sVP6Xgs-mA8AAAAA:FrCNBFOg6YoZAhZ3uAyIr8gp9iDiUURadnsOggPJhwMH2gU5UIkGOrkhsrxj-buRo9n5a5JaeLk) | 2024 | [Neural Networks](https://www.sciencedirect.com/journal/neural-networks)                                                  |
| MFCL           | A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks | [link](https://proceedings.neurips.cc/paper_files/paper/2023/file/d160ea01902c33e30660851dfbac5980-Paper-Conference.pdf)                                                            | 2024 | [Advances in Neural Information Processing Systems](https://)                                                             |

---

### 2023

---


| model | paper name                                                          | link                                                                                                                                                                    | year | published                                                                                                                           |
| ------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| LGA   | No One Left Behind: Real-World Federated Class-Incremental Learning | [link](https://ieeexplore.ieee.org/abstract/document/10323204?casa_token=H07ZQJFzGaIAAAAA:FdRABpDLNCqXuGlXvv69esNHakxdlmgxbbWjjx-JCl9IrJZmLrxZOJFksRVeloDUtgPxE7_bocU4) | 2023 | [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34)<br /> |

---

### 2022

---


| model | paper name                           | link                                     | year | published                                                                                                                                                                                                     |
| ------- | -------------------------------------- | ------------------------------------------ | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GLFC  | Federated Class-Incremental Learning | [link](https://arxiv.org/pdf/2203.11473) | 2022 | [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition(CVPR)](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.html) |

---

## Relevant Survey Papers

### 2024

---


| paper name                                                                          | link                                                                                                                                                                                | year | published                                                                                                                     |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Vertical Federated Learning: Concepts, Advances, and Challenges                     | [link](https://ieeexplore.ieee.org/abstract/document/10415268https:/)                                                                                                               | 2024 | [IEEE Transactions on Knowledge and Data Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=69)            |
| Model aggregation techniques in federated learing: A Comprehensive survey           | [link](https://www.sciencedirect.com/science/article/pii/S0167739X23003333)                                                                                                         | 2024 | [Future Generation Computer Systems](https://www.sciencedirect.com/journal/future-generation-computer-systems)                |
| Recent advances on federated learning: A systematic survey                          | [link](https://www.sciencedirect.com/science/article/pii/S0925231224007902?casa_token=bM_vSCicnyAAAAAA:EPUhnL8yl0QPnc2BGipimqc3z3tDWrW6ewb6UhNfEjI82R8ZJcrC1JEgLWyYkCpxQWXxNwm7o68) | 2024 | [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing)                                                        |
| A survey for federated learning evaluations: Goals and Measures                     | [link](https://ieeexplore.ieee.org/abstract/document/10480259?casa_token=NGQf0Kpx67AAAAAA:pNnaoo4l7ksJu0lUgJmMTJIM0boh8Una19QWGnpk53Bw59WR4ZhiS59AAz-xT8HHcBhiwKxPMr8L)             | 2024 | [IEEE Transactions on Knowledge and Data Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=69)            |
| Federated Learning With Non-IID Data: A Survey                                      | [link](https://ieeexplore.ieee.org/abstract/document/10468591?casa_token=hIokwbbxd2QAAAAA:AB37pCmD_TLwBbT1TUm7vwu-QRuu4XYAAJW3BpP3P_dY1PGT9fKnq0NsgWh1f18KcQMMgvPQRvNz)             | 2024 | [IEEE Internet of Things Journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6488907)                           |
| When Federated Learning Meets Privacy-Preserving Computation                        | [link](https://dl.acm.org/doi/full/10.1145/3679013?casa_token=kSegkmGSJLAAAAAA%3AkwZRVc1ZgDKWwuDBXem_xyP0MsL-EuuJ3BCKxgCj5c2zAHFviQt1vUZ6b0f8H_kMmcqLxXQw-si7)                      | 2024 | [ACM Computing Surveys, Volume 56, Issue 12](https://dl.acm.org/toc/csur/2024/56/12)                                          |
| Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark | [link](https://ieeexplore.ieee.org/abstract/document/10571602?casa_token=3C52LwtXk4oAAAAA:y_ogDDU-FP8ZUlGfQvNGJA5fEhavLVJElFS_tX_kC8PjDlhvcXDGRdproFgWKZD7wgm6tg4gDWnb)             | 2024 | [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) |
| Decentralized Federated Learning: A Survey and Perspective                          | [link](https://ieeexplore.ieee.org/abstract/document/10542323?casa_token=-rjp4ddD8bwAAAAA:_tQw3V15mf16V30nSjFi_uGEgPQCIOgSYcNyOE322fc8dFBTJgzwOhgvEMJ2JpFCeIU9Jv9rBTXa)             | 2024 | [IEEE Internet of Things Journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6488907)                           |
| Emerging trends in federated learning: from model fusion to federated X learning    | [link](https://link.springer.com/article/10.1007/s13042-024-02119-1)                                                                                                                | 2024 | [International Journal of Machine Learning and Cybernetics](https://link.springer.com/journal/13042)                          |
| Class-Incremental Learning: A Survey                                                | [link](https://ieeexplore.ieee.org/abstract/document/10599804?casa_token=RjdcdVGTr-cAAAAA:6Sjf9d0JJAUoGmwWQNZdTtVEtbQXrjlPMien4cTTWd9-pQCEctsMHhTnUaygdW4_uN9jJusbzscN)             | 2024 | [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) |

---


# References

For the repo that is being referenced, please see the [FCIL](https://github.com/conditionWang/FCIL) repo.

and cite them as follows:

```
@InProceedings{dong2022federated,
    author = {Dong, Jiahua and Wang, Lixu and Fang, Zhen and Sun, Gan and Xu, Shichao and Wang, Xiao and Zhu, Qi},
    title = {Federated Class-Incremental Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022},
}
```
