# Notes

<!-- TOC -->

- [Notes](#notes)
  - [2024](#2024)
    - [Federated Learning: Opportunities and Challenge](#federated-learning-opportunities-and-challenge)
    - [Fed class](#fed-class)
    - [FedET](#fedet)
  - [2022](#2022)
    - [FCIL](#fcil)

<!-- /TOC -->

<!-- /TOC -->

<!-- /TOC -->
<!-- /TOC -->

## 2024

### Federated Learning: Opportunities and Challenge
has a good introduction for the business use case of federated learning came to be. using google as an example

### Fed class

self-distillation

balance and privacy and performance is acceptable

### FedET

enhancers frozen pre-trained backbone

stronger privacy than Fedclass

## 2022

### FCIL

#### Summary
- citations at time of reading: 152
- novelty: 
    ```
    The novelty of GLFC framework in FCIL lies in its approach to address the catastrophic forgetting problem by balancing gloval and local knowledge, while still ensuring efficiency and privacy.
    ```
- problem:
    ```
    Catastrophic Forgetting caused by
    
    1. Priorizing learning new classes
        > Due to the limited availability of data from old classes in the client's local dataset, the local model tends to priorite learning new classes at the expense of the old ones.
        > this process involves adjusting the weights and features to optimize for the new data, resulting in a model overwriting and altering weights that were essential for correctly classifying the old classes
        > since the global model is an aggregation of these local models, it becomes susceptile to these same changes when theupdated mdoels from clients are aggregated. This means that the global mdoel progresibely forgets old casses because its parameters are increasingly optimized for the new classes that each client introduces.
        
    2. Lack of access to old data
        > in traditional, centralized class incremnal learning (CIL), a small subset of old class data is often retained (known as a memory buffer) and a replayed during training to mitigate catstriphic forgetting. However, in FCIL, due to privacy contstraints and storage limitations, clients may not retain much or any data related to old classes.
    
    3. Heterogenous Data Distribution Across Clients
        > 

    ```
- method: 
    ```
    Global-Local Forgetting Compensation (GLFC) to learn a global class incrmental model fo alleviating the catastrophic 
    forgetting caused by class imbalance
    ```
- results:
- conclusion:

Federated Learning is used due to data-privacy concerns
"Federated learning (FL) has attracted growing attentions via data-private collaborative training on decentralized clients."



#### Key Words

Non-i.i.d(Non-independent and identically distributed) data 





problem of current federated learning methods:
- static number of classes
- no. of classes in the framework does not change over time
- global model suffers from significant catastrophic forgetting old classes when local clients gather new data for new classes (not enough memory to store the old classes on client side)
- when clients participate in federated learning, it further aggrevates the catastrophic forgetting of the global model


