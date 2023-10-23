### Causal de Finetti: On the identification of invariant causal structure in Exchangeable data

This is the code repository for the paper
_Causal de Finetti: On the identification of invariant causal structure 
in Exchangeable data_ (NeurIPS 2023). 

Performing Causal-de-Finetti algorithm on exchagneable data allows 
one to uniquely identify the correct causal structure in bivariate and 
multivariate settings. 

To reproduce the results, pip install the `requirements.txt` file and 
run the `main.py` file. The results are saved in `experiments/results` folder. 

## Quick Start
To run the causal de Finetti algorithm is as easy as running the following code snippet:

```
from src.models.causaldf import *
from experiments.synthetic_data_generation import *

num_env_bivariate = 1000
num_env_multivariate = 5000
num_sample_per_env = 2

## Bivariate
data = scm_bivariate_continuous(num_env = num_env_bivariate, num_sample = num_sample_per_env)
estimate, _ = run_causaldf_bivariate(data)
correct = True if estimate == data['true_structure'] else False

print('Is correct in bivariate:', correct)

## Multivariate
data = scm_multivariate_binary(num_env = num_env_multivariate, num_sample = num_sample_per_env, num_var = 3)
estimate, _ = run_causaldf_multivariate(data)
correct = True if estimate == data['true_structure'] else False

print('Is correct in multivariate:', correct)
```




