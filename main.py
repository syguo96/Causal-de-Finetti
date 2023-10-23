from multivariate_experiment import *
from bivariate_experiment import *

def main():
    """ Main function for bivariate and multivariate multi-environment causal discovery algorithms.
    """
    _ = bivariate_multienv(scm_bivariate_continuous, "./experiments/results/", ['Causal-de-Finetti', 'FCI', 'GES', 'NOTEARS', 'DirectLinGAM', 'CD-NOD'],
                           [100, 200, 300, 400, 500])
    #_ = multivariate_multienv(scm_multivariate_binary, "./experiments/results/", ['Causal-de-Finetti', 'PC', 'FCI', 'GES', 'NOTEARS'], [2000, 4000, 6000, 8000, 10000], num_var=3)

if __name__ == "__main__":
    main()