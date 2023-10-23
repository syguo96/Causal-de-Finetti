from typing import Callable
from src.models.causaldf import *
from src.models.cdnod import *
from src.models.directlingam import *
from src.models.fci import *
from src.models.ges import *
from src.models.notears import *
from src.utils.plot_utils import *
from src.utils.utils import *
from experiments.synthetic_data_generation import *
from tqdm import tqdm


def bivariate_multienv(data_generating_function: Callable, plot_dir: str, methods: List, env_range: List) -> Dict:
    """
    Run gtest CI to identify unique causal structure using multi environment with 2 samples per environment
    :param data_generating_function:
    :param plot_dir:
    :return:
    """
    set_seed(42)
    num_experiments = 100
    results = {'correct_counts': {'Env': [np.repeat(env, num_experiments)for env in env_range]},
               'computation_time': {'num_samples': [2*env for env in env_range]}
               }
    functions = {'Causal-de-Finetti': run_causaldf_bivariate,
                 'CD-NOD': run_cd_nod,
                 'DirectLinGAM': run_directLinGAM,
                 'FCI': run_fci,
                 'GES': run_ges,
                 'NOTEARS': run_notears,
                 'Random': run_random_bivariate
                 }

    initialize_results(results, methods)
    verbose = True

    for env in tqdm(env_range):
        counts_per_env, time_per_env = initialize_records_per_env(methods)
        for exp in tqdm(range(num_experiments)):
            estimates= []
            times = []
            # If data makes cdnod fail to produce valid results, then try again
            while None in estimates or estimates == []:
                try:
                    data = data_generating_function(num_env = env, num_sample = 2)
                    for method in methods:
                        estimate, time = functions[method](data)
                        estimates.append(estimate)
                        times.append(time)
                    else:
                        raise NotImplementedError('Please provide a valid method')
                except:
                    pass
            for (name, estimate, time) in zip(methods, estimates, times):
                # print(name, estimate)
                # print('true_structure', data['true_structure'])
                counts_per_env[name].append(estimate == data['true_structure'])
                time_per_env[name].append(time)

        print('Environment %d:' %(env))
        # print('\nTotal time spent: %.2f' % (end_time - start_time))

        for method in methods:
            num_correct = np.mean(counts_per_env[method])
            results['correct_counts'][method].append(counts_per_env[method])
            results['computation_time'][method].append(np.mean(time_per_env[method]))
            if verbose:
                print("\n Correct structure identification " + method, num_correct)

    for k, v in results['correct_counts'].items():
        results['correct_counts'][k] = list(flatten(v))

    results['correct_counts']['Random'] = 1./3
    correct_counts_plotter(results['correct_counts'], plot_dir, data_generating_function)
    computation_time_plotter(results['computation_time'], plot_dir, data_generating_function)
    np.save(f'{plot_dir}numpy_results_{data_generating_function.__name__}', results)


