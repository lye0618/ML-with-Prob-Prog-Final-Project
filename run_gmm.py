import pyro
import torch

from models_manual_guide import GMM


def run_gmm(n_comp=11, n_itr=100):
    """
    Runs GMM end to end
    :param  n_comp: Number of components in GMM (default = 11)
            n_itr: Number of iterations (default = 100)

    :return:    resps = posterior probabilities on test set
                nl_idx = index of null rows in resps
    """
    print(f'running for n_comp = {n_comp}')
    pyro.clear_param_store()
    svi = GMM(n_comp=n_comp, infr='svi', n_itr=n_itr)

    print(f'Size of train tensor: {svi.tensor_train.shape}')
    print(f'Size of test tensor: {svi.tensor_test.shape}')

    svi.svi_itr = n_itr
    loss = svi.inference()
    svi.plot_svi_convergence(loss)
    print('\n Losses stored in variable loss')

    samples = svi.get_mean_svi_est_manual_guide()
    print(f'Posterior samples stored in '
          f'variable samples with shape {len(samples)}')

    resps = svi.get_posterior_resp()

    print(f'Number of nulls in resps: {torch.isnan(resps).sum()}')

    svi.mcmc = None
    stats = svi.generate_stats()
    print(stats[0])
    nl_idx = []
    for _, i in enumerate(resps):
        if torch.isnan(i).sum():
            nl_idx.append(_)

    return resps, nl_idx
