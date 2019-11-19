import pyro
import torch

from models_manual_guide import GMM


def run_gmm(n_comp=11):
    """
    Runs GMM end to end
    :param n_comp: Number of components in GMM (default = 11)

    :return:    resps = posterior probabilities on test set
                nl_idx = index of null rows in resps
    """
    print(f'running for n_comp = {n_comp}')
    pyro.clear_param_store()
    svi = GMM(n_comp=n_comp, infr='svi')

    print(f'size of train tensor: {svi.tensor_train.shape}')
    print(f'size of test tensor: {svi.tensor_test.shape}')
    print('number of iterations 1000')
    svi.svi_itr = 1000
    loss = svi.inference()
    svi.plot_svi_convergence(loss)
    print('\n losses stored in variable loss')

    samples = svi.get_mean_svi_est_manual_guide()
    print(f'posterior samples stored in '
          f'variable samples with shape {len(samples)}')

    resps = svi.get_posterior_resp()
    print('posterior probabilities stored in variable resps')

    print(f'Number of nulls in resps: {torch.isnan(resps).sum()}')

    nl_idx = []
    for _, i in enumerate(resps):
        if torch.isnan(i).sum():
            nl_idx.append(_)

    print('Indexes for test tensor stored in variable nl_idx')

    return resps, nl_idx
