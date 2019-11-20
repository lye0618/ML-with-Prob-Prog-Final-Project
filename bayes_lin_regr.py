import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, LogNormal
from pyro.infer import (EmpiricalMarginal, config_enumerate)
import pyro.optim as optim
from torch.distributions import constraints

from preprocess import preprocess


def get_data():
    print('Loading data')
    x, y, x_test, y_test = preprocess(ret_type='tensor')
    return x, y, x_test, y_test


@config_enumerate
def model(x, y):
    w_prior1 = pyro.sample('w_prior1',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior2 = pyro.sample('w_prior2',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior3 = pyro.sample('w_prior3',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior4 = pyro.sample('w_prior4',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior5 = pyro.sample('w_prior5',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior6 = pyro.sample('w_prior6',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior7 = pyro.sample('w_prior7',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior8 = pyro.sample('w_prior8',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior9 = pyro.sample('w_prior9',
                           Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior10 = pyro.sample('w_prior10',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior11 = pyro.sample('w_prior11',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior12 = pyro.sample('w_prior12',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior13 = pyro.sample('w_prior13',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior14 = pyro.sample('w_prior14',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior15 = pyro.sample('w_prior15',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior16 = pyro.sample('w_prior16',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior17 = pyro.sample('w_prior17',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior18 = pyro.sample('w_prior18',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))
    w_prior19 = pyro.sample('w_prior19',
                            Normal(torch.tensor(0.0), torch.tensor(1.0)))

    b_prior = pyro.sample('b_prior',
                          Normal(torch.tensor([1.0]), torch.tensor([5.0])))

    prediction_mean = b_prior + w_prior1 * x[:, 0] + \
        w_prior2 * x[:, 1] + w_prior3 * x[:, 2] + \
        w_prior4 * x[:, 3] + w_prior5 * x[:, 4] + \
        w_prior6 * x[:, 5] + w_prior7 * x[:, 6] + \
        w_prior8 * x[:, 7] + w_prior9 * x[:, 8] + \
        w_prior10 * x[:, 9] + w_prior11 * x[:, 10] + \
        w_prior12 * x[:, 11] + w_prior13 * x[:, 12] + \
        w_prior14 * x[:, 13] + w_prior15 * x[:, 14] + \
        w_prior16 * x[:, 15] + w_prior17 * x[:, 16] + \
        w_prior18 * x[:, 17] + w_prior19 * x[:, 18]

    sigma = pyro.sample("sigma", LogNormal(0., 5.))

    with pyro.plate("map", len(x)):
        pyro.sample("obs", Normal(prediction_mean, sigma), obs=y)


def guide(x, y):
    weights_loc = pyro.param('weights_loc', torch.randn(x.shape[1]))
    weights_scale = pyro.param('weights_scale', torch.ones(1),
                               constraint=constraints.positive)

    bias_loc = pyro.param('bias_loc', torch.randn(1))
    bias_scale = pyro.param('bias_scale', torch.ones(1),
                            constraint=constraints.positive)

    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.0),
                           constraint=constraints.positive)

    w_prior1 = pyro.sample('w_prior1', Normal(weights_loc[0], weights_scale))
    w_prior2 = pyro.sample('w_prior2', Normal(weights_loc[1], weights_scale))
    w_prior3 = pyro.sample('w_prior3', Normal(weights_loc[2], weights_scale))
    w_prior4 = pyro.sample('w_prior4', Normal(weights_loc[3], weights_scale))
    w_prior5 = pyro.sample('w_prior5', Normal(weights_loc[4], weights_scale))
    w_prior6 = pyro.sample('w_prior6', Normal(weights_loc[5], weights_scale))
    w_prior7 = pyro.sample('w_prior7', Normal(weights_loc[6], weights_scale))
    w_prior8 = pyro.sample('w_prior8', Normal(weights_loc[7], weights_scale))
    w_prior9 = pyro.sample('w_prior9', Normal(weights_loc[8], weights_scale))
    w_prior10 = pyro.sample('w_prior10', Normal(weights_loc[9], weights_scale))
    w_prior11 = pyro.sample('w_prior11',
                            Normal(weights_loc[10], weights_scale))
    w_prior12 = pyro.sample('w_prior12',
                            Normal(weights_loc[11], weights_scale))
    w_prior13 = pyro.sample('w_prior13',
                            Normal(weights_loc[12], weights_scale))
    w_prior14 = pyro.sample('w_prior14',
                            Normal(weights_loc[13], weights_scale))
    w_prior15 = pyro.sample('w_prior15',
                            Normal(weights_loc[14], weights_scale))
    w_prior16 = pyro.sample('w_prior16',
                            Normal(weights_loc[15], weights_scale))
    w_prior17 = pyro.sample('w_prior17',
                            Normal(weights_loc[16], weights_scale))
    w_prior18 = pyro.sample('w_prior18',
                            Normal(weights_loc[17], weights_scale))
    w_prior19 = pyro.sample('w_prior19',
                            Normal(weights_loc[18], weights_scale))

    b_prior = pyro.sample('b_prior', Normal(bias_loc, bias_scale))

    sigma = pyro.sample("sigma", LogNormal(sigma_loc, torch.tensor(0.05)))

    prediction_mean = b_prior + w_prior1 * x[:, 0] + \
        w_prior2 * x[:, 1] + w_prior3 * x[:, 2] + \
        w_prior4 * x[:, 3] + w_prior5 * x[:, 4] + \
        w_prior6 * x[:, 5] + w_prior7 * x[:, 6] + \
        w_prior8 * x[:, 7] + w_prior9 * x[:, 8] + \
        w_prior10 * x[:, 9] + w_prior11 * x[:, 10] + \
        w_prior12 * x[:, 11] + w_prior13 * x[:, 12] + \
        w_prior14 * x[:, 13] + w_prior15 * x[:, 14] + \
        w_prior16 * x[:, 15] + w_prior17 * x[:, 16] + \
        w_prior18 * x[:, 17] + w_prior19 * x[:, 18]

    return prediction_mean, sigma, y


def run_svi(itr, x, y):
    print(f'Running SVI for {itr} iterations')
    pyro.enable_validation(True)
    pyro.clear_param_store()
    global svi
    svi = pyro.infer.SVI(model,
                         guide,
                         optim.Adam({"lr": .005}),
                         loss=pyro.infer.Trace_ELBO(max_plate_nesting=1),
                         num_samples=1000)

    pyro.clear_param_store()
    rec_loss = []
    for i in range(itr):
        loss = svi.step(x, y)
        rec_loss.append(loss)

    return rec_loss


def plot_losses(rec_loss):
    print('Plotting losses')
    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.plot(rec_loss)
    plt.title('Loss Function')
    plt.xlabel('Iterations')
    plt.ylabel('-ELBO')
    plt.show()


def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(
            percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[
            ["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats


def get_avg_estimates(x, y):
    print('Getting posterior estimates')
    estimates = svi.run(x, y)
    sites = ['w_prior1', 'w_prior2', 'w_prior3', 'w_prior4', 'w_prior5',
             'w_prior6',
             'w_prior7', 'w_prior8', 'w_prior9', 'w_prior10', 'w_prior11',
             'w_prior12', 'w_prior13', 'w_prior14', 'w_prior15', 'w_prior16',
             'w_prior17', 'w_prior18', 'w_prior19', 'b_prior', 'sigma']

    svi_samples = {site: EmpiricalMarginal(estimates,
                                           sites=site).enumerate_support()
                                                      .detach().cpu().numpy()
                   for site in sites}

    summ = summary(svi_samples)

    post_means = []
    for site in summ:
        if site != 'sigma' and site != 'b_prior':
            post_means.append(summ[site]['mean'].values)

    post_means = np.array(post_means)
    post_means.flatten()
    post_means = torch.tensor(post_means)
    post_weights = post_means.type(torch.FloatTensor)
    sigma = torch.tensor(summ['sigma']['mean'].values).type(torch.FloatTensor)
    bias = torch.tensor(summ['b_prior']['mean'].values).type(torch.FloatTensor)

    return post_weights, sigma, bias


def get_pred(bias, weights, x_test):
    print('Getting predictions')
    preds = bias + torch.mm(x_test, weights)
    return preds


def main(itr=1000):
    x, y, x_test, y_test = get_data()
    losses = run_svi(itr, x, y)
    plot_losses(losses)

    post_weights, post_sigma, post_bias = get_avg_estimates(x, y)
    predictions = get_pred(post_bias, post_weights, x_test)

    return predictions
