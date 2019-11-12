#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.distributions import constraints

#import dmm.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


# In[3]:


def model_0(sequences, lengths, args, batch_size=None, include_prior=True):
    assert not torch._C._get_tracing_state()
    num_sequences, max_length, data_dim = sequences.shape
    with poutine.mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, data_dim])
                                  .to_event(2))
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    for i in pyro.plate("sequences", len(sequences), batch_size):
        length = lengths[i]
        sequence = sequences[i, :length]
        x = 0
        for t in pyro.markov(range(length)):
            # On the next line, we'll overwrite the value of x with an updated
            # value. If we wanted to record all x values, we could instead
            # write x[t] = pyro.sample(...x[t-1]...).
            x = pyro.sample("x_{}_{}".format(i, t), dist.Categorical(probs_x[x]),
                            infer={"enumerate": "parallel"})
            with tones_plate:
                pyro.sample("y_{}_{}".format(i, t), dist.Bernoulli(probs_y[x.squeeze(-1)]),
                            obs=sequence[t])


# In[8]:


def model(sequences, lengths, args, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    with poutine.mask(mask=include_prior):
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        probs_y = pyro.sample("probs_y",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, 2, data_dim])
                                  .to_event(3))
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        lengths = lengths[batch]
        x, y = 0, 0
        for t in pyro.markov(range(max_length if args.jit else lengths.max())):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                                infer={"enumerate": "parallel"})
                # Note the broadcasting tricks here: to index probs_y on tensors x and y,
                # we also need a final tensor for the tones dimension. This is conveniently
                # provided by the plate associated with that dimension.
                with tones_plate as tones:
                    y = pyro.sample("y_{}".format(t), dist.Bernoulli(probs_y[x, y, tones]),
                                    obs=sequences[batch, t]).long()


# In[10]:




models = {name[len('model'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}

def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    data = poly.load_data(poly.JSB_CHORALES)

    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(data['train']['sequences'])))
    sequences = data['train']['sequences']
    lengths = data['train']['sequence_lengths']

    # find all the notes that are present at least once in the training set
    present_notes = ((sequences == 1).sum(0).sum(0) > 0)
    # remove notes that are never played (we remove 37/88 notes)
    sequences = sequences[..., present_notes]

    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
        sequences = sequences[:, :args.truncate]
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    if args.print_shapes:
        first_available_dim = -2 if model is model_0 else -3
        guide_trace = poutine.trace(guide).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        model_trace = poutine.trace(
            poutine.replay(poutine.enum(model, first_available_dim), guide_trace)).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info(model_trace.format_shapes())

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=1 if model is model_0 else 2,
                strict_enumeration_warning=(model is not model_7),
                jit_options={"time_compilation": args.time_compilation})
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    if args.jit and args.time_compilation:
        logging.debug('time to compile: {} s.'.format(elbo._differentiable_loss.compile_time))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False)
    logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    logging.info('-' * 40)
    logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
    sequences = data['test']['sequences'][..., present_notes]
    lengths = data['test']['sequence_lengths']
    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)
    num_observations = float(lengths.sum())

    # note that since we removed unseen notes above (to make the problem a bit easier and for
    # numerical stability) this test loss may not be directly comparable to numbers
    # reported on this dataset elsewhere.
    test_loss = elbo.loss(model, guide, sequences, lengths, args=args, include_prior=False)
    logging.info('test loss = {}'.format(test_loss / num_observations))

    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(value.reshape(-1).size(0)
                   for value in pyro.get_param_store().values())
    logging.info('{} capacity = {} parameters'.format(model.__name__, capacity))


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.4.1')
#     parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
#     parser.add_argument("-m", "--model", default="1", type=str,
#                         help="one of: {}".format(", ".join(sorted(models.keys()))))
#     parser.add_argument("-n", "--num-steps", default=50, type=int)
#     parser.add_argument("-b", "--batch-size", default=8, type=int)
#     parser.add_argument("-d", "--hidden-dim", default=16, type=int)
#     parser.add_argument("-nn", "--nn-dim", default=48, type=int)
#     parser.add_argument("-nc", "--nn-channels", default=2, type=int)
#     parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
#     parser.add_argument("-t", "--truncate", type=int)
#     parser.add_argument("-p", "--print-shapes", action="store_true")
#     parser.add_argument("--seed", default=0, type=int)
#     parser.add_argument('--cuda', action='store_true')
#     parser.add_argument('--jit', action='store_true')
#     parser.add_argument('--time-compilation', action='store_true')
#     parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    args = parser.parse_args()
    main(args)


# In[ ]:




