import torch.nn.init as init


def uniform_init_rnn(model, a=-0.08, b=0.08):
    """init network weight with uniform distribution and bias with zero

    Arguments:
        model {nn.Module} -- rnn based model

    Keyword Arguments:
        a {float} -- bottom value of uniform distribution (default: {-0.08})
        b {float} -- upper value of uniform distribution (default: {0.08})
    """

    for name, param in model.named_parameters():
        if 'weight' in name:
            init.uniform_(param, a, b)
        if 'bias' in name:
            init.zeros_(param)
