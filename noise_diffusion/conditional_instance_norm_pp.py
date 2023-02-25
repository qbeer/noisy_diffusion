import torch
import torch.nn as nn

class ConditionalInstanceNorm2d(nn.Module):
    """
        From: https://github.com/ermongroup/ncsn/blob/master/models/cond_refinenet_dilated.py#L48
    """
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out

"""
    In theory this should be very similar in nature to the above example.
"""
class ConditionalInstanceNormPP(nn.Module):
    def __init__(self, in_features, in_labels, eps=1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(size=(in_labels, in_features, 1, 1)))
        self.beta = nn.Parameter(torch.zeros(size=(in_labels, in_features, 1, 1)))
        self.gamma = nn.Parameter(torch.ones(size=(in_labels, in_features, 1, 1)))
        self.eps = eps

    def forward(self, x, y):
        """
            x : feature_maps: [bs, h, w, F]
            y : labels: [bs, 1]
        """
        y = y.view(-1).long()
        alphas = self.alpha[y, :] # [bs, F, 1, 1]
        betas = self.beta[y, :] # [bs, F, 1, 1]
        gammas = self.gamma[y, :] # [bs, F, 1, 1]

        mu = torch.mean(x, dim=(2, 3), keepdim=True) # [bs, F, 1, 1]
        s = torch.std(x, dim=(2, 3), keepdim=True) # [bs, F, 1, 1]

        m = torch.mean(mu, dim=1, keepdim=True) # [bs, 1, 1, 1]
        v = torch.std(mu, dim=1, keepdim=True) # [bs, 1, 1, 1]

        # [bs, F, 1, 1] * ( [bs, F, h, w] - [bs, F, 1, 1] ) / [bs, F, 1, 1] +
        # [bs, F, 1, 1] + [bs, F, 1, 1] * ( [bs, F, 1, 1] - [bs, 1, 1, 1] ) / [bs, 1, 1, 1]
        out = alphas * ( x - mu ) / (s + self.eps) + betas + gammas * ( mu - m ) / (v + self.eps) # [bs, F, h, w]

        return out