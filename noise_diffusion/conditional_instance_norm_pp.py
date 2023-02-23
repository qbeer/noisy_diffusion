import torch
import torch.nn as nn

class ConditionalInstanceNormPP(nn.Module):
    def __init__(self, in_features, in_labels):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(in_labels, in_features, 1, 1) / torch.sqrt(torch.tensor(in_labels + in_features)))
        self.beta = nn.Parameter(torch.randn(in_labels, in_features, 1, 1) / torch.sqrt(torch.tensor(in_labels + in_features)))
        self.gamma = nn.Parameter(torch.randn(in_labels, in_features, 1, 1) / torch.sqrt(torch.tensor(in_labels + in_features)))

    def forward(self, x, y):
        """
            x : feature_maps: [bs, h, w, F]
            y : labels: [bs, 1]
        """
        y = y.view(-1)
        alphas = self.alpha[y, :] # [bs, F, 1, 1]
        betas = self.beta[y, :] # [bs, F, 1, 1]
        gammas = self.gamma[y, :] # [bs, F, 1, 1]

        mu = torch.mean(x, dim=(2, 3), keepdim=True) # [bs, F, 1, 1]
        s = torch.std(x, dim=(2, 3), keepdim=True) # [bs, F, 1, 1]

        m = torch.mean(mu, dim=-1, keepdim=True) # [bs, 1, 1, 1]
        v = torch.std(mu, dim=-1, keepdim=True) # [bs, 1, 1, 1]

        # [bs, F, 1, 1] * ( [bs, F, h, w] - [bs, F, 1, 1] ) / [bs, F, 1, 1] +
        # [bs, F, 1, 1] + [bs, F, 1, 1] * ( [bs, F, 1, 1] - [bs, 1, 1, 1] ) / [bs, 1, 1, 1]
        out = alphas * ( x - mu ) / s + betas + gammas * ( mu - m ) / v # [bs, F, h, w]
        return out