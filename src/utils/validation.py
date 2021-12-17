"""Utility modules and functions for validation."""


import torch
from torch import nn
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sktime.transformations.panel.rocket import MiniRocketMultivariate


class FrechetInceptionDistance:
    """"
    Frechet inception distance module/closure.

    Notes
    -----
    1. See for reference Heusel, M., Ramsauer, H., Unterthiner, T., Nessler,
       B., & Hochreiter, S. (2017). GANs trained by a two time-scale update
       rule converge to a local nash equilibrium. Advances in neural
       information processing systems, 30.
    2. The formula approximates the data with multivariate Gaussian
       distributions.
    """

    def __init__(self, feature: nn.Module):
        """
        Arguments
        ---------
        feature: Feature extractor acted upon real/generated data
        """
        super().__init__()
        self.feat = feature
    
    def __call__(self, real: torch.Tensor, fake: torch.Tensor):
        """
        Return the Frechet incepction distance between features of the real
        and the generated data.

        Pre-condition
        -------------
        `real` and `fake` must have the same shape.
        """
        # Flatten the feature vector for individual samples
        real = self.feat(real).view(real.size(0), -1)
        fake = self.feat(fake).view(fake.size(0), -1)
        # Compute the Frechet inception distance
        dist = np.linalg.norm(real.mean(dim=0) - fake.mean(dim=0)) ** 2
        cov_r, cov_f = np.cov(real.T), np.cov(fake.T)  # covariance matrices
        dist += np.trace(cov_r + cov_f - 2 * sp.linalg.sqrtm(cov_r.dot(cov_f)))
        return dist


class MiniRocket(nn.Module):
    """
    MiniRocket module to extract features from time series.

    Notes
    -----
    1. This adopted implementation supposes that data (e.g., validation set)
       only has a single batch.
    2. See for reference Dempster, A., Schmidt, D. F., & Webb, G. I. (2021).
       Minirocket: A very fast (almost) deterministic transform for time series
       classification. In Proceedings of the 27th ACM SIGKDD Conference on
       Knowledge Discovery & Data Mining (pp. 248-257).
    """

    def __init__(self, **kwargs):
        """
        Arguments
        ---------
        kwargs: Keyword arguments passed to MiniRocketMultivariate
        """
        super().__init__()
        self.minirocket = MiniRocketMultivariate(**kwargs)

    def forward(self, data: torch.Tensor):
        """Return the transformed features from time series data."""
        data = pd.DataFrame({  # convert time series to sktime format
            f'dim_{var_idx}': [
                pd.Series(data[ts_idx, var_idx])
                for ts_idx in range(data.size(0))
            ]
            for var_idx in range(data.size(1))
        })
        if not self.minirocket.check_is_fitted(): self.minirocket.fit(data)
        feats = self.minirocket.transform(data)
        feats = data.new_tensor(feats.values)  # convert features to tensor
        return feats


def roc_performance(labels: torch.Tensor, scores: torch.Tensor):
    """
    Return the area under ROC curve and the accuracy at the cut-off point that
    maximizes the sum of sensitivity and specificity.

    Arguments
    ---------
    labels: Binary annotation of data that takes value of either 0 or 1
    scores: Predicted probabilities/scores for classification

    Pre-condition
    -------------
    `labels` and `scores` must have the same shape.
    """
    fpr, tpr, threshold = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    cutoff = threshold[np.argmax(tpr - fpr)]
    acc = ((scores >= cutoff) == labels).float().mean().item()
    return roc_auc, acc


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()