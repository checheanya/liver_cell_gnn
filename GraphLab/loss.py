import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import GraphLab.register as register
from GraphLab.config import cfg
from GraphLab.utils.utils import seed_anything

seed_anything(cfg.seed)
indexs = 0


def cross_entropy_loss(y_pred, y_true):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    return -torch.sum(y_true * torch.log(y_pred + 1e-10) + (1 - y_true) * torch.log(1 - y_pred + 1e-10))


def log_likelihood_loss(y_true, y_pred):
    # NLL / cross-entropy style loss
    losses = torch.nn.functional.nll_loss(torch.log(y_pred), y_true)
    # Mean over batch
    return losses.mean()


def compute_loss(model, pred, true):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    """
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)
    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    # pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    # true = true.squeeze(-1) if true.ndim > 1 else true

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        # binary or multilabel
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    elif cfg.model.loss_fun == 'cox':
        return CoxLoss(pred, true, pred.device), pred
    elif cfg.model.loss_fun == 'CensoredCrossEntropyLoss':
        return CensoredCrossEntropyLoss(pred, true), pred
    elif cfg.model.loss_fun == 'multi_task':
        return (CoxLoss(pred[:, 0], true, pred.device) + 0.5*CensoredCrossEntropyLoss(pred[:, 1:6], true) + 0.5 * bce_loss(pred[:, 6], true[:, 1]))/2, pred
    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))


# CPU variant (commented out)
# def CoxLoss(hazard_pred=None, labels=None, device=None):
#     survtime = labels[:, 0]
#     censor = labels[:, 1]
#     current_batch_len = len(survtime)
#     # DSLoss=DeepSurvLoss(hazard_pred,survtime,censor)
#     R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
#     for i in range(current_batch_len):
#         for j in range(current_batch_len):
#             R_mat[i, j] = survtime[j] >= survtime[i]
#     R_mat = torch.FloatTensor(R_mat).to(device)
#     theta = hazard_pred.reshape(-1)
#     exp_theta = torch.exp(theta)
#     loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
#     # target_labels = labels[:, 2].unsqueeze(1)
#     # loss_mse = F.mse_loss(hazard_pred, target_labels)
#     # return 0.1 * loss_cox + loss_mse
#     return loss_cox

def CoxLoss(hazard_pred=None, labels=None, device=None):
    survtime = labels[:, 0]
    censor = labels[:, 1]
    current_batch_len = len(survtime)

    # Risk set matrix R
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    # Time-difference matrix Delta T
    DeltaT_mat = np.zeros([current_batch_len, current_batch_len])

    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]
            DeltaT_mat[i, j] = survtime[j] - survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    DeltaT_mat = torch.FloatTensor(DeltaT_mat).to(device)

    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)

    # Standard Cox partial likelihood term
    adjusted_loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
    # DeltaT-weighted Cox term
    weighted_loss_cox = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat / (1 + torch.abs(DeltaT_mat)), dim=1))) * censor)

    # Sum both Cox terms
    total_loss = adjusted_loss_cox + weighted_loss_cox

    return total_loss


def DeepSurvLoss(risk_pred, y, e):
    mask = torch.ones(y.shape[0], y.shape[0]).to(torch.device(cfg.device))
    mask[(y.T - y) > 0] = 0
    log_loss = torch.exp(risk_pred) * mask
    log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
    log_loss = torch.log(log_loss).reshape(-1, 1)
    neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
    return neg_log_loss

def TransformLabel(y):
    '''
    :param y: Columns [time, event/censoring indicator]
    :return: Binned time tensor and observation mask
    '''
    survtime = y[:, 0]
    obs = y[:, 1]
    n = y.shape[0]

    # Zero-initialized time bins
    new_time = torch.zeros((n, 5)).to(y.device)

    # Uncensored: set bin from floor(time/25)
    uncensored_mask = (obs > 0)
    uncensored_time = survtime[uncensored_mask]
    uncensored_labels = torch.clamp((uncensored_time / 25).floor(), max=4).long()
    if len(uncensored_labels) is not 0:
        new_time[uncensored_mask, uncensored_labels] = 1

    # Censored: set bins from ceil(time/25)
    censored_mask = ~uncensored_mask
    censored_time = survtime[censored_mask]
    censored_labels = torch.clamp((censored_time / 25).ceil(), max=4).long()

    if len(censored_labels) is not 0:
        # Index grid for bin dimensions
        index_matrix = torch.arange(5).unsqueeze(0).expand(len(censored_labels), -1).to(y.device)

        # Mask: bins at or after censoring time
        censored_mask_2d = (index_matrix >= censored_labels.unsqueeze(1))

        new_time[censored_mask] = censored_mask_2d.float()

    return new_time, obs


def L(X, t, EPS=1e-12):
    loss = -(t * torch.log(X + EPS) + (1 - t) * torch.log(1 - X + EPS))
    return loss


# def CensoredCrossEntropyLoss(x, y, EPS=1e-12):
#     # x: predicted
#     # y: labels
#     # obs: 1 observed event (uncensored), 0 unobserved event (right-censored)
#     # EPS: avoid log(0)
#
#     survtime, obs = TransformLabel(y)
#     x = F.softmax(x, dim=1).clamp(min=EPS, max=1 - EPS)
#     loss = L(x, survtime, EPS)
#
#     uncensored_mask = (obs > 0.5)
#     censored_mask = ~uncensored_mask
#
#     loss_uncensored = loss[uncensored_mask].sum()
#     loss_censored = loss[censored_mask].sum()
#
#     N1 = obs.sum()
#     N2 = len(x) - N1
#     return N2 / len(x) * loss_uncensored + 0.5 * N1 / len(x) * loss_censored

def CensoredCrossEntropyLoss(x, y, EPS=1e-12):
    # x: predicted
    # y: labels
    # obs: 1 observed event (uncensored), 0 unobserved event (right-censored)
    obs = y[:, 1]
    y = y[:, 0]
    y = (y/25).floor().long()
    x = F.softmax(x,dim=1).clamp(min=EPS)
    n = x.shape[0]
    loss_sum = 0
    for i in range(n):
        if obs[i] > 0.5:
            loss_sum += torch.log(x[i, y[i]])
        else:
            if y[i].item() == x.shape[1] - 1:
                # loss_sum += x[i, y[i]] * 0.0
                loss_sum += torch.log(x[i, y[i]].clamp(min=EPS))
            else:
                loss_sum += torch.log(torch.sum(x[i, y[i]+1:]).clamp(min=EPS))
    loss_val = loss_sum / n * -1
    return loss_val