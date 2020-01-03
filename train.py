"""Main entrance for train/eval with/without KD on CIFAR-10"""

import argparse
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
from datasets import fetch_dataloader
import model.resnet as resnet
#from evaluate import evaluate, evaluate_kd


def loss_fn_kd(logits, labels, teacher_logits, params):
    """
    Compute the knowledge-distillation (KD) loss given logits, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha # 0.9
    T = params.temperature # 20

    # KLDivLoss can be computed as follows:
    #p = torch.softmax(logits / T, dim=0)             
    #q = torch.softmax(teacher_logits / T, dim=0)     
    #kldiv_loss = sum(p * torch.log(p / q))

    KD_loss = nn.KLDivLoss()(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (alpha * T * T) + F.cross_entropy(logits, labels) * (1. - alpha)

    return KD_loss



def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., num_classes-1]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)



# Helper function: get [batch_idx, teacher_logits_list] list by running teacher model once
def fetch_teacher_logits_list(teacher_model, dataloader, params):
    # set teacher_model to evaluation mode
    teacher_model.eval()
    teacher_logits_list = []
    for i, (X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()

        logits = teacher_model(X).data.cpu().numpy()
        teacher_logits_list.append(logits)

    return teacher_logits_list


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_logits_list, optimizer, dataloader, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        dataloader: 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()
    # teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (X, y) in enumerate(dataloader):
            # move to GPU if available
            X = X.cuda()
            y = y.cuda()

            # compute model output, fetch teacher output, and compute KD loss
            logits = model(X)

            # get one batch output from teacher_logits_list list
            teacher_logits = torch.from_numpy(teacher_logits_list[i]).cuda()

            loss = loss_fn_kd(logits, y, teacher_logits, params)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                logits = logits.data.cpu().numpy()
                y = y.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {'accuracy':accuracy(logits, y)}
                                 
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    print("- Train metrics: " + metrics_string)


def evaluate_kd(model, dataloader, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for i, (X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()

        # compute model output
        logits = model(X)

        # loss = loss_fn_kd(output_batch, y, output_teacher_batch, params)
        loss = 0.0  #force validation loss to zero to reduce computation time

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        logits = logits.data.cpu().numpy()
        y = y.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {'accuracy': accuracy(logits, y)}
        # summary_batch['loss'] = loss.item()
        summary_batch['loss'] = loss
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    print("- Eval metrics : " + metrics_string)
    return metrics_mean




def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer, params):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
    """
    best_val_acc = 0.0
    
    # fetch teacher outputs using teacher_model under eval() mode
    teacher_model.eval()
    teacher_logits_list = fetch_teacher_logits_list(teacher_model, train_dataloader, params) # (num_batches, batch_size, num_classes), before softmax

    # learning rate schedulers for different models:
    scheduler = StepLR(optimizer, step_size=100, gamma=0.2) 

    for epoch in range(params.num_epochs):

        scheduler.step()

        # Run one epoch
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_kd(model, teacher_logits_list, optimizer, train_dataloader, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, val_dataloader, params)





if __name__ == '__main__':
    params = {
        "model_version": "cnn_distill",
        "augmentation": "yes",
        "teacher": "resnet18",
        "alpha": 0.9,
        "temperature": 20,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "num_epochs": 30,
        "dropout_rate": 0.5, 
        "num_channels": 32,
        "save_summary_steps": 100,
        "num_workers": 4
    }
    # bracket => dot notation
    from types import SimpleNamespace
    params = SimpleNamespace(**params)


    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Create the input data pipeline
    print("Loading the datasets...")

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    train_dl = fetch_dataloader('train', params)
    
    dev_dl = fetch_dataloader('dev', params)

    print("- done.")

    # train a 5-layer CNN or a 18-layer ResNet with knowledge distillation
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    # fetch loss function and metrics definition in model files

    teacher_model = resnet.ResNet18()
    teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
    teacher_model = teacher_model.cuda() if params.cuda else teacher_model


    #utils.load_checkpoint(teacher_checkpoint, teacher_model)

    # Train the model with KD
    print("Experiment - model version: {}".format(params.model_version))
    print("Starting training for {} epoch(s)".format(params.num_epochs))
    print("First, loading the teacher model and computing its outputs...")
    train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, params)
