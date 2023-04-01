"""
functions for train, test and log
"""

import os
import pprint
import numpy as np
import pandas as pd
from typing import Literal
from tqdm.notebook import tqdm

from sklearn.metrics import f1_score

import wandb

import torch
from torch.optim.lr_scheduler import StepLR

def create_labels_mapping(dataset: pd.DataFrame()) -> dict():
    """
    create label2id and id2label dicts 
    for mapping between categories and labels
    Parameters
    ----------
        dataset (pd.DataFrame): 
            dataframe with all content inside
        
    Return
    ------
        label2id (dict()): 
            label <-> id mapping
        id2label (dict()): 
            id <-> label mapping
    """

    labels = sorted(list(set(dataset['category_id'])))
    label2id, id2label = dict(), dict()

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print('Number of labels:', len(labels))

    return label2id, id2label

def trainer(model, train_loader, valid_loader, loss_function,
            optimizer, scheduler, cfg):
    """
    Parameters
    ----------
        trainer:
            iterating through epochs and call train_epoch
        cfg (dict()):
            keys():
                count_of_epoch:
                    numer of epochs
                batch_size:
                    batch size
                lr:
                    learing rate for optimizer
                model_name:
                    name of model, needed for saving weights in right folder
                device:
                    use cpu or gpu for training
            
        train_loader:
            train split
        valid_loader:
            validation split
        model:
            model for training
        loss_function:
            criterion for optimisation
        
        optimizer:
            optimizer
        scheduler:
            learning rate scheduler for optimizer
    """

    wandb.watch(model, loss_function, log="all", log_freq=10)

    min_valid_loss = np.inf

    # in this folder will save model weights
    if not os.path.exists(f'/content/model_weights/{cfg["model_name"]}'):
        os.makedirs(f'/content/model_weights/{cfg["model_name"]}')

    # main loop
    for e in tqdm(range(cfg['count_of_epoch']), desc='epochs'):

        # train
        epoch_loss = train_epoch(train_generator=train_loader,
                                 model=model,
                                 loss_function=loss_function,
                                 optimizer=optimizer,
                                 device=os.environ['device'])

        # validation
        valid_loss = 0.0
        model.eval()
        valid_loss, valid_f1 = tester(model=model,
                                        test_loader=valid_loader,
                                        loss_function=loss_function,
                                        device=os.environ['device'])

        scheduler.step()

        # log things
        trainer_log(epoch_loss, valid_loss, valid_f1, e,
                    optimizer.param_groups[0]['lr'], cfg)

        # saving models
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(model.state_dict(),
                       f'/content/model_weights/{cfg["model_name"]}/saved_model_{e}.pth')
            wandb.log_artifact(f'/content/model_weights/{cfg["model_name"]}/saved_model_{e}.pth',
                                name=f'saved_model_{e}', type='model')
        print()

def train_epoch(train_generator, model,
                loss_function, optimizer, device):
    """
    iterating through batches inside batch generator 
    and call train_on_batch
    Parameters
    ----------
        train_generator:
            batch generator for training
        model:
            model for training
        loss_function:
            criterion function
        optimizer:
            optimizer
    Return
    ------
        mean loss over epoch
    """
    epoch_loss = 0
    total = 0
    for batch in train_generator:
        batch_loss = train_on_batch(model,
                                    batch['pixel_values'], batch['label'],
                                    optimizer, loss_function, device)

        epoch_loss += batch_loss*len(batch['pixel_values'])
        total += len(batch['pixel_values'])

    return epoch_loss/total

def train_on_batch(model, x_batch, y_batch,
                   optimizer, loss_function, device):
    """
    train on single batch
    Parameters
    ----------
        model:
            model for traning
        x_batch:
            batch features
        y_batch:
            batch targets
        optimizer:
            optimizer
        loss_function:
            criterion
    Return
    ------
        loss on the batch
    """
    model.train()
    optimizer.zero_grad()

    output = model(x_batch.to(device))

    loss = loss_function(output, y_batch.to(device))
    loss.backward()

    optimizer.step()
    return loss.cpu().item()

def tester(model, test_loader, loss_function = None, 
           print_stats=False, device='cpu'):
    """
    testing or validating on provided dataset
    also if needed print some statistics
    Parameters
    ----------
        model:
            model for traning
        test_loader:
            dataset for testing or validating
        loss_function:
            criterion for calculating loss function on validation
        print_stats (bool):
            rather print statistics or not
        device (str):
            use cpu or gpu for testing
    Return
    ------
        mean loss over validation dataset and f1_score or just f1_score
    """
    pred = []
    real = []
    loss = 0
    model.eval()
    for batch in test_loader:

        x_batch = batch['pixel_values'].to(device)
        with torch.no_grad():
            output = model(x_batch)

            if loss_function is not None:
                loss += loss_function(output, batch['label'].to(device))

        pred.extend(torch.argmax(output, dim=-1).cpu().numpy().tolist())
        real.extend(batch['label'].cpu().numpy().tolist())

    F1 = f1_score(real, pred, average='weighted')

    if print_stats:
        print(F1)

    if loss_function is not None:
        return loss.cpu().item()/len(test_loader), F1
    else:
        wandb.log({'test_f1': F1})
        return F1

def trainer_log(train_loss, valid_loss, valid_f1, epoch, lr, cfg):
    """
    make logging
    """
    wandb.log({'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_f1': valid_f1, 
                'epoch': epoch, 
                'learning_rate': lr})

    print(f'train loss on {str(epoch).zfill(3)} epoch: {train_loss:.6f} with lr: {lr:.10f}')
    print(f'valid loss on {str(epoch).zfill(3)} epoch: {valid_loss:.6f}')
    print(f'valid f1 score: {valid_f1:.2f}')

def train_pipeline(model, train_dataset, valid_dataset, cfg,
                   saved_model=None, to_train=True, to_test=True):
    """
    run training and/or testing process
    Parameters
    ----------
        model:
            model for traning and/ot testing
        train_dataset:
            dataset for trainig
        test_dataset:
            dataset for testing or validating
        saved_model:
            path to saved checkpoint to resume training
            or to test saved model
        
        to_train (bool):
            if True train, else only test
        to_test (bool):
            if True test, else only train
    Return
    ------
        trained or tested model
    """

    def make(model, train_dataset, valid_dataset):
        """
        make dataloaders, init optimizers and criterions
        """

        trainloader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=cfg.batch_size,
                                                  shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid_dataset, 
                                                  batch_size=cfg.batch_size,
                                                  shuffle=False, num_workers=2)

        criterion = torch.nn.CrossEntropyLoss()
        
        if cfg.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        scheduler = StepLR(optimizer, cfg.step_size, cfg.step_gamma)

        return trainloader, validloader, criterion, optimizer, scheduler

    # pretty print dict()
    pretty_print = pprint.PrettyPrinter()

    print('config:')
    pretty_print.pprint(cfg)
    print()
    print('running on device:', os.environ['device'], '\n')

    # data and optimization
    trainloader, validloader,  \
        criterion, optimizer, scheduler = make(model, train_dataset,
                                               valid_dataset)

    if to_train:
        trainer(model, trainloader, validloader,
                criterion, optimizer, scheduler, cfg)

    if to_test:
        tester(model, validloader, print_stats=True, 
                device=os.environ['device'])

    return model