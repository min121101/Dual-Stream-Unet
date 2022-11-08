# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from tqdm import tqdm
from loss_function import *
from collections import defaultdict
from colorama import Fore, Back, Style
import gc
import numpy as np
import wandb
import copy
import time

c_  = Fore.GREEN
sr_ = Style.RESET_ALL


def distribution_score(lambda_, x):
    miu = [0.6175, 0.6557, 0.4800]
    sigma = [0.1518, 0.1264, 0.1209]
    delta = 6

    if(miu[0]-(sigma[0]/delta) <= x and x <= miu[0]+(sigma[0]/delta)):
        LB_score = 1
    elif(x <= miu[0]-(sigma[0]/delta)):
        LB_score = 1-((miu[0]-(sigma[0]/delta)-x)**(lambda_))
    elif(miu[0]+(sigma[0]/delta) <= x ):
        LB_score = 1-((x-miu[0]+(sigma[0]/delta))**(lambda_))

    if (miu[1]-(sigma[1]/delta) <= x and x <= miu[1]+(sigma[1]/delta)):
        SB_score = 1
    elif (x <= miu[1]-(sigma[1]/delta)):
        SB_score = 1-((miu[1]-(sigma[1]/delta)-x)**(lambda_))
    elif (miu[1]+(sigma[1]/delta) <= x ):
        SB_score = 1-((x-miu[1]+(sigma[1]/delta))**(lambda_))


    if (miu[2]-(sigma[2]/delta) <= x and x <= miu[2]+(sigma[2]/delta)):
        ST_score = 1
    elif (x <= miu[2]-(sigma[2]/delta)):
        ST_score = 1-((miu[2]-(sigma[2]/delta)-x)**(lambda_))
    elif (miu[2]+(sigma[2]/delta) <= x ):
        ST_score = 1-((x-miu[2]+(sigma[2]/delta))**(lambda_))

    return torch.tensor([LB_score, SB_score, ST_score])



def train_one_epoch(timestamp_model, model, optimizer, scheduler, dataloader, device, epoch, lambda_, CFG):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train')
    for step, (images, masks, timestamp) in pbar:
        images = images.to(device)
        masks = masks.to(device)
        timestamp = timestamp.to(device)
        batch_size = images.size(0)


        with amp.autocast(enabled=True):

            y_pred = model(images)
            timestamp_pred = timestamp_model(images)

            if CFG['use_timestamp']:
                num_class = 3
                for i in range(len(y_pred)):
                    score = distribution_score(lambda_, timestamp_pred[i])
                    for j in range(num_class):
                        y_pred[i][j] = y_pred[i][j] * score[j]



            mask_loss = criterion(y_pred, masks, CFG)
            # loss_ = nn.L1Loss()
            # timestamp_loss = loss_(timestamp_pred.squeeze(1), timestamp)
            # timestamp_loss = loss_(timestamp_pred.squeeze(1), timestamp)
            # loss = mask_loss + timestamp_loss
            loss = mask_loss

            # print('*' * 25)
            # print(timestamp_pred)
            # print(timestamp_pred.squeeze(1))
            # print(timestamp)
            # # print(test_loss)
            # print(timestamp_loss)
            # print('*' * 25)
            loss = loss / CFG['n_accumulate']


        scaler.scale(loss).backward()

        if (step + 1) % CFG['n_accumulate'] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(timestamp_model, model, optimizer, dataloader, device, epoch, lambda_, CFG):
    model.eval()
    print(lambda_)

    dataset_size = 0
    running_loss = 0.0
    LB_val_scores = []
    SB_val_scores = []
    ST_val_scores = []
    total_val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks, timestamp) in pbar:

        images = images.to(device)
        masks = masks.to(device)
        timestamp = timestamp.to(device)


        batch_size = images.size(0)


        y_pred= model(images)
        timestamp_pred = timestamp_model(images)


        num_class = 3
        for i in range(len(y_pred)):
            score = distribution_score(lambda_, timestamp_pred[i])
            for j in range(num_class):
                y_pred[i][j] = y_pred[i][j] * score[j]
                # print('true')


        mask_loss = criterion(y_pred, masks, CFG)
        # loss_ = nn.L1Loss()
        # timestamp_loss = loss_(timestamp_pred.squeeze(1), timestamp)

        # loss = 0.8 * mask_loss + 0.2 * timestamp_loss
        loss = mask_loss


        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        LB_val_dice = dice_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        SB_val_dice = dice_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        ST_val_dice = dice_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        total_val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_jaccard = iou_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        SB_val_jaccard = iou_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        ST_val_jaccard = iou_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        total_val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_scores.append([LB_val_dice, LB_val_jaccard])
        SB_val_scores.append([SB_val_dice, SB_val_jaccard])
        ST_val_scores.append([ST_val_dice, ST_val_jaccard])
        total_val_scores.append([total_val_dice, total_val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    LB_val_scores = np.mean(LB_val_scores, axis=0)
    SB_val_scores = np.mean(SB_val_scores, axis=0)
    ST_val_scores = np.mean(ST_val_scores, axis=0)
    total_val_scores = np.mean(total_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, LB_val_scores, SB_val_scores, ST_val_scores, total_val_scores


@torch.no_grad()
def test_model(timestamp_model, model, optimizer, dataloader, device, lambda_, CFG):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    LB_val_scores = []
    SB_val_scores = []
    ST_val_scores = []
    total_val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Test')
    for step, (images, masks, timestamp) in pbar:

        images = images.to(device)
        masks = masks.to(device)


        batch_size = images.size(0)


        y_pred= model(images)
        timestamp_pred = timestamp_model(images)

        if CFG['use_timestamp']:
            num_class = 3
            for i in range(len(y_pred)):
                score = distribution_score(lambda_, timestamp_pred[i])
                for j in range(num_class):
                    y_pred[i][j] = y_pred[i][j] * score[j]


        mask_loss = criterion(y_pred, masks, CFG)

        loss = mask_loss


        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        LB_val_dice = dice_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        SB_val_dice = dice_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        ST_val_dice = dice_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        total_val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_jaccard = iou_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        SB_val_jaccard = iou_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        ST_val_jaccard = iou_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        total_val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_scores.append([LB_val_dice, LB_val_jaccard])
        SB_val_scores.append([SB_val_dice, SB_val_jaccard])
        ST_val_scores.append([ST_val_dice, ST_val_jaccard])
        total_val_scores.append([total_val_dice, total_val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    LB_val_scores = np.mean(LB_val_scores, axis=0)
    SB_val_scores = np.mean(SB_val_scores, axis=0)
    ST_val_scores = np.mean(ST_val_scores, axis=0)
    total_val_scores = np.mean(total_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, LB_val_scores, SB_val_scores, ST_val_scores, total_val_scores


def times_run_training(timestamp_model, model, optimizer, scheduler, num_epochs, CFG, device, train_loader, valid_loader, test_loader, run, fold):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)
    lambda_ = CFG['lambda_']
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}', end='')
        gc.collect()

        train_loss = train_one_epoch(timestamp_model, model, optimizer, scheduler,
                                     dataloader=train_loader, device=device,
                                     epoch=epoch, lambda_=lambda_, CFG=CFG)

        lambda_list = [lambda_-(0.4/(epoch**(1))), lambda_-(0.1/(epoch**(1))), lambda_, lambda_+(0.1/(epoch**(1))), lambda_+(0.4/(epoch**(1)))]
        temp_val_dice = []
        temp_val_jaccard = []
        temp_LB_val_dice = []
        temp_LB_val_jaccard = []
        temp_SB_val_dice = []
        temp_SB_val_jaccard = []
        temp_ST_val_dice = []
        temp_ST_val_jaccard = []
        for i in lambda_list:
            print('*'*25)
            print(f'Lambda: {i:0.4f}')
            
            temp_lambda = i

            val_loss, LB_val_scores, SB_val_scores, ST_val_scores, val_scores = valid_one_epoch(timestamp_model, model, optimizer, valid_loader, device,
                                                                   epoch=epoch, lambda_=temp_lambda, CFG=CFG)
            val_dice, val_jaccard = val_scores
            LB_val_dice, LB_val_jaccard = LB_val_scores
            SB_val_dice, SB_val_jaccard = SB_val_scores
            ST_val_dice, ST_val_jaccard = ST_val_scores
            temp_val_dice.append(val_dice)
            temp_val_jaccard.append(val_jaccard)
            temp_LB_val_dice.append(LB_val_dice)
            temp_LB_val_jaccard.append(LB_val_jaccard)
            temp_SB_val_dice.append(SB_val_dice)
            temp_SB_val_jaccard.append(SB_val_jaccard)
            temp_ST_val_dice.append(ST_val_dice)
            temp_ST_val_jaccard.append(ST_val_jaccard)
            print(val_dice)


        index = temp_val_dice.index(max(temp_val_dice))
        lambda_ = lambda_list[index]
        print('*'*25)
        print(index)
        print(temp_val_dice[0])
        print(lambda_list[index])
        print(lambda_)
        print(temp_val_dice[index])
        print('*' * 25)

        val_dice = temp_val_dice[index]
        val_jaccard = temp_val_jaccard[index]
        LB_val_dice = temp_LB_val_dice[index]
        LB_val_jaccard = temp_LB_val_jaccard[index]
        SB_val_dice = temp_SB_val_dice[index]
        SB_val_jaccard = temp_SB_val_jaccard[index]
        ST_val_dice = temp_ST_val_dice[index]
        ST_val_jaccard = temp_ST_val_jaccard[index]

        history['lambda_'].append(lambda_)
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['LB Valid Dice'].append(LB_val_dice)
        history['LB Valid Jaccard'].append(LB_val_jaccard)
        history['SB Valid Dice'].append(SB_val_dice)
        history['SB Valid Jaccard'].append(SB_val_jaccard)
        history['ST Valid Dice'].append(ST_val_dice)
        history['ST Valid Jaccard'].append(ST_val_jaccard)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        # Log the metrics
        wandb.log({"lambda_": lambda_,
                   "Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   'LB Valid Dice': LB_val_dice,
                   'LB Valid Jaccard': LB_val_jaccard,
                   'SB Valid Dice': SB_val_dice,
                   'SB Valid Jaccard': SB_val_jaccard,
                   'ST Valid Dice': ST_val_dice,
                   'ST Valid Jaccard': ST_val_jaccard,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": scheduler.get_last_lr()[0]})

        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            PATH = f"model_save/unet-{CFG['backbone']}-{CFG['model_name']}-{CFG['2.5D']}-best_epoch-{epoch:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{sr_}")
            best_lambda = lambda_

    model.load_state_dict(torch.load(PATH))



    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    val_loss, LB_val_scores, SB_val_scores, ST_val_scores, val_scores = test_model(timestamp_model, model, optimizer, test_loader, device, best_lambda, CFG = CFG)
    val_dice, val_jaccard = val_scores
    LB_val_dice, LB_val_jaccard = LB_val_scores
    SB_val_dice, SB_val_jaccard = SB_val_scores
    ST_val_dice, ST_val_jaccard = ST_val_scores

    history['LB Valid Dice'].append(LB_val_dice)
    history['LB Valid Jaccard'].append(LB_val_jaccard)
    history['SB Valid Dice'].append(SB_val_dice)
    history['SB Valid Jaccard'].append(SB_val_jaccard)
    history['ST Valid Dice'].append(ST_val_dice)
    history['ST Valid Jaccard'].append(ST_val_jaccard)
    history['Valid Dice'].append(val_dice)
    history['Valid Jaccard'].append(val_jaccard)

    wandb.log({"lambda_": best_lambda,
               "Train Loss": train_loss,
               "Valid Loss": val_loss,
               'LB Valid Dice': LB_val_dice,
               'LB Valid Jaccard': LB_val_jaccard,
               'SB Valid Dice': SB_val_dice,
               'SB Valid Jaccard': SB_val_jaccard,
               'ST Valid Dice': ST_val_dice,
               'ST Valid Jaccard': ST_val_jaccard,
               "Valid Dice": val_dice,
               "Valid Jaccard": val_jaccard,
               "LR": scheduler.get_last_lr()[0]})

    return model, history


