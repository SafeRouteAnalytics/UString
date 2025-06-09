#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse
import shutil

from torch.utils.data import DataLoader, Subset
from src.Models import UString
from src.eval_tools import evaluation, print_results, vis_results
import ipdb
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from src.DataLoader import DADDatasetCV, DADDataset
from sklearn.model_selection import KFold

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
ROOT_PATH = os.path.dirname(__file__)

feature_dim = 4096


metrics_arr = []

def average_losses(losses_all):
    total_loss, cross_entropy, log_posterior, log_prior, aux_loss, rank_loss = 0, 0, 0, 0, 0, 0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        log_posterior += losses['log_posterior']
        log_prior += losses['log_prior']
        aux_loss += losses['auxloss']
        rank_loss += losses['ranking']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['log_posterior'] = log_posterior / len(losses_all)
    losses_mean['log_prior'] = log_prior / len(losses_all)
    losses_mean['auxloss'] = aux_loss / len(losses_all)
    losses_mean['ranking'] = rank_loss / len(losses_all)
    return losses_mean


def test_all(testdata_loader, model):
    
    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas) in enumerate(testdata_loader):
            # run forward inference
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas, graph_edges, 
                    hidden_in=None, edge_weights=edge_weights, npass=10, nbatch=len(testdata_loader), testing=False)
            # make total loss
            losses['total_loss'] = p.loss_alpha * (losses['log_posterior'] - losses['log_prior']) + losses['cross_entropy']
            losses['total_loss'] += p.loss_beta * losses['auxloss']
            losses['total_loss'] += p.loss_yita * losses['ranking']
            losses_all.append(losses)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            # run inference
            for t in range(num_frames):
                pred = all_outputs[t]['pred_mean']
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int)
            all_toas.append(toas)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
    
    return all_pred, all_labels, all_toas, losses_all


def test_all_vis(testdata_loader, model, vis=True, multiGPU=False, device=torch.device('cuda')):
    
    if multiGPU:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.eval()

    all_pred = []
    all_labels = []
    all_toas = []
    vis_data = []
    all_uncertains = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas, detections, video_ids) in tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            # run forward inference
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas, graph_edges, 
                    hidden_in=None, edge_weights=edge_weights, npass=10, nbatch=len(testdata_loader), testing=False, eval_uncertain=True)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            pred_uncertains = np.zeros((batch_size, num_frames, 2), dtype=np.float32)
            # run inference
            for t in range(num_frames):
                # prediction
                pred = all_outputs[t]['pred_mean']  # B x 2
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
                # uncertainties
                aleatoric = all_outputs[t]['aleatoric']  # B x 2 x 2
                aleatoric = aleatoric.cpu().numpy() if aleatoric.is_cuda else aleatoric.detach().numpy()
                epistemic = all_outputs[t]['epistemic']  # B x 2 x 2
                epistemic = epistemic.cpu().numpy() if epistemic.is_cuda else epistemic.detach().numpy()
                pred_uncertains[:, t, 0] = aleatoric[:, 0, 0] + aleatoric[:, 1, 1]
                pred_uncertains[:, t, 1] = epistemic[:, 0, 0] + epistemic[:, 1, 1]

            # gather results and ground truth
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int)
            all_toas.append(toas)
            all_uncertains.append(pred_uncertains)

            if vis:
                # gather data for visualization
                vis_data.append({'pred_frames': pred_frames, 'label': label, 'pred_uncertain': pred_uncertains,
                                'toa': toas, 'detections': detections, 'video_ids': video_ids})

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
    all_uncertains = np.vstack((np.vstack(all_uncertains[:-1]), all_uncertains[-1]))

    return all_pred, all_labels, all_toas, all_uncertains, vis_data


def write_scalars(logger, cur_epoch, cur_iter, losses, lr):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    log_prior = losses['log_prior'].mean().item()
    log_posterior = losses['log_posterior'].mean().item()
    aux_loss = losses['auxloss'].mean().item()
    rank_loss = losses['ranking'].mean().item()
    # print info
    print('----------------------------------')
    print('epoch: %d, iter: %d' % (cur_epoch, cur_iter))
    print('total loss = %.6f' % (total_loss))
    print('cross_entropy = %.6f' % (cross_entropy))
    print('log_posterior = %.6f' % (log_posterior))
    print('log_prior = %.6f' % (log_prior))
    print('aux_loss = %.6f' % (aux_loss))
    print('rank_loss = %.6f' % (rank_loss))
    # write to tensorboard
    logger.add_scalars("train/losses/total_loss", {'total_loss': total_loss}, cur_iter)
    logger.add_scalars("train/losses/cross_entropy", {'cross_entropy': cross_entropy}, cur_iter)
    logger.add_scalars("train/losses/log_posterior", {'log_posterior': log_posterior}, cur_iter)
    logger.add_scalars("train/losses/log_prior", {'log_prior': log_prior}, cur_iter)
    logger.add_scalars("train/losses/complexity_cost", {'complexity_cost': log_posterior-log_prior}, cur_iter)
    logger.add_scalars("train/losses/aux_loss", {'aux_loss': aux_loss}, cur_iter)
    logger.add_scalars("train/losses/rank_loss", {'rank_loss': rank_loss}, cur_iter)
    # write learning rate
    logger.add_scalars("train/learning_rate/lr", {'lr': lr}, cur_iter)


def write_test_scalars(logger, cur_epoch, cur_iter, losses, metrics):
    # fetch results
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    # write to tensorboard
    loss_info = {'total_loss': total_loss, 'cross_entropy': cross_entropy}
    aux_loss = losses['auxloss'].mean().item()
    loss_info.update({'aux_loss': aux_loss})
    logger.add_scalars("test/losses/total_loss", loss_info, cur_iter)
    logger.add_scalars("test/accuracy/AP", {'AP': metrics['AP']}, cur_iter)
    logger.add_scalars("test/accuracy/time-to-accident", {'mTTA': metrics['mTTA'], 
                                                          'TTA_R80': metrics['TTA_R80']}, cur_iter)


def write_weight_histograms(writer, net, epoch):
    writer.add_histogram('histogram/w1_mu', net.predictor.l1.weight_mu, epoch)
    writer.add_histogram('histogram/w1_rho', net.predictor.l1.weight_rho, epoch)
    writer.add_histogram('histogram/w2_mu', net.predictor.l2.weight_mu, epoch)
    writer.add_histogram('histogram/w2_rho', net.predictor.l2.weight_rho, epoch)
    writer.add_histogram('histogram/b1_mu', net.predictor.l1.bias_mu, epoch)
    writer.add_histogram('histogram/b1_rho', net.predictor.l1.bias_rho, epoch)
    writer.add_histogram('histogram/b2_mu', net.predictor.l2.bias_mu, epoch)
    writer.add_histogram('histogram/b2_rho', net.predictor.l2.bias_rho, epoch)


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    next_epoch_to_run = 0
    resumed_iter_cur = 0
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage) # Ensure loading on CPU if GPU not available, then move
        
        saved_epoch = checkpoint['epoch']  # Epoch that was completed
        next_epoch_to_run = saved_epoch + 1
        
        # Adjust model loading for DataParallel
        state_dict = checkpoint['model']
        if isinstance(model, torch.nn.DataParallel):
            # if checkpoint was saved from DataParallel, keys start with 'module.'
            # if checkpoint was saved from single GPU, keys don't.
            # We need to ensure consistency.
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            is_dataparallel_checkpoint = any(key.startswith('module.') for key in state_dict.keys())
            if is_dataparallel_checkpoint: # if saved from DP
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k # remove `module.`
                    new_state_dict[name] = v
                model.module.load_state_dict(new_state_dict)
            else: # if saved from single GPU
                model.module.load_state_dict(state_dict) # Load into model.module
        else: # model is not DataParallel
            is_dataparallel_checkpoint = any(key.startswith('module.') for key in state_dict.keys())
            if is_dataparallel_checkpoint: # if saved from DP, strip 'module.'
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k 
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)


        if isTraining and optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        resumed_iter_cur = checkpoint.get('iter_cur', 0)
        print(f"=> loaded checkpoint '{filename}' (completed epoch {saved_epoch}, iter_cur {resumed_iter_cur}, next epoch to run {next_epoch_to_run})")
    else:
        print(f"=> no checkpoint found at '{filename}'")

    return model, optimizer, next_epoch_to_run, resumed_iter_cur


def train_eval(traindata_loader, testdata_loader, current_fold_num):
    global p, device, gpu_ids # Ensure p, device, gpu_ids are accessible (assuming they are global or passed)

    data_path = p.data_path
    
    fold_snapshot_dirname = f"fold_{current_fold_num}_snapshot"
    fold_logs_dirname = f"fold_{current_fold_num}_logs"

    model_dir = os.path.join(p.output_dir, p.dataset, fold_snapshot_dirname)
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    logs_dir = os.path.join(p.output_dir, p.dataset, fold_logs_dirname)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)

    n_obj_val = 19 
    if hasattr(traindata_loader.dataset.dataset, 'n_obj'):
        n_obj_val = traindata_loader.dataset.dataset.n_obj
    else:
        print(f"Warning: 'n_obj' attribute not found in dataset. Using default value: {n_obj_val}")

    model = UString(feature_dim, p.hidden_dim, p.latent_dim, 
                       n_layers=p.num_rnn, n_obj=n_obj_val, n_frames=p.n_frames, fps=p.fps, 
                       with_saa=True, uncertain_ranking=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    
    next_epoch_to_run = 0
    iter_cur = 0 
    best_metric_for_fold = 0.0 # Renamed from best_metric to be specific to fold

    # Initialize current_eval_metrics with default "worst" values
    current_eval_metrics = {'AP': -1.0, 'mTTA': float('inf'), 'TTA_R80': float('inf')}

    resume_model_file_path = p.model_file # Default global model file
    # For true fold-specific resume, you might want to look for 'best_model_for_fold.pth' or 'latest_checkpoint.pth' in model_dir
    # e.g., resume_model_file_path = os.path.join(model_dir, 'best_model_for_fold.pth')
    # For simplicity, using p.model_file, assuming user sets it appropriately if resuming a specific fold.

    if p.resume:
        if os.path.isfile(resume_model_file_path):
            model, optimizer, next_epoch_to_run, iter_cur = load_checkpoint(
                model, optimizer=optimizer, filename=resume_model_file_path, isTraining=True
            )
            # Potentially load best_metric_for_fold from checkpoint if saved
            # checkpoint = torch.load(resume_model_file_path)
            # best_metric_for_fold = checkpoint.get('best_metric_for_fold', 0.0)
        else:
            print(f"Resume specified, but checkpoint '{resume_model_file_path}' not found for fold {current_fold_num}. Starting fresh.")
    
    model_for_histograms = model.module if isinstance(model, torch.nn.DataParallel) else model
    initial_hist_epoch = (next_epoch_to_run -1) if (p.resume and os.path.isfile(resume_model_file_path) and next_epoch_to_run > 0) else 0
    write_weight_histograms(logger, model_for_histograms, initial_hist_epoch)

    for k in range(next_epoch_to_run, p.epoch):
        model.train()
        
        epoch_ran_at_least_one_batch = False
        last_batch_losses = None # Initialize for each epoch

        if len(traindata_loader) == 0:
            print(f"Warning: Fold {current_fold_num} - Epoch {k}: train_data_loader is empty. Skipping training iterations for this epoch.")
        else:
            for i, (batch_xs, batch_ys, graph_edges, edge_weights, batch_toas) in enumerate(traindata_loader):
                epoch_ran_at_least_one_batch = True
                optimizer.zero_grad()
                
                # Renamed 'losses' to 'current_batch_losses' to avoid confusion with the one for scheduler
                current_batch_losses, all_outputs, hidden_st = model(batch_xs, batch_ys, batch_toas, graph_edges, edge_weights=edge_weights, npass=2, nbatch=len(traindata_loader), eval_uncertain=True)
                
                complexity_loss = current_batch_losses['log_posterior'] - current_batch_losses['log_prior']
                current_batch_losses['total_loss'] = p.loss_alpha * complexity_loss + current_batch_losses['cross_entropy']
                current_batch_losses['total_loss'] += p.loss_beta * current_batch_losses['auxloss']
                current_batch_losses['total_loss'] += p.loss_yita * current_batch_losses['ranking']
                
                current_batch_losses['total_loss'].mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                
                last_batch_losses = current_batch_losses # Store for scheduler
                
                lr = optimizer.param_groups[0]['lr']
                write_scalars(logger, k, iter_cur, current_batch_losses, lr)
                
                iter_cur += 1
                if iter_cur % p.test_iter == 0:
                    model.eval()
                    # Note: test_all uses p.batch_size, ensure testdata_loader is created with this.
                    all_pred, all_labels, all_toas, losses_all_eval = test_all(testdata_loader, model)
                    model.train()
                    
                    loss_val = average_losses(losses_all_eval) # Ensure average_losses handles empty losses_all_eval
                    print('----------------------------------')
                    print(f"Fold {current_fold_num} - Evaluation at Epoch {k}, Iteration {iter_cur}")
                    
                    # current_eval_metrics is updated here
                    eval_fps = p.fps # Assuming test_data.fps is p.fps
                    if hasattr(testdata_loader.dataset.dataset, 'fps'): # If Subset, .dataset is Subset, .dataset.dataset is original
                        eval_fps = testdata_loader.dataset.dataset.fps

                    current_eval_metrics['AP'], current_eval_metrics['mTTA'], current_eval_metrics['TTA_R80'] = evaluation(all_pred, all_labels, all_toas, fps=eval_fps)
                    print('----------------------------------')
                    write_test_scalars(logger, k, iter_cur, loss_val, current_eval_metrics)

        # End of epoch k
        model_file_epoch = os.path.join(model_dir, f'gcrnn_model_fold{current_fold_num}_epoch_{k:02d}.pth')
        torch.save({
            'epoch': k, # Epoch k has been completed
            'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_cur': iter_cur,
            'best_metric_for_fold': best_metric_for_fold 
        }, model_file_epoch)
        
        # Use current_eval_metrics['AP'] which holds the latest eval AP (or -1.0 if no eval this epoch)
        if current_eval_metrics['AP'] > best_metric_for_fold:
            best_metric_for_fold = current_eval_metrics['AP']
            best_model_path = os.path.join(model_dir, f'best_model_fold{current_fold_num}.pth')
            update_final_model(model_file_epoch, best_model_path)
            print(f'Fold {current_fold_num} - Epoch {k}: New best AP: {best_metric_for_fold:.4f}. Best model updated: {best_model_path}')
        else:
            print(f'Fold {current_fold_num} - Epoch {k}: Model saved: {model_file_epoch}. Current AP: {current_eval_metrics["AP"]:.4f} (Best for fold: {best_metric_for_fold:.4f})')

        # Step the scheduler using losses from the last processed batch of the epoch
        if epoch_ran_at_least_one_batch and last_batch_losses is not None:
            # Ensure the metric for scheduler is a scalar
            scheduler_metric = last_batch_losses['log_posterior'].mean().item()
            scheduler.step(scheduler_metric)
            print(f"Fold {current_fold_num} - Epoch {k}: Scheduler stepped with metric {scheduler_metric:.6f}")
        elif not epoch_ran_at_least_one_batch and len(traindata_loader) > 0:
             # This case should ideally not happen if len(traindata_loader) > 0
            print(f"Fold {current_fold_num} - Epoch {k}: Scheduler not stepped. No batches processed despite non-empty loader.")
        elif len(traindata_loader) == 0:
            print(f"Fold {current_fold_num} - Epoch {k}: Scheduler not stepped (train_loader was empty).")


        model_for_histograms = model.module if isinstance(model, torch.nn.DataParallel) else model
        write_weight_histograms(logger, model_for_histograms, k + 1) # Log for epoch k completed

    metrics_arr.append(best_metric_for_fold) # metrics_arr is global
    logger.close()

# --- Main script part ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/dad',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='dad',
                        help='The name of the dataset.')
    parser.add_argument('--n_frames', type=int, default=100,
                        help='no of frames in a video')
    parser.add_argument('--n_folds', type=int, default=3,
                        help='no of folds in cv')
    parser.add_argument('--fps', type=int, default=20,
                        help='no of fps in a video')
    parser.add_argument('--toa', type=float, default=None,
                        help='common toa of all videos')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--epoch', type=int, default=15,
                        help='The number of training epoches. Default: 30')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--num_rnn', type=int, default=1,
                        help='The number of RNN cells for each timestamp. Default: 1')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='The name of feature embedding methods. Default: vgg16')
    parser.add_argument('--test_iter', type=int, default=64,
                        help='The number of iteration to perform a evaluation process. Default: 64')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The dimension of hidden states in RNN. Default: 256')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='The dimension of latent space. Default: 256')
    parser.add_argument('--loss_alpha', type=float, default=0.001,
                        help='The weighting factor of posterior and prior losses. Default: 1e-3')
    parser.add_argument('--loss_beta', type=float, default=10,
                        help='The weighting factor of auxiliary loss. Default: 10')
    parser.add_argument('--loss_yita', type=float, default=10,
                        help='The weighting factor of uncertainty ranking loss. Default: 10')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--phase', type=str, choices=['train', 'test'],
                        help='The state of running the model. Default: train')
    parser.add_argument('--evaluate_all', action='store_true',
                        help='Whether to evaluate models of all epoches. Default: False')
    parser.add_argument('--visualize', action='store_true',
                        help='The visualization flag. Default: False')
    parser.add_argument('--resume', action='store_true',
                        help='If to resume the training. Default: False')
    parser.add_argument('--model_file', type=str, default='./output_debug/bayes_gcrnn/vgg16/dad/snapshot/gcrnn_model_90.pth',
                        help='The trained GCRNN model file for demo test only.')
    parser.add_argument('--output_dir', type=str, default='./output_debug/bayes_gcrnn/vgg16',
                        help='The directory of src need to save in the training.')
    # ... (your argparse definitions) ...
    p = parser.parse_args() # p becomes global here
    
    # gpu options (device and gpu_ids become global)
    gpu_ids = [int(id) for id in p.gpus.split(',')]
    print("Using GPU devices: ", gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Seed setting should ideally be at the very top if not already
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


    full_dataset = DADDatasetCV(p.data_path, 'training', toTensor=True, device=device, n_frames=p.n_frames, fps=p.fps, toa=p.toa)

    folds = p.n_folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    metrics_arr.clear() # Ensure it's clean if script is run multiple times in a session

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(full_dataset)))): # kf.split needs an array-like of indices or samples
        current_fold_num = fold_idx + 1
        print(f'\n========== Running Fold {current_fold_num}/{folds} ==========')

        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)

        traindata_loader = DataLoader(dataset=train_subset, batch_size=p.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True if device.type == 'cuda' else False)
        testdata_loader = DataLoader(dataset=test_subset, batch_size=p.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True if device.type == 'cuda' else False)
        # Note: test_all in your original code uses p.batch_size implicitly. The DataLoader above uses p.batch_size.
        # The test_all function doesn't take test_data directly but testdata_loader.
        # The fps for evaluation in test_all comes from test_data.fps.
        # test_data is created inside test_eval, not directly available in train_eval.
        # For evaluation inside train_eval, you might need to access fps from testdata_loader.dataset.dataset.fps if it's a Subset.

        train_eval(traindata_loader, testdata_loader, current_fold_num)

    print(f"\n========== Cross-Validation Summary ==========")
    if metrics_arr:
        print(f"All best APs per fold: {', '.join(map(lambda x: f'{x:.4f}', metrics_arr))}")
        mean_ap = sum(metrics_arr) / len(metrics_arr) if metrics_arr else 0
        std_ap = np.std(metrics_arr) if len(metrics_arr) > 1 else 0
        print(f"Average AP over {folds} folds: {mean_ap:.4f} +/- {std_ap:.4f}")
    else:
        print("No metrics recorded.")
