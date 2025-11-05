import torch
import tqdm
import argparse
import os

import time

import os
import sys
import gc
import glob
import argparse
import json
import logging
import os, sys
import numpy as np
import torch

from dataset_loader import EEGDatasetSimdata
from clip_model import *
from vlaai import VLAAI
from train_clip_helper_functions import *

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from torch.utils.data import DataLoader

def printf(s, file):
    print(s)
    with open(file, 'a') as file:
        file.write(s + '\n')


def load_eeg_encoder(eeg_encoder, units_lstm, padding, spatial_filters, number_conv_layers,window_length, latent_dim,attention_depth ):
    if eeg_encoder == 'lstm':
        eeg = EEGModel(spatial_filters_eeg=32, filters_cnn_eeg=16, kerSize_temporal=9, stride_temporal=3,
                 units_hidden=128, units_lstm=units_lstm, fun_act=nn.LeakyReLU(), padding=padding)

    elif eeg_encoder== 'double_lstm':
        eeg = EEGLstm(speech_dim=64, units_lstm=units_lstm, spatial_filters=spatial_filters)

    elif eeg_encoder == 'vlaai':
        eeg = VLAAI()
        stride_temporal = 1
    elif eeg_encoder == 'convLSTM':
        eeg = EEGConvLSTM(units_lstm=128,
                          output_dim=latent_dim,
                          dropout_rate=0.4, eeg_dim=64,
                          filters=(64,) * number_conv_layers,
                          kernels=(32,) * number_conv_layers,
                          dilation_rate=1,
                          input_channels=64,
                          time_dimension=window_length,
                          normalization_fn='layer_norm',
                          activation_fn='leaky_relu')
        output_dim = latent_dim

    elif eeg_encoder == 'convLSTMnew':
        eeg = EEGConvLSTMNew(
                          output_dim=latent_dim,
                          dropout_rate=0.4, eeg_dim=64,
                          filters=(64,) * number_conv_layers,
                          kernels=(64,) * number_conv_layers,
                          dilation_rate=1,
                          input_channels=64,
                          time_dimension=window_length,
                          normalization_fn='layer_norm',
                          activation_fn='leaky_relu')
        output_dim = latent_dim

    elif eeg_encoder == 'conformer':
        eeg = EEGConformer(
                 output_dim = latent_dim,
                 conformer_input_dim=64,
                 dropout_rate=0.2, eeg_dim=64,
                 filters=(64,) * number_conv_layers,
                 kernels=(64,) * number_conv_layers,
                 dilation_rate=1,
                 input_channels=64,
                 time_dimension=window_length,
                 depth=attention_depth)
        stride_temporal = 1

    elif eeg_encoder == 'EEGConformerInterleaved':
        eeg = EEGConformerInterleaved(
                 output_dim = latent_dim,
                 conformer_input_dim=64,
                 dropout_rate=0.2, eeg_dim=64,
                 filters=(64,) * number_conv_layers,
                 kernels=(64,) * number_conv_layers,
                 dilation_rate=1,
                 input_channels=64,
                 time_dimension=window_length,
                 depth=attention_depth)
        stride_temporal = 1

    return eeg

def load_speech_encoder(speech_encoder, units_lstm, padding, spatial_filters, number_conv_layers, window_length, stride_temporal, speech_dimension):
    if speech_encoder == 'lstm':
        speech = MelModel(spatial_filters=spatial_filters, filters_cnn=16, kerSize_temporal=9, stride_temporal=stride_temporal,
                          units_lstm=units_lstm, padding=padding, dropout_rate=0, activation=nn.LeakyReLU(), speech_dim= speech_dimension)

    elif speech_encoder == 'double_lstm':
        speech = EEGLstm(speech_dim=speech_dimension, units_lstm=units_lstm, spatial_filters=spatial_filters)

    elif speech_encoder == 'Wav2vecSmallModel':
        speech = Wav2vecSmallModel(speech_dim=speech_dimension,spatial_filters=units_lstm, stride_temporal=stride_temporal)

    elif speech_encoder == 'smallConv':
        speech = SpeechSmallConv(output_dim=latent_dim, ks_temporal=16,
                                 dropout_rate=0.4, speech_dim=speech_dimension, time_dimension=window_length)

    elif speech_encoder == 'convLSTM':
        speech = EEGConvLSTM(units_lstm=128,
                          output_dim=latent_dim,
                          dropout_rate=0.4, eeg_dim=speech_dimension,
                          filters=(64,) * number_conv_layers,
                          kernels=(32,) * number_conv_layers,
                          dilation_rate=1,
                          input_channels=speech_dimension,
                          time_dimension=window_length,
                          normalization_fn='layer_norm',
                          activation_fn='leaky_relu')
        output_dim = latent_dim

    return speech

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f'using device {device}')
    print(f'number of gpus {torch.cuda.device_count()}')
    print(f'found gpu {torch.cuda.is_available()}')

    torch.backends.cudnn.benchmark = True

    # Parameters
    params = {'batch_size': None,
              'batch_sampler' : None,
              'shuffle': False,
              # 'num_workers': 2,
              'pin_memory':True
            }
    window_length_s = 3
    fs = 64
    window_length = window_length_s * fs  # 5 seconds
    # Hop length between two consecutive decision windows
    hop_length = window_length
    epochs = 500
    patience = 20

    parser = argparse.ArgumentParser(description="Train CLIP model.")


    # add argument input
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=str, default='no', choices=['yes', 'no']) # if yes, only loads in a few files for the training, to run through a few epochs before launching for real
    parser.add_argument('--only_evaluate', type=str, default='no', choices=['yes', 'no']) # If yes, the code will disregard all the parameters below, and load the saved arguments file from the results_folder directory
    parser.add_argument('--results_folder', type=str,
                        default=os.path.join(os.path.dirname(__file__), "results")) # if only_evaluate ==yes, then the model will be loaded from the results folder, so give the correct results folder to the arguments then

    parser.add_argument('--run', type=int, default=4),  # run = 0 # between 0 and 9

    parser.add_argument('--lstm_units', type=int, default=128)
    parser.add_argument('--lambda_sim_loss', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--momentum_membank', type=float, default=0.90)

    parser.add_argument('--eeg_norm', type=str, default='mvn', choices=['mvn'])
    parser.add_argument('--stimulus_features', type=str, default='envelope')  # , choices=['mel', 'envelope', 'wav2vec_19'])
    parser.add_argument('--model_arch', type=str, default='clip_sim_no_latent_proj',choices=['no_contrastive_learning','clip_kld', 'clip_kld_latent_proj','clip_mp','clip_sim','clip_sim_no_latent_proj', 'clip_extended', 'clip_no_eeg_loss', 'clip_correct'])
    parser.add_argument('--speech_encoder', type=str, default='convLSTM', choices=['conformer', 'smallConv','lstm','convLSTM', 'no', 'double_lstm', 'Wav2vecSmallModel'])
    parser.add_argument('--eeg_encoder', type=str, default='EEGConformerInterleaved', choices=['EEGConformerInterleaved','conformer','convLSTMnew','convLSTM','lstm_newvals','vlaai','clipmeta','lstm', 'lstm_lstm', 'double_lstm', 'transformerEncoder'])
    parser.add_argument('--attention_depth', type=int, default=10) # depth for the eeg_encoderConformer
    parser.add_argument('--load_pretrain', type=str, default='no', choices=['yes', 'no'])

    # arguments for shuffling, or data augmentation
    parser.add_argument('--shuffle', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--shuffle_percentage', type=float, default=1.0) # how much of the batch is shuffled each time ( to have segments of multiple subjects per batch)
    parser.add_argument('--addEEG', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--data_augmentation', type=str, default='no', choices=['no', 'SignFlip', 'FTSurrogate', 'FrequencyShift', 'BandstopFilter', 'GaussianNoise', 'SmoothTimeMask', 'ChannelsDropout', 'ChannelsShuffle'])
    parser.add_argument('--data_augmentation_percentage', type=float, default=0.5) #probability of data augmentation

    # learning rate parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.90)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--use_amsgrad', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['no', 'plateau','step', 'cosine', 'cosine_warmup'])
    parser.add_argument('--step_size_scheduler', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--number_conv_layers', type=int, default=1)

    parser.add_argument('--fun_act', type=str, default='relu')
    parser.add_argument('--temperature', type=float, default=0.075)
    parser.add_argument('--subject_split', type=str, default='icassp_testset', choices=['within', 'heldout','icassp_testset'])

    parser.add_argument('--data_dir', type=str, default='/esat/audioslave/lbollens/sparrkulee_data/sparrkulee' )
    parser.add_argument('--number_of_training_subjects', type=int, default=1000)
    parser.add_argument('--lambda_clip_loss', type=float, default=1)

    parser.add_argument('--latent_dim', type=int, default=8)

    args = parser.parse_args()
    debug = args.debug.lower() == 'yes'

    # ensure all the parameters are set
    subject_split = args.subject_split
    lambda_clip_loss = args.lambda_clip_loss
    lambda_sim_loss = args.lambda_sim_loss
    eeg_norm = args.eeg_norm
    model_arch = args.model_arch
    speech_encoder = args.speech_encoder
    eeg_encoder = args.eeg_encoder
    load_pretrain = args.load_pretrain
    only_evaluate = args.only_evaluate
    number_conv_layers = args.number_conv_layers
    epochs = args.epochs
    attention_depth = args.attention_depth
    patience = args.patience
    warmup_epochs = args.warmup_epochs
    optimizer = args.optimizer

    batch_size = args.batch_size
    bs = batch_size
    momentum_membank = args.momentum_membank
    fun_act = args.fun_act
    temperature = args.temperature
    learning_rate = args.learning_rate

    beta1 = args.beta1
    beta2 = args.beta2
    weight_decay = args.weight_decay
    lr_scheduler = args.lr_scheduler
    step_size_scheduler = args.step_size_scheduler
    use_amsgrad = (args.use_amsgrad == 'yes')

    number_of_training_subjects = args.number_of_training_subjects

    shuffle = args.shuffle.lower() == 'yes'
    shuffle_percentage = args.shuffle_percentage
    addEEG = args.addEEG.lower() == 'yes'

    data_augmentation = args.data_augmentation
    if data_augmentation == 'no':
        data_augmentation = []
    else:
        data_augmentation = [data_augmentation]
    data_augmentation_percentage = args.data_augmentation_percentage
    stimulus_features = args.stimulus_features
    # get latent dimension of speech/eeg models
    latent_dim = args.latent_dim
    # save
    run = args.run
    data_folder = args.data_dir
    units_lstm = args.lstm_units
    results_folder = args.results_folder


    if args.only_evaluate == 'yes':
        only_evaluate = True
        # load and overwrite all the parameter from the saved results model folder
        with open(os.path.join(results_folder, 'args.txt'), 'r') as f:
            # read
            args_saved = json.load(f)
            # overwrite
            for key, value in args_saved.items():
                if key != 'only_evaluate' and key != 'results_folder' and key !='debug':
                    # set the python varible with name key to value, it should also work if value is a string
                    exec(f"{key} = value", globals(), globals())


    else:
        only_evaluate = False


    # set the spatial_filters, at the beginning of the speech encoder
    if stimulus_features == 'mel':
        speech_dimension = 28
        spatial_filters = 64
    elif stimulus_features == 'envelope':
        speech_dimension = 1
        spatial_filters = 8
    elif 'wav2vec' in stimulus_features :
        speech_dimension = 1024
        spatial_filters = 128


    experiments_folder = os.path.join(os.path.dirname(__file__), "results")

    os.makedirs(experiments_folder, exist_ok=True)

    print(f'data folder {data_folder}')

    # Create a directory to store (intermediate) results
    os.makedirs(experiments_folder, exist_ok=True)
    if not only_evaluate:
        results_folder = os.path.join(experiments_folder,
                                      f"results_{model_arch}_eeg_{eeg_encoder}_speech_{speech_encoder}_date_{time.strftime('%m-%d-%H-%M-%S')}")
        os.makedirs(results_folder, exist_ok=True)
        # save the arguments
        with open(os.path.join(results_folder, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    checkpoint_path = os.path.join(results_folder, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    file_loss = os.path.join(results_folder, 'loss.txt')

    dataset_split_stories = os.path.join(os.path.dirname(__file__), "fold_split.json")


    train_files, val_files, test_files, test_files_heldout, train_audio, val_audio, test_audio, test_audio_heldout = get_train_val_test_files_final(data_folder, run, stimulus_features,
                                                                  dataset_split_stories,  number_of_training_subjects, debug=debug)


    padding = 'valid'
    stride_temporal =3
    if speech_encoder == 'Wav2vecSmallModel':
        padding = 'valid'
    else:
        padding = 'valid'

    # load EEG encoder model
    eeg = load_eeg_encoder(eeg_encoder, units_lstm, padding, spatial_filters, number_conv_layers,window_length, latent_dim,attention_depth)

    # load speech encoder model
    speech = load_speech_encoder(speech_encoder, units_lstm, padding, spatial_filters, number_conv_layers, window_length, stride_temporal, speech_dimension)


    if not only_evaluate:
        print(train_files)
        print(val_files)
        print(test_files)
        print(f'number of training files {len(train_files)}')
        print(f'number of validation files {len(val_files)}')
        print(f'number of test files {len(test_files)}')


        train_data = EEGDatasetSimdata(train_files, train_audio, window_length, hop_length, batch_size=bs, shuffle=shuffle,
                                       addEEG = addEEG,
                                        shuffle_percentage=shuffle_percentage, data_augmentation=data_augmentation, data_augmentation_probability=data_augmentation_percentage,
                                       )
        train_loader= DataLoader(train_data, **params)


        val_data = EEGDatasetSimdata(val_files, val_audio, window_length, hop_length,  batch_size=bs,
                                     shuffle_percentage=shuffle_percentage,
                                     )
        val_loader = DataLoader(val_data, **params)


        # load memory bank
        if model_arch == 'clip_sim_no_latent_proj' or  model_arch == 'clip_sim_no_latent_proj' or  model_arch == 'clip_kld' :
            latent_dim  = speech.get_output_dim(window_length) # get the output dimension of the eeg model, since this is the dimension of the vector in the memory bank


        memoryBank = memoryBank(bank_size=train_data.get_number_of_stimuli_segments(), dim=latent_dim, momentum=momentum_membank,
                                device=device)

    else:
        memoryBank = None


    # Load the experiment configuration
    if model_arch == 'clip_sim':
        model = CLIPSim(eeg, speech, memoryBank, temperature=temperature, latent_dim=latent_dim, window_length=window_length, lambda_clip=lambda_clip_loss, lambda_average=lambda_sim_loss)

    elif model_arch == 'clip_sim_no_latent_proj':
        model = CLIPSimNoLatentProj(eeg, speech, memoryBank, temperature=temperature, window_length=window_length, lambda_clip=lambda_clip_loss, lambda_average=lambda_sim_loss)

    elif model_arch == 'clip_mp':
        model = CLIPSimMultiplePositives(eeg, speech, temperature=temperature, window_length=window_length,
                                    lambda_clip=lambda_clip_loss, lambda_average=lambda_sim_loss)

    elif model_arch == 'clip_kld':
        model = CLIPKLDNoLatentProj(eeg, speech, latent_dimension=latent_dim, number_of_classes=train_data.get_number_of_stimuli_segments(),
                                    temperature=temperature, window_length=window_length, lambda_clip=lambda_clip_loss,
                                    lambda_lower_bound=lambda_sim_loss,
                                    lambda_discriminative=lambda_sim_loss)

    elif model_arch == 'no_contrastive_learning':
        model = CLIPNoContrastiveLearning(eeg, speech, window_length=window_length)


    model.to(device)


    # choose the optimizer (Adam or AdamW)
    if optimizer == 'adam':
        optimizer_all = Adam(model.parameters(),
                             betas=(beta1, beta2),
                             amsgrad=use_amsgrad,
                             lr=learning_rate)
    elif optimizer == 'adamw':
        optimizer_all = AdamW(model.parameters(),
                                betas=(beta1, beta2),
                                amsgrad=use_amsgrad,
                                weight_decay=weight_decay,
                             lr=learning_rate)

    # Set the learning rate scheduler
    if lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_all, mode='min', factor=0.1, patience=5, verbose=True)
    elif lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all, step_size=step_size_scheduler, gamma=0.1)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr.scheduler.CosineAnnealingLR(optimizer_all, T_max=10, eta_min=0)
    elif lr_scheduler == 'cosine_warmup':
        scheduler = torch.optim.lr.scheduler.CosineAnnealingWarmRestarts(optimizer_all, T_0=10, T_mult=2, eta_min=0, last_epoch=-1)
    else:
        scheduler = optimizer_all



    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


    print(f'number of parameters {get_n_params(model)}')
    print(f'number of parameters speech  {get_n_params(speech)}')
    print(f'number of parameters eeg  {get_n_params(eeg)}')

    if not only_evaluate:
        # if there is a model already trained, load it
        if os.path.exists(os.path.join(checkpoint_path, f'model.ckpt')):
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'model.ckpt')))
            print(f'loaded model from {os.path.join(checkpoint_path, f"model.ckpt")}')
        else:
            print(f'no model found at {os.path.join(checkpoint_path, f"model.ckpt")}, training from scratch')


        # early stopping parameters
        best_loss = torch.inf
        best_epoch = 0
        best_state_dict = {}
        early_stopping_patience = patience


        # training loop
        for epoch in range(epochs):

            # check if we need to stop according to early stopping criteria
            if( epoch > best_epoch + early_stopping_patience) and (epoch > warmup_epochs ):
                # restore the best model
                model.load_state_dict(best_state_dict)
                printf(f'early stopping at epoch {epoch}', file_loss)

                break

            train_losses = []
            train_accuracies = []

            model.train()

            for batch, data in enumerate(train_loader):
                eeg = data[0].to(device, dtype=torch.float)
                speech = data[1][0].to(device, dtype=torch.float)
                # ids are a list of strings
                ids = data[2].to(device, dtype=torch.int64)
                if model_arch == 'clip_kld' or model_arch == 'clip_kld_latent_proj':
                    loss_total, loss_ce, log_pmu2, log_z2 = model(eeg,speech, ids)

                else:
                    loss_ce, loss_average_eeg , loss_total = model(eeg,speech, ids)

                optimizer_all.zero_grad()

                if epoch >= warmup_epochs:
                    loss_total.backward()
                else:
                    loss_ce.backward() # loss_ce and loss_total only different in normal model if we are using memorybank (lambda_sim_loss >0), which by default is not the case
                optimizer_all.step()

                if batch%100 == 0:
                    if model_arch == 'clip_kld' or model_arch == 'clip_kld_latent_proj':
                        printf(
                            f'train epoch {epoch} batch {batch} loss_ce  {loss_ce.item()} loss pmu2 {log_pmu2.item()}, log z2: {log_z2.item()}',
                            file_loss)
                    else:
                        printf(f'train epoch {epoch} batch {batch} loss_ce  {loss_ce.item()} loss average eeg {loss_average_eeg.item()}', file_loss)

            # scheduler update
            if lr_scheduler != 'no':
                scheduler.step()

            # validation loss
            model.eval()
            losses_ce = []
            losses_average_eeg = []
            losses_total = []
            with torch.no_grad():
                for batch, (data) in enumerate(val_loader):
                    eeg = data[0].to(device, dtype=torch.float)
                    speech = data[1][0].to(device, dtype=torch.float)
                    ids = data[2].to(device, dtype=torch.int64)
                    if model_arch == 'clip_kld' or model_arch == 'clip_kld_latent_proj':
                        loss_total, loss_ce, loss_average_eeg, log_z2 = model(eeg, speech, ids)
                    else:

                        loss_ce, loss_average_eeg , loss_total = model(eeg,speech, ids)
                    losses_ce.append(loss_ce)
                    losses_average_eeg.append(loss_average_eeg)
                    losses_total.append(loss_total)


            mean_loss_ce = torch.mean(torch.hstack(losses_ce)).item()
            mean_loss_average_eeg = torch.mean(torch.hstack(losses_average_eeg)).item()
            mean_loss_total = torch.mean(torch.hstack(losses_total)).item()
            printf(f'validation epoch {epoch}: mean loss ce : {mean_loss_ce}, mean loss average: {mean_loss_average_eeg}, mean loss total: {mean_loss_total}', file_loss)

            if mean_loss_ce < best_loss:
                if checkpoint_path is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_path, f'model.ckpt')
                    )

                best_loss = mean_loss_ce
                best_epoch = epoch
                best_state_dict = model.state_dict()
    else:

        checkpoint_path = os.path.join(results_folder, 'checkpoints')

        # load best state_dict
        pretrained_dict = torch.load(os.path.join(checkpoint_path, f'model.ckpt'), map_location=device)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict)

        print(f'loaded model from {os.path.join(checkpoint_path, f"model.ckpt")}')

    # evaluation of the two downstream tasks - match/mismatch and regression
    if True:
        testfolder = os.path.join(os.path.dirname(data_folder), 'ICASSP-2023-eeg-decoding-challenge-dataset','TEST_task1_matchmismatch')

        evalutation, evaluation_with_logits, evaluation_top_x, evaluation_top_x_with_logits = evaluate_model_challenge_2023_mm(model, device, speech_feature=stimulus_features,
                                                       eeg_folder = testfolder)

        # save evaluation
        with open(os.path.join(results_folder, 'evaluation_challenge_set_2023_mm.json'), 'w') as f:
            json.dump(evalutation, f)

        with open(os.path.join(results_folder, 'evaluation_challenge_set_2023_mm_logits.json'), 'w') as f:
            json.dump(evaluation_with_logits, f)

        with open(os.path.join(results_folder, 'evaluation_challenge_set_2023_mm_top_x.json'), 'w') as f:
            json.dump(evaluation_top_x, f)

        with open(os.path.join(results_folder, 'evaluation_challenge_set_2023_mm_top_x_logits.json'), 'w') as f:
            json.dump(evaluation_top_x_with_logits, f)

    if True:
        # evaluate model
        evalutation = evaluate_model_do_regression_sub_specific(model, train_files, val_files,
                                                                test_files, train_audio, val_audio, test_audio, device, results_folder,
                                                                regress_to='envelope', window_length=3, fs=64)

    if True:
        evalutation = evaluate_model_do_regression_sub_independent(model, train_files, val_files,
                                                                test_files, train_audio, val_audio, test_audio,
                                                                   device, results_folder,
                                                                   regress_to='envelope', window_length=3, fs=64)


    if True:
        test_folder = os.path.join(os.path.dirname(data_folder), 'ICASSP-2023-eeg-decoding-challenge-dataset',
                                  'TEST_task2_regression')
        evalutation, evalutation_sub_specific = evaluate_model_challenge_2023_regression(model,results_folder, device,eeg_folder=test_folder)

        # save evaluation
        with open(os.path.join(results_folder, 'evaluation_challenge_set_2023_regression.json'), 'w') as f:
            json.dump(evalutation, f)

        with open(os.path.join(results_folder, 'evaluation_challenge_set_2023_regression_sub_specific.json'), 'w') as f:
            json.dump(evalutation_sub_specific, f)

