import time

import os
import sys
import gc
import glob
import argparse
import json
import pickle
import os, sys
import numpy as np
import torch
from dataset_loader import EEGDatasetSimdata
import torch.nn.functional as F
import scipy.stats
import scipy.signal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

def printf(s, file):
    print(s)
    with open(file, 'a') as file:
        file.write(s + '\n')


def load_labels_match_mismatch_2023(path_true_labels):
    # load the true labels for all subjects
    labels_all= {}
    for file in glob.glob(os.path.join(path_true_labels, '*.json')):
        with open(file, 'r') as f:
            labels = json.load(f)
            # update labels_all, but only keep the label as the value
            # (not the subject and the label)
            labels_all.update(labels)
    return labels_all

def load_labels_regression_2023(path_true_labels, subject=None):
    # load the true labels for all subjects
    labels_all= {}
    all_files = glob.glob(os.path.join(path_true_labels, '*.json'))
    if subject is not None:
        all_files = [x for x in all_files if subject in x]

    for file in all_files:
        with open(file, 'r') as f:
            labels = json.load(f)
            # update labels_all, but only keep the label as the value
            # (not the subject and the label)
            labels_all.update(labels)
    return labels_all

def evaluate_model_challenge_2023_mm(model, device, subject=None, speech_feature='omsimel',
                                     eeg_folder='' ):

    data_folder = eeg_folder

    path_labels_match_mismatch = os.path.join(data_folder, 'labels')  # labels.json with the correct labels

    stimulus_folder = os.path.join(data_folder, 'wav2vec_segments_wholefile_64hz/')

    all_groundtruth_labels = load_labels_match_mismatch_2023(path_labels_match_mismatch)

    # uncomment if you want to train with the mel spectrogram stimulus representation
    stimulus_features = [speech_feature]

    model.eval()

    evaluation = {}
    evaluation_with_logits = {}
    evaluation_top_x = {}
    evaluation_top_x_with_logits = {}
    if subject is not None:
        test_eeg_mapping =[os.path.join(eeg_folder, f'{subject}.json')]
        # load stimulus mapping
        try:
            sub_stimulus_mapping = json.load(open(test_eeg_mapping[0]))
        except:
            print(f'error with {sub_stimulus_mapping}')
            return evaluation, evaluation_with_logits
        # get the unique used stimuli
        stimuli = list(set([sub_stimulus_mapping[key][1].split('_-_')[0] for key in sub_stimulus_mapping.keys()]))
        test_stimuli = glob.glob(os.path.join(stimulus_folder, f'*{stimulus_features[0]}.pkl'))
        # filter on if stimulus is in stimuli
        test_stimuli = [x for x in test_stimuli if os.path.basename(x).split('_-_')[1] in stimuli]

    else:
        test_eeg_mapping = glob.glob(os.path.join(eeg_folder, 'sub*.json'))
        test_stimuli = glob.glob(os.path.join(stimulus_folder, f'*{stimulus_features[0]}.pkl'))

    # load all test stimuli
    test_stimuli_data = {}
    test_stimuli_data_embeddings = {}
    for stimulus_path in test_stimuli:
        with open(stimulus_path, 'rb') as f:
            data = pickle.load(f)
            test_stimuli_data.update(data)

        # get the embeddings
        keys = list(data.keys())
        if data[keys[-1]].shape != data[keys[-2]].shape:
            keys = keys[:-1]
        stimulus_segments = np.stack([data[key] for key in keys])

        with torch.no_grad():
            stimulus_segments = torch.from_numpy(stimulus_segments).to(device, dtype=torch.float)
            stimulus_embeddings = model.speechModel(stimulus_segments)
            stimulus_embeddings = torch.flatten(stimulus_embeddings, start_dim=1)
            stimulus_embeddings = F.normalize(stimulus_embeddings, p=2, dim=1)
        test_stimuli_data_embeddings.update({key: stimulus_embeddings[i] for i, key in enumerate(keys)})

    print(f'number of test stimuli: {len(list(test_stimuli_data.keys()))}')
    print(f'number of test stimuli embeddings: {len(list(test_stimuli_data_embeddings.keys()))}')

    list_keys_stimuli = list(test_stimuli_data_embeddings.keys())
    if not list_keys_stimuli:
        print("No test stimuli found for match-mismatch evaluation, skipping.")
        return {}, {}, {}, {}
    list_torch_stimuli_embeddings = torch.stack([test_stimuli_data_embeddings[key] for key in list_keys_stimuli])



    for sub_stimulus_mapping in test_eeg_mapping:
        subject = os.path.basename(sub_stimulus_mapping).split('.')[0]
        print(f'evaluating {subject}')

        # load stimulus mapping
        try:
            sub_stimulus_mapping = json.load(open(sub_stimulus_mapping))
        except:
            print(f'error with {sub_stimulus_mapping}')
            continue
        id_list = list(sub_stimulus_mapping.keys())

        data_eeg = np.stack([sub_stimulus_mapping[key][0] for key in id_list])
        data_eeg = np.squeeze(data_eeg)

        data_eeg_mvn = (data_eeg - np.mean(data_eeg, axis=(0,1), keepdims=True)) / np.std(data_eeg, axis=(0,1), keepdims=True)

        # get the labels
        labels_in_order = [all_groundtruth_labels[x] for x in id_list]
        correct_keys = [sub_stimulus_mapping[key][all_groundtruth_labels[key]+1].split('.')[0] for key in id_list]


        with (torch.no_grad()):

            # do the same for mvn_eeg data
            eeg_mvn = torch.from_numpy(data_eeg_mvn).to(device, dtype=torch.float)
            eeg_embeddings_mvn = model.eegModel(eeg_mvn)
            eeg_embeddings_mvn = torch.flatten(eeg_embeddings_mvn, start_dim=1)
            eeg_embeddings_mvn = F.normalize(eeg_embeddings_mvn, p=2, dim=1)

            # get the speech embeddings

            speech_embeddings = torch.stack([torch.stack([test_stimuli_data_embeddings[sub_stimulus_mapping[key][1].split('.')[0]],
                                                          test_stimuli_data_embeddings[sub_stimulus_mapping[key][2].split('.')[0]]]) for key in id_list])

            correct_labels = torch.Tensor(labels_in_order).to(device, dtype=torch.float)

            # do the same for mvn_eeg
            logits_mvn = [torch.matmul(eeg_embeddings_mvn, speech_embeddings[:,j].T) for j in range(len(speech_embeddings[1]))]
            logits_mvn = [torch.diag(logit) for logit in logits_mvn]
            speech_eeg_logits_mvn = torch.stack(logits_mvn)
            max_sim_mvn = torch.argmax(speech_eeg_logits_mvn, axis=0)
            accuracy_mvn = torch.sum(max_sim_mvn == correct_labels) / max_sim_mvn.shape[0]
            evaluation[subject+'_mvn'] = accuracy_mvn.item()
            print(f"evaluation mm with mvn : {evaluation[subject+'_mvn']}, {subject}")

            # also save the logits, along with their labels_in_order key
            logits_with_labels = {key: (speech_eeg_logits_mvn[:,i].tolist(), labels_in_order[i]) for i, key in enumerate(id_list)}
            evaluation_with_logits[subject] = logits_with_labels

            # now we want to calculate the top x accuracy, with x = 1..100
            # find the correct place of the label in the list_keys_stimuli
            # the correct keys are in correct_keys

            # get the index of the correct keys in the list_keys_stimuli
            correct_keys_idx = [list_keys_stimuli.index(x) for x in correct_keys]
            # this is our logits true label, put on the gpu
            correct_keys_idx = torch.Tensor(correct_keys_idx).to(device)

            # transpose the list_torch_stimuli_embeddings

            logits = torch.matmul(eeg_embeddings_mvn, torch.transpose(list_torch_stimuli_embeddings, 0, 1))

            # get top 10
            maxtop = min(100, logits.shape[1])
            # topk in torch
            topk = torch.topk(logits, k=maxtop, dim=1)
            # get labels and set them as integers
            labels= correct_keys_idx.to(torch.int).cpu()
            labels = np.array(labels, dtype=np.int32)

            labels = np.reshape(np.repeat(labels, maxtop), (len(labels), -1))

            is_correct = np.equal(labels, topk.indices.cpu().numpy())
            # cast is_correct to int


            correct_top = np.cumsum(is_correct, axis=1)
            correct_top = np.mean(correct_top, axis=0)
            evaluation_top_x[subject] = correct_top.tolist()

            evaluation_top_x_with_logits[subject] = {'logits': logits.tolist(), 'correct_keys_idx': correct_keys_idx.tolist(), 'correct_top': correct_top.tolist()}
            print(f"evaluation mm top x: {subject} : top1 {evaluation_top_x[subject][0]*100}, top10: {evaluation_top_x[subject][9]*100}")

    return evaluation, evaluation_with_logits, evaluation_top_x, evaluation_top_x_with_logits


def evaluate_model_challenge_2023_regression(model, results_folder, device, subject=None,
                                             eeg_folder='' ):


    data_folder = eeg_folder
    path_labels_regression = os.path.join(data_folder, 'labels')
    # labels.json with the correct labels
    labels_regression = load_labels_regression_2023(path_labels_regression, subject=subject)

    if not labels_regression:
        print("No labels found for regression evaluation, skipping.")
        return {}, {}

    # get dimension of labels
    time_dim = len(list(labels_regression.values())[0][0])

    # load the general regression model
    model_regression_path = os.path.join(results_folder, 'regression_model_general_env.pth')
    # load

    # model_regression = RegressionModel(model.eegModel.final_layer.out_features, output_dim=1)
    model_regression = RegressionModel(8, output_dim=1)
    model_regression.to(device)
    model_regression.load_state_dict(torch.load(model_regression_path, map_location=device))
    model_regression.eval()

    model.eval()
    evaluation = {}
    evaluation_sub_specific = {}
    if subject is not None:
        test_eeg_mapping = [os.path.join(eeg_folder, f'{subject}.json')]
    else:
        test_eeg_mapping = glob.glob(os.path.join(eeg_folder, 'sub*.json'))

    for sub_data in test_eeg_mapping:
        subject = os.path.basename(sub_data).split('.')[0]
        model_regression_path_sub_specific = os.path.join(results_folder,'sub_specific', f'regression_model_{subject}.pth')
        print(f'evaluating {subject}')

        # load stimulus mapping
        try:
            sub_data = json.load(open(sub_data))
        except:
            print(f'error with {sub_data}')
            continue
        id_list = list(sub_data.keys())

        data_eeg = np.stack([sub_data[key] for key in id_list])
        data_eeg = np.squeeze(data_eeg)

        data_eeg_mvn = (data_eeg - np.mean(data_eeg, axis=(0,1), keepdims=True)) / np.std(data_eeg, axis=(0,1), keepdims=True)

        # split the data into chunks of 3 seconds, with overlap of 50%
        time_window = model.window_length
        data_eeg_mvn2 = np.stack([data_eeg_mvn[:, i:i+time_window] for i in range(0, data_eeg_mvn.shape[1], time_window//2)][:-1])

        # transpose axis 0 and 1
        data_eeg_mvn2 = np.transpose(data_eeg_mvn2, (1, 0, 2, 3))

        final_outputs = []
        final_outputs_sub_specific = []

        with (torch.no_grad()):
            for i in range(data_eeg_mvn2.shape[0]):
                # do the same for mvn_eeg data
                eeg_mvn = torch.from_numpy(data_eeg_mvn2[i]).to(device, dtype=torch.float)
                eeg_embeddings = model.eegModel(eeg_mvn)
                # # for env, time dimension is the last. check if there is a stride used in the model
                # if eeg_embeddings.shape[1] != time_dim:
                #     # find stride
                #     stride = int(time_dim / eeg_embeddings.shape[1])
                #     # do upsamplig with factor stride
                #     eeg_embeddings = torch.transpose(
                #         F.interpolate(torch.transpose(eeg_embeddings, 1, 2), scale_factor=stride, mode='nearest'), 1, 2)
                #
                #     # check if the shape is now the same, if not, extrapolate the last values until we have the same shape
                #     if eeg_embeddings.shape[1] != time_dim:
                #         # extrapolate the last values
                #         eeg_embeddings = torch.cat([eeg_embeddings, eeg_embeddings[:, -1:, :].repeat(1, time_dim -
                #                                                                                      eeg_embeddings.shape[
                #                                                                                          1], 1)], dim=1)
                #     elif eeg_embeddings.shape[1] > time_dim:
                #         # cut off the last values
                #         eeg_embeddings = eeg_embeddings[:, 0:time_dim, :]
                #

                # do the regression
                # transpose the data

                eeg_embeddings = torch.transpose(eeg_embeddings, 1, 2)

                # for the general model
                model_regression.load_state_dict(torch.load(model_regression_path, map_location=device))
                regression_output = model_regression(eeg_embeddings)
                # sqeeuze
                regression_output = torch.squeeze(regression_output)


                # now create one big regression envelope, by adding the segments together using overlap add. There was an overlap of 50%
                # do this using overlap add and add a hann window to the data
                # first create a hann window
                hann = torch.hann_window(eeg_embeddings.shape[2]).to(device)
                hann = hann.repeat(eeg_embeddings.shape[0], 1)
                # first half of first regression output and last half of last regression output should be one
                hann[0, 0:time_window//2] = 1
                hann[-1, time_window//2:] = 1
                # multiply the regression output with the hann window
                regression_output = regression_output * hann

                # now do the overlap add
                # first create a matrix with zeros
                regression_output_final = torch.zeros((1, time_dim)).to(device)
                # put each regression output in the right place
                for j in range(eeg_embeddings.shape[0]):
                    regression_output_final[0, j * time_window // 2:j * time_window // 2 + time_window] += regression_output[j]


                final_outputs.append(regression_output_final)

                # for the sub specific model, if it exists
                if os.path.exists(model_regression_path_sub_specific):
                    model_regression.load_state_dict(torch.load(model_regression_path_sub_specific, map_location=device))
                    regression_output = model_regression(eeg_embeddings)
                    # sqeeuze
                    regression_output = torch.squeeze(regression_output)

                    # now create one big regression envelope, by adding the segments together using overlap add. There was an overlap of 50%
                    # do this using overlap add and add a hann window to the data
                    # first create a hann window
                    hann = torch.hann_window(eeg_embeddings.shape[2]).to(device)
                    hann = hann.repeat(eeg_embeddings.shape[0], 1)
                    # first half of first regression output and last half of last regression output should be one
                    hann[0, 0:time_window // 2] = 1
                    hann[-1, time_window // 2:] = 1
                    # multiply the regression output with the hann window
                    regression_output = regression_output * hann

                    # now do the overlap add
                    # first create a matrix with zeros
                    regression_output_final = torch.zeros((1, time_dim)).to(device)
                    # put each regression output in the right place
                    for j in range(eeg_embeddings.shape[0]):
                        regression_output_final[0, j * time_window // 2:j * time_window // 2 + time_window] += \
                        regression_output[j]

                    final_outputs_sub_specific.append(regression_output_final)


            # get the labels
            labels_in_order = [labels_regression[x] for x in id_list]

            # calculate the pearson correlation, for all labels separately
            pearson_corr = [scipy.stats.pearsonr(final_outputs[i][0].cpu().numpy(), np.squeeze(np.array(labels_in_order[i][0])))[0] for i in range(len(labels_in_order))]

            # also create a json file with the predictions for this subjects
            os.makedirs(os.path.join(results_folder,'regression_2023_icassp'), exist_ok=True)

            evaluation[subject] = np.mean(pearson_corr)
            print(f"evaluation regression: {evaluation[subject]}, {subject}")

            with open(os.path.join(results_folder,'regression_2023_icassp', f'{subject}_predictions.json'), 'w') as f:
                json.dump({key: final_outputs[i][0].tolist() for i, key in enumerate(id_list)}, f)

            if os.path.exists(model_regression_path_sub_specific):
                pearson_corr_sub_specific = [scipy.stats.pearsonr(final_outputs_sub_specific[i][0].cpu().numpy(), np.squeeze(np.array(labels_in_order[i][0])))[0] for i in range(len(labels_in_order))]
                evaluation_sub_specific[subject] = np.mean(pearson_corr_sub_specific)
                print(f"evaluation regression sub specific: {evaluation_sub_specific[subject]}, {subject}")

                with open(os.path.join(results_folder,'regression_2023_icassp', f'{subject}_predictions_sub_specific.json'), 'w') as f:
                    json.dump({key: final_outputs_sub_specific[i][0].tolist() for i, key in enumerate(id_list)}, f)


    return evaluation, evaluation_sub_specific

# 18/12: CHANGED
def get_train_val_test_files_final(data_folder,run, stimulus_feature, dataset_split_stories, number_of_training_subjects, debug=False):
    # open
    with open(dataset_split_stories) as json_file:
        data_split = json.load(json_file)

    all_eeg_files = glob.glob(os.path.join(data_folder, 'derivatives', 'preprocessed_eeg', "**/*_eeg.npy"), recursive=True)
    all_audio_files = glob.glob(os.path.join(data_folder, 'derivatives', 'preprocessed_stimuli', f"**/*{stimulus_feature}.npy"),
                                    recursive=True)

    test_split = 'test_set_2023_1'
    val_split = f'{run}'
    # split files in train, val and test
    test_stories = data_split[test_split]
    val_stories = data_split[val_split]
    train_stories = np.concatenate(
        [data_split[f'{x}'] for x in range(0, 9) if f'{x}' != test_split and f'{x}' != val_split])

    # if debug, only take one story
    if debug:
        test_stories = test_stories[0:1]
        val_stories = val_stories[0:1]
        train_stories = train_stories[0:2]

    train_subjects = data_split['train_subs'][0:number_of_training_subjects]
    test_subjects = data_split['test_subs']

    print(f'number of training subjects {len(train_subjects)}')
    print(f'training subs: {train_subjects}')

    split_idx_subject = 0

    # first filter on subjects, to get subs for test set 2 vs train + val + test subs
    test_files_heldout = [x for x in all_eeg_files if (os.path.basename(x).split("_")[split_idx_subject] in test_subjects) ]
    files_seen_subs = [x for x in all_eeg_files if os.path.basename(x).split("_")[split_idx_subject] in train_subjects]

    test_files  = [x for x in files_seen_subs if (os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] in test_stories) ]
    val_files   = [x for x in files_seen_subs if (os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] in val_stories)]
    train_files = [x for x in files_seen_subs if (os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] in train_stories)]

    # get a list of the distinct stories per split and add the audio files
    test_stories_heldout = list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in test_files_heldout]))
    test_stories = list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in test_files]))
    val_stories = list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in val_files]))
    train_stories = list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in train_files]))

    test_audio_heldout = [x for x in all_audio_files if (os.path.basename(x).split('_-_')[0] in test_stories_heldout)]
    test_audio = [x for x in all_audio_files if (os.path.basename(x).split('_-_')[0] in test_stories)]
    val_audio = [x for x in all_audio_files if (os.path.basename(x).split('_-_')[0] in val_stories)]
    train_audio = [x for x in all_audio_files if (os.path.basename(x).split('_-_')[0] in train_stories)]

    # if debug, only keep some files
    if debug:
        train_files =train_files[0:5]
        val_files = val_files[0:5]
        test_files = test_files[0:5]
        test_files_heldout = test_files_heldout[0:5]

    return train_files, val_files, test_files , test_files_heldout, train_audio, val_audio, test_audio, test_audio_heldout


def evaluate_model_do_regression_sub_specific(model, train_files,val_files, test_files, train_files_audio, val_files_audio, test_files_audio, device,result_folder, regress_to=['env', 'mel'], window_length=5, fs=64):
    # we have the train and test files, we want to do regression to the envelope
    # we will use the model to do this
    # first calculate for all the train_files per subject the embeddings
    # then do regression to the envelope
    # then calculate the correlation between the predicted envelope and the true envelope
    # then do the same for the test_files
    # then return the correlation for the train_files and the test_files

    # first create the result folder for sub specific models
    os.makedirs(os.path.join(result_folder, 'sub_specific'), exist_ok=True)

    model.eval()
    evaluation = {}
    evaluation_mel = {}
    all_subs = list(set([os.path.basename(x).split("_")[0] for x in train_files]))
    print(f'number of subjects {len(all_subs)}')
    for sub in all_subs:
        subject = sub
        try:
        # if True:
            print(f'subject {sub}')
            # create train and val loader
            train_files_sub = [x for x in train_files if os.path.basename(x).split("_")[0] in [sub]]
            val_files_sub = [x for x in val_files if os.path.basename(x).split("_")[0] in [sub]]
            test_files_sub = [x for x in test_files if os.path.basename(x).split("_")[0] in [sub]]

            # now for each file, get the audio file and add this to the train_files
            audio_stimuli_train =list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in train_files_sub ]))
            audio_stimuli_val =list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in val_files_sub ]))
            audio_stimuli_test =list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in test_files_sub ]))

            audio_files_train = [x for x in train_files_audio if os.path.basename(x).split("_-_")[0] in audio_stimuli_train ]
            audio_files_val = [x for x in val_files_audio if os.path.basename(x).split("_-_")[0] in audio_stimuli_val ]
            audio_files_test = [x for x in test_files_audio if os.path.basename(x).split("_-_")[0] in audio_stimuli_test ]

            # check if test files with sub present
            sub_had_test = len(test_files_sub) > 0
            sub_has_val = len(val_files_sub) > 0
            sub_has_train = len(train_files_sub) > 0
            if not sub_has_train:
                print(f'subject {sub} has no train files')
                continue
            if not sub_had_test:
                print(f'subject {sub} has no test files')
                continue
            if not sub_has_val:
                print(f'subject {sub} has no val files')
                val_files_sub = test_files_sub

            # first do the train files

            train_generator = EEGDatasetSimdata(train_files_sub, audio_files_train, window_length*fs, window_length*fs, batch_size=128)
            val_generator = EEGDatasetSimdata(val_files_sub, audio_files_val, window_length*fs, window_length*fs, batch_size=128)


            all_train_embeddings = []
            all_train_env = []
            all_train_mel = []
            all_val_embeddings = []
            all_val_env = []
            all_val_mel = []

            # get all the embeddings
            with (torch.no_grad()):
                for idx, data in enumerate(train_generator):
                    sub = data[0]
                    story = data[1]
                    # make into torch tensor and put on gpu
                    if len(data) != 5:
                        print(f'error with {sub} {story}')
                        continue
                    # if dim of env =4, prune one
                    if data[4].ndim == 4:
                        env = data[4][:, :, :, 0]
                    else:
                        env = data[4]

                    eeg = torch.from_numpy(data[2]).to(device, dtype=torch.float)

                    speech = torch.from_numpy(data[3]).to(device, dtype=torch.float)
                    env = torch.from_numpy(env).to(device, dtype=torch.float)

                    eeg_embeddings = model.eegModel(eeg)


                    # cut off env batch to shape of eeg batch
                    env = env[0:eeg.shape[0]]
                    # cut off speech batch to shape of eeg batch
                    speech = speech[0:eeg.shape[0]]

                    # for env, time dimension is the last. check if there is a stride used in the model
                    if eeg_embeddings.shape[1] != env.shape[1] :
                        # find stride
                        stride = int(env.shape[1] / eeg_embeddings.shape[1])
                        # do upsamplig with factor stride
                        eeg_embeddings = torch.transpose(F.interpolate(torch.transpose(eeg_embeddings, 1,2), scale_factor=stride, mode='nearest'), 1,2)

                        # check if the shape is now the same, if not, extrapolate the last values until we have the same shape
                        if eeg_embeddings.shape[1] != env.shape[1]:
                            # extrapolate the last values
                            eeg_embeddings = torch.cat([eeg_embeddings, eeg_embeddings[:,-1:,:].repeat(1, env.shape[1]-eeg_embeddings.shape[1], 1)], dim=1)
                        elif eeg_embeddings.shape[1] > env.shape[1]:
                            # cut off the last values
                            eeg_embeddings = eeg_embeddings[:,0:env.shape[1],:]


                        # different stride used, don't regress
                        # printf(f'error with {sub} {story}, different stride used in model', os.path.join(result_folder, 'error_regression.txt'))
                        # return evaluation

                    all_train_embeddings.append(eeg_embeddings)

                    all_train_env.append(env)
                    all_train_mel.append(speech)

                for idx, data in enumerate(val_generator):
                    sub = data[0]
                    story = data[1]
                    # make into torch tensor and put on gpu
                    if len(data) != 5:
                        print(f'error with {sub} {story}')
                        continue
                    # if dim of env =4, prune one
                    if data[4].ndim == 4:
                        env = data[4][:, :, :, 0]
                    else:
                        env = data[4]

                    eeg = torch.from_numpy(data[2]).to(device, dtype=torch.float)



                    speech = torch.from_numpy(data[3]).to(device, dtype=torch.float)
                    env = torch.from_numpy(env).to(device, dtype=torch.float)
                    env = env[0:eeg.shape[0]]
                    speech = speech[0:eeg.shape[0]]

                    eeg_embeddings = model.eegModel(eeg)
                    # for env, time dimension is the last. check if there is a stride used in the model
                    if eeg_embeddings.shape[1] != env.shape[1]:
                        # find stride
                        stride = int(env.shape[1] / eeg_embeddings.shape[1])
                        # do upsamplig with factor stride
                        eeg_embeddings = torch.transpose(F.interpolate(torch.transpose(eeg_embeddings, 1,2), scale_factor=stride, mode='nearest'), 1,2)
                        # check if the shape is now the same, if not, extrapolate the last values until we have the same shape
                        if eeg_embeddings.shape[1] != env.shape[1]:
                            # extrapolate the last values
                            eeg_embeddings = torch.cat([eeg_embeddings, eeg_embeddings[:, -1:, :].repeat(1,
                                                                                                         env.shape[1] -
                                                                                                         eeg_embeddings.shape[
                                                                                                             1], 1)],
                                                       dim=1)
                        elif eeg_embeddings.shape[1] > env.shape[1]:
                            # cut off the last values
                            eeg_embeddings = eeg_embeddings[:, 0:env.shape[1], :]


                    all_val_embeddings.append(eeg_embeddings)
                    all_val_env.append(env)
                    all_val_mel.append(speech)

            all_train_embeddings = torch.cat(all_train_embeddings, dim=0)
            all_train_env = torch.cat(all_train_env, dim=0)
            all_train_mel = torch.cat(all_train_mel, dim=0)
            all_val_embeddings = torch.cat(all_val_embeddings, dim=0)
            all_val_env = torch.cat(all_val_env, dim=0)
            all_val_mel = torch.cat(all_val_mel, dim=0)

            # swap the axes
            all_train_env = all_train_env.permute(0, 2, 1)
            all_val_env = all_val_env.permute(0, 2, 1)
            all_train_mel = all_train_mel.permute(0, 2, 1)
            all_val_mel = all_val_mel.permute(0, 2, 1)
            all_train_embeddings = all_train_embeddings.permute(0, 2, 1)
            all_val_embeddings = all_val_embeddings.permute(0, 2, 1)


            # do regression to the envelope
            #define the model
            regression_model = RegressionModel(all_train_embeddings.shape[1], output_dim=all_train_env.shape[1])
            regression_model.to(device)
            # train with the pearson loss
            criterion = PearsonLoss()
            optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.001)
            file_loss = os.path.join(result_folder, 'loss_regression.txt')
            # train the model
            best_epoch = 0
            best_val_loss = torch.inf
            patience = 10
            batch_size = 64
            for epoch in range(250):
                train_loss = []
                for i in range(0, all_train_embeddings.shape[0], batch_size):
                    # forward pass
                    outputs = regression_model(all_train_embeddings[i:i+batch_size].to(device))
                    loss = criterion(outputs, all_train_env[i:i+batch_size].to(device))
                    train_loss.append(loss.item())
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_lossses = []
                # calculate the validation loss
                for i in range(0, all_val_embeddings.shape[0], batch_size):
                    val_outputs = regression_model(all_val_embeddings[i:i + batch_size].to(device))
                    val_loss = criterion(val_outputs, all_val_env[i:i + batch_size].to(device))
                    val_lossses.append(val_loss.item())
                # print to file
                printf(f'epoch {epoch}, loss {np.mean(train_loss)}, val_loss {np.mean(val_lossses)}', file_loss)

                val_loss = np.mean(val_lossses)
                # check if val_loss is better
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    # save the model checkpoint from the best epoch

                    torch.save(regression_model.state_dict(),
                               os.path.join(result_folder, 'sub_specific', f'regression_model_{subject}.pth'))

                else:
                    if epoch - best_epoch > patience:
                        print(f'early stopping at epoch {epoch}')
                        # restore the best model
                        regression_model.load_state_dict(torch.load(
                            os.path.join(result_folder, 'sub_specific', f'regression_model_{subject}.pth')))

                        break




            # do the test files'

            test_generator = EEGDatasetSimdata(test_files_sub, audio_files_test, window_length * fs, window_length * fs,
                                            batch_size=128)
            # get all the embeddings
            all_test_embeddings = []
            all_test_env = []
            all_test_mel = []
            with (torch.no_grad()):
                for idx, data in enumerate(test_generator):
                    sub = data[0]
                    story = data[1]
                    # make into torch tensor and put on gpu
                    if len(data) != 5:
                        print(f'error with {sub} {story}')
                        continue
                    # if dim of env =4, prune one
                    if data[4].ndim == 4:
                        env = data[4][:, :, :, 0]
                    else:
                        env = data[4]

                    eeg = torch.from_numpy(data[2]).to(device, dtype=torch.float)
                    speech = torch.from_numpy(data[3]).to(device, dtype=torch.float)
                    env = torch.from_numpy(env).to(device, dtype=torch.float)

                    # cut of env to shape of eeg
                    env = env[0:eeg.shape[0]]
                    # cut off speech to shape of eeg
                    speech = speech[0:eeg.shape[0]]

                    eeg_embeddings = model.eegModel(eeg)
                    # for env, time dimension is the last. check if there is a stride used in the model
                    if eeg_embeddings.shape[1] != env.shape[1]:
                        # find stride
                        stride = int(env.shape[1] / eeg_embeddings.shape[1])
                        # do upsamplig with factor stride
                        eeg_embeddings = torch.transpose(F.interpolate(torch.transpose(eeg_embeddings, 1,2), scale_factor=stride, mode='nearest'), 1,2)
                        # check if the shape is now the same, if not, extrapolate the last values until we have the same shape
                        if eeg_embeddings.shape[1] != env.shape[1]:
                            # extrapolate the last values
                            eeg_embeddings = torch.cat([eeg_embeddings, eeg_embeddings[:, -1:, :].repeat(1,
                                                                                                         env.shape[1] -
                                                                                                         eeg_embeddings.shape[
                                                                                                             1], 1)],
                                                       dim=1)
                        elif eeg_embeddings.shape[1] > env.shape[1]:
                            # cut off the last values
                            eeg_embeddings = eeg_embeddings[:, 0:env.shape[1], :]

                    all_test_embeddings.append(eeg_embeddings)
                    all_test_env.append(env)
                    all_test_mel.append(speech)

            # do regression to the envelope
            all_test_embeddings = torch.cat(all_test_embeddings, dim=0)
            all_test_env = torch.cat(all_test_env, dim=0)
            all_test_mel = torch.cat(all_test_mel, dim=0)
            # swap the axes
            all_test_env = all_test_env.permute(0, 2, 1)
            all_test_embeddings = all_test_embeddings.permute(0, 2, 1)
            all_test_mel = all_test_mel.permute(0, 2, 1)

            test_outputs = regression_model(all_test_embeddings)
            test_loss = criterion(test_outputs, all_test_env)
            evaluation[sub] = test_loss.item()
            print(f'evaluation for subject {sub} is {evaluation[sub]}')


            # save evaluation to file, do it every time so we don't lose the results

            with open(os.path.join(result_folder, 'evaluation_regression.json'), 'w') as f:
                json.dump(evaluation, f)




            # try to free up some memory

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f'error with subject {sub}')
            printf(f'error with subject {sub}', os.path.join(result_folder, 'error_regression.txt'))
            printf(str(e), os.path.join(result_folder, 'error_regression.txt'))
            continue
    return evaluation


def evaluate_model_do_regression_sub_independent(model, train_files,val_files, test_files, train_files_audio, val_files_audio, test_files_audio,device,result_folder, regress_to='env', window_length=5, fs=64):
    # we have the train and test files, we want to do regression to the envelope
    # we will use the model to do this
    # first calculate for all the train_files per subject the embeddings
    # then do regression to the envelope
    # then calculate the correlation between the predicted envelope and the true envelope
    # then do the same for the test_files
    # then return the correlation for the train_files and the test_files

    model.eval()
    evaluation = {}
    all_subs = list(set([os.path.basename(x).split("_")[0] for x in train_files+ val_files + test_files]))

    print(f'number of subjects {len(all_subs)}')

    train_files_sub = [x for x in train_files if os.path.basename(x).split("_")[0] in all_subs]
    val_files_sub = [x for x in val_files if os.path.basename(x).split("_")[0] in all_subs]
    test_files_sub = [x for x in test_files if os.path.basename(x).split("_")[0] in all_subs]

    # now for each file, get the audio file and add this to the train_files
    audio_stimuli_train = list(
        set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in train_files_sub]))
    audio_stimuli_val = list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in val_files_sub]))
    audio_stimuli_test = list(set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in test_files_sub]))

    audio_files_train = [x for x in train_files_audio if os.path.basename(x).split("_-_")[0] in audio_stimuli_train]
    audio_files_val = [x for x in val_files_audio if os.path.basename(x).split("_-_")[0] in audio_stimuli_val]
    audio_files_test = [x for x in test_files_audio if os.path.basename(x).split("_-_")[0] in audio_stimuli_test]


    # first do the train files

    train_generator = EEGDatasetSimdata(train_files_sub, audio_files_train, window_length * fs, window_length * fs,
                                        batch_size=128)
    val_generator = EEGDatasetSimdata(val_files_sub, audio_files_val, window_length * fs, window_length * fs,
                                      batch_size=128)


    all_train_embeddings = []
    all_train_env = []
    all_train_mel   = []
    all_val_embeddings = []
    all_val_env = []
    all_val_mel = []

    # check if we still need to train a model
    if os.path.exists(os.path.join(result_folder, 'regression_model_general_env.pth')):
        # load one exampple to get the shape
        with (torch.no_grad()):
            for idx, data in enumerate(train_generator):
                sub = data[0]
                story = data[1]
                # make into torch tensor and put on gpu
                if len(data) != 5:
                    print(f'error with {sub} {story}')
                    continue
                # if dim of env =4, prune one
                if data[4].ndim == 4:
                    env = data[4][:, :, :, 0]
                else:
                    env = data[4]

                eeg = torch.from_numpy(data[2]).to(device, dtype=torch.float)
                speech = torch.from_numpy(data[3]).to(device, dtype=torch.float)
                env = torch.from_numpy(env).to(device, dtype=torch.float)

                eeg_embeddings = model.eegModel(eeg)

                train_shape = eeg_embeddings.shape
                env_shape = env.shape
                mel_shape = speech.shape
                break
        print('loading model')
        regression_model = RegressionModel(train_shape[2], output_dim=env_shape[2])
        regression_model.load_state_dict(torch.load(os.path.join(result_folder, 'regression_model_general_env.pth')))
        regression_model.to(device)
        criterion = PearsonLoss()
    else:
        # get all the embeddings
        with (torch.no_grad()):
            for idx, data in enumerate(train_generator):
                sub = data[0]
                story = data[1]
                # make into torch tensor and put on gpu
                if len(data) != 5:
                    print(f'error with {sub} {story}')
                    continue
                # if dim of env =4, prune one
                if data[4].ndim == 4:
                    env = data[4][:, :, :, 0]
                else:
                    env = data[4]

                eeg = torch.from_numpy(data[2]).to(device, dtype=torch.float)
                speech = torch.from_numpy(data[3]).to(device, dtype=torch.float)
                env = torch.from_numpy(env).to(device, dtype=torch.float)

                eeg_embeddings = model.eegModel(eeg)

                # cut off env batch to shape of eeg batch
                env = env[0:eeg.shape[0]]
                # cut off speech batch to shape of eeg batch
                speech = speech[0:eeg.shape[0]]

                # for env, time dimension is the last. check if there is a stride used in the model
                if eeg_embeddings.shape[1] != env.shape[1]:
                    # find stride
                    stride = int(env.shape[1] / eeg_embeddings.shape[1])
                    # do upsamplig with factor stride
                    eeg_embeddings = torch.transpose(
                        F.interpolate(torch.transpose(eeg_embeddings, 1, 2), scale_factor=stride, mode='nearest'), 1, 2)

                    # check if the shape is now the same, if not, extrapolate the last values until we have the same shape
                    if eeg_embeddings.shape[1] != env.shape[1]:
                        # extrapolate the last values
                        eeg_embeddings = torch.cat([eeg_embeddings, eeg_embeddings[:, -1:, :].repeat(1, env.shape[1] -
                                                                                                     eeg_embeddings.shape[
                                                                                                         1], 1)], dim=1)
                    elif eeg_embeddings.shape[1] > env.shape[1]:
                        # cut off the last values
                        eeg_embeddings = eeg_embeddings[:, 0:env.shape[1], :]

                    # different stride used, don't regress
                    # printf(f'error with {sub} {story}, different stride used in model', os.path.join(result_folder, 'error_regression.txt'))
                    # return evaluation

                all_train_embeddings.append(eeg_embeddings.cpu())
                all_train_env.append(env.cpu())
                all_train_mel.append(speech.cpu())

            for idx, data in enumerate(val_generator):
                sub = data[0]
                story = data[1]
                # make into torch tensor and put on gpu
                if len(data) != 5:
                    print(f'error with {sub} {story}')
                    continue
                # if dim of env =4, prune one
                if data[4].ndim == 4:
                    env = data[4][:, :, :, 0]
                else:
                    env = data[4]

                eeg = torch.from_numpy(data[2]).to(device, dtype=torch.float)
                speech = torch.from_numpy(data[3]).to(device, dtype=torch.float)
                env = torch.from_numpy(env).to(device, dtype=torch.float)

                # cut off
                env = env[0:eeg.shape[0]]
                speech = speech[0:eeg.shape[0]]

                eeg_embeddings = model.eegModel(eeg)


                # for env, time dimension is the last. check if there is a stride used in the model
                if eeg_embeddings.shape[1] != env.shape[1]:
                    # find stride
                    stride = int(env.shape[1] / eeg_embeddings.shape[1])
                    # do upsamplig with factor stride
                    eeg_embeddings = torch.transpose(
                        F.interpolate(torch.transpose(eeg_embeddings, 1, 2), scale_factor=stride, mode='nearest'), 1, 2)

                    # check if the shape is now the same, if not, extrapolate the last values until we have the same shape
                    if eeg_embeddings.shape[1] != env.shape[1]:
                        # extrapolate the last values
                        eeg_embeddings = torch.cat([eeg_embeddings, eeg_embeddings[:, -1:, :].repeat(1, env.shape[1] -
                                                                                                     eeg_embeddings.shape[
                                                                                                         1], 1)], dim=1)
                    elif eeg_embeddings.shape[1] > env.shape[1]:
                        # cut off the last values
                        eeg_embeddings = eeg_embeddings[:, 0:env.shape[1], :]

                    # different stride used, don't regress
                    # printf(f'error with {sub} {story}, different stride used in model', os.path.join(result_folder, 'error_regression.txt'))
                    # return evaluation


                all_val_embeddings.append(eeg_embeddings.cpu())
                all_val_env.append(env.cpu())
                all_val_mel.append(speech.cpu())

        all_train_embeddings = torch.cat(all_train_embeddings, dim=0)
        all_train_env = torch.cat(all_train_env, dim=0)
        all_train_mel = torch.cat(all_train_mel, dim=0)
        all_val_embeddings = torch.cat(all_val_embeddings, dim=0)
        all_val_env = torch.cat(all_val_env, dim=0)
        all_val_mel = torch.cat(all_val_mel, dim=0)

        # swap the axes
        all_train_env = all_train_env.permute(0, 2, 1)
        all_val_env = all_val_env.permute(0, 2, 1)
        all_train_mel = all_train_mel.permute(0, 2, 1)
        all_train_embeddings = all_train_embeddings.permute(0, 2, 1)
        all_val_embeddings = all_val_embeddings.permute(0, 2, 1)
        all_val_mel = all_val_mel.permute(0, 2, 1)


        # do regression to the envelope
        #define the model
        regression_model = RegressionModel(all_train_embeddings.shape[1], output_dim=all_train_env.shape[1])
        regression_model.to(device)
        # train with the pearson loss
        criterion = PearsonLoss()
        optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.001)
        file_loss = os.path.join(result_folder, 'loss_regression_general_env.txt')
        # train the model
        best_epoch = 0
        best_val_loss = torch.inf
        patience = 10
        batch_size = 64
        for epoch in range(250):
            train_loss = []
            for i in range(0, all_train_embeddings.shape[0], batch_size):
                # forward pass
                outputs = regression_model(all_train_embeddings[i:i+batch_size].to(device))
                loss = criterion(outputs, all_train_env[i:i+batch_size].to(device))
                train_loss.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # calculate the validation loss
            val_outputs = regression_model(all_val_embeddings.to(device))
            val_loss = criterion(val_outputs, all_val_env.to(device))
            # print to file
            printf(f'epoch {epoch}, loss {np.mean(train_loss)}, val_loss {val_loss.item()}', file_loss)

            # check if val_loss is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # save the model checkpoint from the best epoch
                torch.save(regression_model.state_dict(), os.path.join(result_folder,'regression_model_general_env.pth'))

            else:
                if epoch - best_epoch > patience:
                    print(f'early stopping at epoch {epoch}')
                    # restore the best model
                    regression_model.load_state_dict(torch.load( os.path.join(result_folder,'regression_model_general_env.pth')))
                    break


    if not os.path.exists(os.path.join(result_folder, 'evaluation_regression_general_model_env.json')):
        # do the test files'
        # test per subject
        for sub in all_subs:
            test_files = [x for x in test_files_sub if os.path.basename(x).split("_")[0] in [sub]]
            if len(test_files_sub) == 0:
                continue
            # add audio
            # audio_files_test
            audio_stimuli_test = list(
                set([os.path.basename(x).split("-audio-")[-1].split('_eeg')[0] for x in test_files]))
            audio_files_test_sub = [x for x in audio_files_test if os.path.basename(x).split("_-_")[0] in audio_stimuli_test ]

            test_generator = EEGDatasetSimdata(test_files, audio_files_test_sub, window_length*fs, window_length*fs, batch_size=128)

            # get all the embeddings
            all_test_embeddings = []
            all_test_env = []

            with (torch.no_grad()):
                for idx, data in enumerate(test_generator):
                    sub = data[0]
                    story = data[1]
                    # make into torch tensor and put on gpu
                    if len(data) != 5:
                        print(f'error with {sub} {story}')
                        continue
                    # if dim of env =4, prune one
                    if data[4].ndim == 4:
                        env = data[4][:, :, :, 0]
                    else:
                        env = data[4]

                    eeg = torch.from_numpy(data[2]).to(device, dtype=torch.float)

                    env = torch.from_numpy(env).to(device, dtype=torch.float)

                    # cut of env to shape of eeg
                    env = env[0:eeg.shape[0]]

                    eeg_embeddings = model.eegModel(eeg)

                    # for env, time dimension is the last. check if there is a stride used in the model
                    if eeg_embeddings.shape[1] != env.shape[1]:
                        # find stride
                        stride = int(env.shape[1] / eeg_embeddings.shape[1])
                        # do upsamplig with factor stride
                        eeg_embeddings = torch.transpose(
                            F.interpolate(torch.transpose(eeg_embeddings, 1, 2), scale_factor=stride, mode='nearest'), 1, 2)

                        # check if the shape is now the same, if not, extrapolate the last values until we have the same shape
                        if eeg_embeddings.shape[1] != env.shape[1]:
                            # extrapolate the last values
                            eeg_embeddings = torch.cat([eeg_embeddings, eeg_embeddings[:, -1:, :].repeat(1, env.shape[1] -
                                                                                                         eeg_embeddings.shape[
                                                                                                             1], 1)], dim=1)
                        elif eeg_embeddings.shape[1] > env.shape[1]:
                            # cut off the last values
                            eeg_embeddings = eeg_embeddings[:, 0:env.shape[1], :]

                        # different stride used, don't regress
                        # printf(f'error with {sub} {story}, different stride used in model', os.path.join(result_folder, 'error_regression.txt'))
                        # return evaluation


                    all_test_embeddings.append(eeg_embeddings.cpu())
                    all_test_env.append(env.cpu())

            # do regression to the envelope
            all_test_embeddings = torch.cat(all_test_embeddings, dim=0)
            all_test_env = torch.cat(all_test_env, dim=0)
            # swap the axes
            all_test_env = all_test_env.permute(0, 2, 1)
            all_test_embeddings = all_test_embeddings.permute(0, 2, 1)
            # put regression model on cpu
            regression_model.to('cpu')


            test_outputs = regression_model(all_test_embeddings)
            test_loss = criterion(test_outputs, all_test_env)
            evaluation[sub] = test_loss.item()
            print(f'evaluation for subject {sub} is {evaluation[sub]}')

            # save evaluation to file, do it every time so we don't lose the results

            with open(os.path.join(result_folder, 'evaluation_regression_general_model_env.json'), 'w') as f:
                json.dump(evaluation, f)


    gc.collect()
    torch.cuda.empty_cache()


    return evaluation


# define a loss function in torch, which calculated the pearson correlation
class PearsonLoss(torch.nn.Module):
    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, x, y):
        # calculate the pearson correlation
        # pearson correlation in torch
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        pearson = cos(x - x.mean(dim=2, keepdim=True), y - y.mean(dim=2, keepdim=True))
        mean_pearson = torch.mean(pearson, axis=0)
        return -mean_pearson

# create a class which takes the mean of the last dimension for the pearson loss
class PearsonLossMean(torch.nn.Module):
    def __init__(self):
        super(PearsonLossMean, self).__init__()
        self.pearsonCalculator = PearsonLoss()

    def forward(self, x, y):
        # calculate the pearson correlation
        # pearson correlation in torch
        mean_pearson = self.pearsonCalculator(x, y)
        return mean_pearson.mean()

# define a simple model, easy to train
class RegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1, receptive_field=32):
        super(RegressionModel, self).__init__()
        self.conv = torch.nn.Conv1d(input_dim, output_dim, kernel_size=receptive_field, padding='same')
        self.activation = torch.nn.LeakyReLU()
        self.model = torch.nn.Sequential(self.conv, self.activation)

    def forward(self, x):
        return self.model(x)








