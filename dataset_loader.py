import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import itertools
from braindecode.augmentation import SignFlip, FTSurrogate, FrequencyShift, BandstopFilter, GaussianNoise, SmoothTimeMask, ChannelsDropout, ChannelsShuffle
import os
import pickle

import numpy as np


def batch_equalizer(data):
    """Batch equalizer.
    Prepares the inputs for a model to be trained in
    match-mismatch task. It makes sure that match_env
    and mismatch_env are equally presented as the first
    envelope in the match-mismatch task.

    Parameters
    ----------
    args : Sequence[np.ndarray]
        List of NumPy arrays representing feature data

    Returns
    -------
    Tuple[Tuple[np.ndarray], np.ndarray]
        Tuple of the EEG/speech features serving as the input to the model and
        the labels for the match/mismatch task

    Notes
    -----
    This function will also double the batch size. For example, if the batch size of
    the elements in each of the args was 32, the output features will have
    a batch size of 64.
    """
    eeg = data[0]
    num_stimuli = len(data) - 1

    # Repeat EEG data num_stimuli times
    new_eeg = np.concatenate([eeg] * num_stimuli, axis=0)
    all_features = [new_eeg]

    # Create sub-batches
    args_to_zip = [data[i::num_stimuli] for i in range(1, num_stimuli + 1)]
    for stimuli_features in zip(*args_to_zip):
        for i in range(num_stimuli):
            # Roll the stimulus features
            stimulus_rolled = np.roll(stimuli_features, shift=i, axis=0)
            # Reshape stimulus_rolled to merge the first two dimensions
            stimulus_rolled = stimulus_rolled.reshape(
                (stimulus_rolled.shape[0] * stimulus_rolled.shape[1], stimuli_features[0].shape[-2],
                 stimuli_features[0].shape[-1])
            )
            all_features.append(stimulus_rolled)

    # Create labels for the match/mismatch task
    labels = np.concatenate(
        [
            np.tile(np.array([[1 if ii == i else 0 for ii in range(num_stimuli)]], dtype=np.int32), (eeg.shape[0], 1))
            for i in range(num_stimuli)
        ],
        axis=0
    )

    return tuple(all_features), labels


class EEGDatasetSimdata(IterableDataset):

    def __init__(self, files, audio_files, window_length, hop_length, number_mismatch=None,
                 data_augmentation = [],data_augmentation_probability = 0.5, addEEG=False,
                 exclusion_list = [], batch_size=64, shuffle=False,shuffle_percentage=0.5, load_speech_memory=True, load_eeg_memory=True):

        self.exclusion_list = exclusion_list
        files = self.exclude_subjects(files, exclusion_list)
        self.eeg_files, self.audio_files = self.group_recordings(files, audio_files)

        self.eeg_in_memory = load_eeg_memory # not implemented yet currently - everything is always loaded
        self.speech_in_memory = load_speech_memory

        self.batch_size = batch_size
        self.window_length = window_length
        self.number_mismatch = number_mismatch
        self.hop_length = hop_length

        self.shuffle_subs = shuffle
        self.shuffle_percentage = shuffle_percentage
        self.addEEG = addEEG
        self.number_features = 2

        self.features_to_load = self.filter_features(self.audio_files)
        self.data_augmentation = self.init_data_augmentation(data_augmentation, data_augmentation_probability) # list of data augmentation techniques - which we will use from the braindecode library
        self.data_augmentation_probability = data_augmentation_probability


        self.idx=0


        if load_eeg_memory:
            self.eeg = self.load_files(self.eeg_files, self.audio_files)
            # self.eeg is a dict with story as key and then a dict with feature as key and then the data
            # eg. self.eeg['story1']['eeg'] = eeg_data for all subs [n_subs, len_batch, wl, 64]
            # eg. self.eeg['story1']['wav2vec'] = wav2vec_data for all subs [ len_batch,wl,  1024]
            # eg. self.eeg['story1']['env'] = env_data for all subs [ len_batch, wl, 1]
            # eg. self.eeg['story1']['mel'] = mel_data for all subs [ len_batch, wl, 28]
            # eg. self.eeg['story1']['identifiers'] = identifiers for all subs [ len_batch, 1]
            # eg. self.eeg['story1']['sub'] = sub for all subs [ nsubs]
            self.batches_keys = list(self.eeg.keys())




    def init_data_augmentation(self, list_of_augmentations, data_augmentation_probability):
        # initialize data augmentation
        BEST_AUG_PARAMS = {
            'GaussianNoise': {
                'std': 0.16,
            },
            'FrequencyShift': {
                'max_delta_freq': 0.6,
                'sfreq': 100
            },
            'FTSurrogate': {
                'phase_noise_magnitude': 1,
            },
            'SmoothTimeMask': {
                'mask_len_samples': 20,
            },
            'ChannelsDropout': {
                'p_drop': 0.2,
            },
            'ChannelsShuffle': {
                'p_shuffle': 0.6,
            },
            'IdentityTransform': {
            },
            'BandstopFilter': {
                'bandwidth': 0.6,
                'sfreq': 100
            },
            'ChannelsSymmetry': {
                'ordered_ch_names': ['Fz', 'Pz']
            }
        }
        # list_of_augmentations is a list of strings with the names of the augmentations
        # we want to apply
        augmentations = []
        for aug in list_of_augmentations:
            if aug == 'SignFlip':
                augmentations.append(SignFlip(probability=data_augmentation_probability))
            elif aug == 'FTSurrogate':
                augmentations.append(FTSurrogate(probability=data_augmentation_probability, **BEST_AUG_PARAMS['FTSurrogate']))
            elif aug == 'FrequencyShift':
                augmentations.append(FrequencyShift(probability=data_augmentation_probability, **BEST_AUG_PARAMS['FrequencyShift']))
            elif aug == 'BandstopFilter':
                augmentations.append(BandstopFilter(probability=data_augmentation_probability, **BEST_AUG_PARAMS['BandstopFilter']))
            elif aug == 'GaussianNoise':
                augmentations.append(GaussianNoise(probability=data_augmentation_probability, **BEST_AUG_PARAMS['GaussianNoise']))
            elif aug == 'SmoothTimeMask':
                augmentations.append(SmoothTimeMask(probability=data_augmentation_probability, **BEST_AUG_PARAMS['SmoothTimeMask']))
            elif aug == 'ChannelsDropout':
                augmentations.append(ChannelsDropout(probability=data_augmentation_probability, **BEST_AUG_PARAMS['ChannelsDropout']))
            elif aug == 'ChannelsShuffle':
                augmentations.append(ChannelsShuffle(probability=data_augmentation_probability, **BEST_AUG_PARAMS['ChannelsShuffle']))
        return augmentations

    def get_number_of_stimuli_segments(self):
        # get the total number of unique speech segemnets
        # this is the number of segments in the eeg data
        # we need this to create the labels - which are needed when you want to use a model which regularizes on the EEG segment, since then you need to know which EEG segments belong to the same stimulus
        number_of_segments = sum([self.eeg[story]['eeg'].shape[1] for story in self.eeg.keys()])
        return number_of_segments

    def filter_features(self, files):
        # get unique features
        features = []
        for story, files in files.items():
            for feature, speech_file in files.items():
                if feature not in features:
                    features.append(feature)
        return features

    def load_speech(self, files):
        # get all the speech files
        speech_data = {}
        for story, files in files.items():
            if story not in speech_data:
                speech_data[story] = {}
            for feature,  speech_file in files.items():
                if feature not in speech_data[story]:
                    with open(speech_file, 'rb') as f:
                        data = pickle.load(f)
                    speech_data[story][feature] = data

        return speech_data

    def load_files(self, eeg_dict, audio_dict):
        # get all the eeg files
        eeg_data = {}
        id_identifier_max = 0

        for story, eeg_files in eeg_dict.items():
            first = True
            audio_files = audio_dict[story]

            for feature_name in audio_files.keys():
                data = np.load(audio_files[feature_name])

                if 'wav2vec' not in feature_name:
                    data = np.concatenate(data, axis=0)

                if first:
                    len= data.shape[0]
                    # define number of split the story needs
                    number_batches = int(len/(self.hop_length*(self.batch_size-1)+self.window_length))
                    if number_batches == 0:
                        print('story', story, 'is too short, skipping this one')
                        break
                    len_per_batch = int(len/number_batches)
                    for i in range(number_batches):
                        eeg_data[story + '_batch_' + str(i)] = {}
                    first = False

                # split the data into batches
                data_split = np.split(data, range(len_per_batch, len, len_per_batch), axis=0)[:number_batches]
                # add to dict
                for i in range(number_batches):
                   eeg_data[story + '_batch_' + str(i)][feature_name] = self.split_into_windows(data_split[i])

            # now add the EEG
            if number_batches == 0:
                continue

            for eeg_file in eeg_files:
                data = np.load(eeg_file)
                data = np.transpose(data)

                # check dimension of EEG
                if data.shape[1] < 64:
                    # print to file
                    print(f'eeg to short: {eeg_file} , {data.shape}')
                    with open('eeg_too_short.txt', 'a') as f:
                        f.write(eeg_file + '\n')
                    continue

                # get the subject
                sub = os.path.basename(eeg_file).split("_")[0]

                # split the data into batches
                data_split = np.split(data, range(len_per_batch, len, len_per_batch), axis=0)[:number_batches]
                # add to dict
                for i in range(number_batches):
                    # ensure data_split[i] has dimension [len_per_batch, 64]
                    if data_split[i].shape[0] < len_per_batch:
                        # do some padding
                        data_split_t = np.concatenate([data_split[i], np.zeros((len_per_batch-data_split[i].shape[0], 64))], axis=0)
                    elif data_split[i].shape[0] > len_per_batch:
                        # do some cropping
                        data_split_t= data_split[i][:len_per_batch, :]
                    else:
                        data_split_t = data_split[i]
                    if 'eeg' not in eeg_data[story + '_batch_' + str(i)]:

                        # if dimension smaller than
                        eeg_data[story + '_batch_' + str(i)]['eeg'] = self.split_into_windows(data_split_t)[None,:]
                    else:
                        eeg_data[story + '_batch_' + str(i)]['eeg'] = np.concatenate([eeg_data[story + '_batch_' + str(i)]['eeg'], self.split_into_windows(data_split_t)[None,:]], axis=0)

                    # add unique indices to each segment
                    if 'identifiers' not in eeg_data[story + '_batch_' + str(i)]:
                        eeg_data[story + '_batch_' + str(i)]['identifiers'] = [j+id_identifier_max+1 for j in range(eeg_data[story + '_batch_' + str(i)]['eeg'].shape[1])]
                        id_identifier_max = eeg_data[story + '_batch_' + str(i)]['identifiers'][-1]

                    # add the subject
                    if 'sub' not in eeg_data[story + '_batch_' + str(i)]:
                        eeg_data[story + '_batch_' + str(i)]['sub'] = [sub]
                    else:
                        eeg_data[story + '_batch_' + str(i)]['sub'].append(sub)


        return eeg_data

    def exclude_subjects(self, files, exclusion_list):
        # filter out the subjects in the exclusion list
        files_filtered = []
        for file in files:
            if not any([x in file for x in exclusion_list]):
                files_filtered.append(file)
        return files_filtered

    def group_recordings(self, files, audio_files):
        """Group recordings and corresponding stimuli.

        Parameters
        ----------
        files : Sequence[Union[str, pathlib.Path]]
            List of filepaths to preprocessed and split EEG and speech features

        Returns
        -------
        list
            Files grouped by the self.group_key_fn and subsequently sorted
            by the self.feature_sort_fn.
        """
        eeg_dict = {}
        for file in files:
            story = os.path.basename(file).split("-audio-")[-1].split('_eeg')[0]

            if story not in eeg_dict:
                eeg_dict[story] = []

            # add the eeg file to
            eeg_dict[story].append(file)


        # put all the audio files in a dict with the story name as key
        audio_dict = {}
        for file in audio_files:
            story = os.path.basename(file).split('_-_')[0]

            # we only want the audio if there are corresponding EEG files
            if story not in eeg_dict:
                continue

            feature = os.path.basename(file).split('_-_')[1].split('.')[0]
            if story not in audio_dict:
                audio_dict[story] = {}
            audio_dict[story][feature] = file


        return eeg_dict, audio_dict

    def constructNewEEG(self, eeg_data):
        if not self.addEEG:
            return eeg_data
        # construct new EEG data
        # we go over the time axis and sometimes create new EEG, by randomly selecting a segment from the same story and subject, with probability augmentation_probability
        # we then add this segment to the EEG data

        probabilities = np.random.rand(eeg_data.shape[0], eeg_data.shape[1])
        mixup_alphas = np.random.rand(eeg_data.shape[0], eeg_data.shape[1])
        do_augmentation = probabilities < self.data_augmentation_probability

        mixup_factor_original = 1 -mixup_alphas*do_augmentation

        mixup_eeg_idx = np.random.randint(0, eeg_data.shape[0], [eeg_data.shape[0], eeg_data.shape[1]])

        eeg_data_orig = eeg_data.copy()

        for i in range(eeg_data.shape[0]):
            for j in range(eeg_data.shape[1]):

                # select a random segment from the same story and subject
                # get a random segment
                eeg_data[i,j] = mixup_factor_original[i,j]*eeg_data_orig[i,j] + (1-mixup_factor_original[i,j])*eeg_data_orig[mixup_eeg_idx[i,j],j]
        return eeg_data

    def load_features(self, recording_index):
        if self.eeg_in_memory:
            story = self.batches_keys[recording_index]
            data_feat = [self.eeg[story][feat] for feat in self.features_to_load]
            eeg = self.eeg[story]['eeg'] # eeg for all subs listening to this story, [n_subs, len_batch, wl, 64]
            ids = self.eeg[story]['identifiers']
            subs = self.eeg[story]['sub']

            eeg = self.constructNewEEG(eeg)

        else:
            print('no eeg in memory/ not implemented yet')
            pass

        return data_feat, eeg, ids, subs

    def __len__(self):
        return len(self.batches_keys) # number of stories

    def batch(self, data_list):
        # batch data into batches of batch_size
        # return a list of batches
        # each batch is a list of tuples with (eeg, speech)
        bs = self.batch_size
        eeg, speech = data_list[0], data_list[1]

        # return a tuple of tensors with batch_size = bs
        for i in range(0, len(eeg), bs):
            if len(eeg[i:i + bs]) < bs:
                i = eeg.shape[0]-bs
            eeg_batch = eeg[i:i + bs]
            speech_batch = speech[i:i + bs]
            yield eeg_batch, speech_batch

    def get_item(self):

        for idx in range(self.__len__()):

            speech, eeg, ids, subs = self.load_features(idx)
            ids = np.array(ids)

            if speech:

                if idx == self.__len__() - 1:
                    self.on_epoch_end()

                bs = self.batch_size

                # return a tuple of tensors with batch_size = bs
                # (ids, eeg, speech)
                idx_per_batch, subs_per_batch = self.create_eeg_indices(eeg)

                for i in range(0, len(eeg)):
                    # we loop over all the subjects in this story

                    # get the batch index
                    idx = idx_per_batch[i,: ] # shoudl be [bs]
                    eeg_batch = eeg[subs_per_batch[i], idx, :, :] # should be [bs, wl, 64]
                    # get rid of first dim
                    eeg_batch = np.squeeze(eeg_batch)
                    eeg_batch = self.eeg_augmentation(eeg_batch)

                    speech_batch = [feature[idx, :,:] for feature in speech] # should be [bs, wl,1024] or whatever the speech_dimension is as last dim
                    ids_batch = ids[idx]
                    yield eeg_batch, speech_batch, ids_batch, np.array(subs)[subs_per_batch[i]]

    def eeg_augmentation(self, eeg_batch):
        # apply data augmentation to eeg
        # eeg_batch is a tensor with shape [bs, wl, 64]
        # apply data augmentation with probability data_augmentation_probability
        # return the augmented eeg_batch

        # apply data augmentation
        for aug in self.data_augmentation:
            eeg_batch = aug(eeg_batch)

        return eeg_batch

    def create_eeg_indices(self, eeg):
        # helper function, which
        # creates an array with the indices of the eeg data
        # eeg is a list of arrays with shape [n_subs, len_batch, wl, 64]
        # returns an array with shape [n_subs, bs,
        # 1) len_batch is > bs, we have to choose bs random indices
        # 2) if shuffle_subs = True, we have to shuffle the indices such that different subs are chosen each batch
        n_subs = eeg.shape[0]
        len_batch = eeg.shape[1]
        bs = self.batch_size
        shuffle_subs = self.shuffle_subs
        speech_idx= []
        # for subs idx, create an array with dim [nsub, bs], where the nsub = 0...n_subs
        subs_idx = np.arange(n_subs)
        for i in range(n_subs):
            # create random indices
            idx = np.random.choice(len_batch, bs, replace=False)
            speech_idx.append(idx)

        subs_per_batch = np.repeat(range(n_subs), bs).reshape(n_subs, bs)
        if shuffle_subs:
            # shuffle but in the other direction
            rng = np.random.default_rng()
            half_len = int(bs*self.shuffle_percentage)
            subs_per_batch = np.concatenate([rng.permuted(subs_per_batch[:, :half_len], axis=0), subs_per_batch[:,half_len:]], axis=1)
            # subs_per_batch = rng.permuted(subs_per_batch[0:], axis=0)

        speech_idx = np.array(speech_idx)
        return speech_idx, subs_per_batch

    def __iter__(self):
        return iter(self.get_item())

    def on_epoch_end(self):
        """Change state at the end of an epoch."""
        np.random.shuffle(self.batches_keys)


    def split_into_windows(self, feature):
        # split data into windows of length window_length with overlap window_overlap
        split_data = np.split(feature, range(self.window_length, len(feature), self.window_length), axis=0)[:-1]
        return np.stack(split_data, axis=0)









