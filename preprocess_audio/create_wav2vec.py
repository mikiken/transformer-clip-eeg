import numpy as np
import os
import torch
import librosa
import glob

import scipy.signal
import gzip
import shutil

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "nl"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"
SAMPLES = 10
sr = 16000
import scipy.signal as sps

final_samplingate_hz = 64 # from orginial 50Hz
dataset_root = os.path.join(os.path.dirname(__file__), '..', '..', 'auditory-eeg-dataset', 'downloads')
wav2vec_layers_to_extract = list(range(19, 20, 1))
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

save_dir = os.path.join(dataset_root, 'derivatives', 'preprocessed_stimuli')


def get_hidden_output_single(audio_path, layer=15, overlap=10, segment_length=60):
    print(f'getting latent representations for: {audio_path}')
    speech_array, sampling_rate = librosa.load(audio_paths, sr=sr)
    speech_array = np.expand_dims(speech_array, 0)

    segment_length = segment_length * sr
    wav_length = np.size(speech_array)

    # pad zeros before the start
    speech_array2 = np.concatenate([np.zeros((1, int(overlap / 2) * sr), dtype=np.float32), speech_array], axis=1)
    end_of_file = False
    outputs = []
    for i in range(int(wav_length / segment_length) + 1):
        start = 0 + i * segment_length
        end = start + segment_length + overlap * sr

        if end < np.size(speech_array2):
            spch_seg = speech_array2[:, start:end]
            print(f'segment {i + 1}')
            print('end in term of seconds: ', end / sr)
        else:
            print('last segment of the file')
            spch_seg = speech_array2[:, start:]
            end_of_file = True

        input = torch.tensor(spch_seg)
        with torch.no_grad():
            logits = model.base_model(input, attention_mask=torch.ones_like(input), output_hidden_states=True)
        out = logits['hidden_states'][layer]
        out = out.numpy()
        out = np.squeeze(out)
        if end_of_file:
            out = out[int(overlap / 2) * 50:]
        else:
            # +1 is used to account for shortening of the output length (2999 instead of 3000)
            out = out[int(overlap / 2) * 50: -int(overlap / 2) * 50 + 1, :]
        outputs.append(out)
    return np.vstack(outputs)


def get_hidden_output(audio_path, layers=list(range(15, 18, 1)), overlap=10, segment_length=60):
    print(f'getting latent representations for: {audio_path}')
    # load the audio
    #$ check if it is a wav file
    if  audio_path.endswith('.wav'):
        speech_array, sampling_rate = librosa.load(audio_path, sr=sr)
    else:
        # it is a npz file
        speech_data = dict(np.load(audio_path))
        speech_array = speech_data['audio']
        sampling_rate = speech_data['fs']
        # do resampling from sampling_rate to sr
        speech_array = scipy.signal.resample_poly(speech_array, sr, sampling_rate)

    speech_array = np.expand_dims(speech_array, 0)

    segment_length = segment_length * sr
    wav_length = np.size(speech_array)

    # pad zeros before the start
    speech_array2 = np.concatenate([np.zeros((1, int(overlap / 2) * sr), dtype=np.float32), speech_array], axis=1)
    end_of_file = False
    outputs = {}
    for layer in layers:
        outputs[layer] = []

    for i in range(int(wav_length / segment_length) + 1):
        start = 0 + i * segment_length
        end = start + segment_length + overlap * sr

        if end < np.size(speech_array2):
            spch_seg = speech_array2[:, start:end]
            print(f'segment {i + 1}')
            print('end in term of seconds: ', end / sr)
        else:
            print('last segment of the file')
            spch_seg = speech_array2[:, start:]
            end_of_file = True

        input = torch.tensor(spch_seg)
        with torch.no_grad():
            logits = model.base_model(input, attention_mask=torch.ones_like(input), output_hidden_states=True)

        for layer in layers:
            out = logits['hidden_states'][layer]
            out = out.numpy()
            out = np.squeeze(out)
            if end_of_file:
                out = out[int(overlap / 2) * 50:]
            else:
                # +1 is used to account for shortening of the output length (2999 instead of 3000)
                out = out[int(overlap / 2) * 50: -int(overlap / 2) * 50 + 1, :]
            outputs[layer].append(out)
    for key, value in outputs.items():
        outputs[key] = np.vstack(value)
    return outputs



# now do the npz files
audio_dir = os.path.join(dataset_root, 'stimuli', 'eeg')
audio_paths = sorted(glob.glob(os.path.join(audio_dir, "*.npz.gz")), reverse=True)
#filter out the noise and trigger paths
audio_paths = [x for x in audio_paths if not (os.path.basename(x).startswith('noise_') or os.path.basename(x).startswith('t_'))]
for path in audio_paths:
    story = os.path.basename(path).split('.')[0]
    print('Processing ', story)
    # check if already unzipped
    unzipped_name = '.'.join(path.split('.')[:-2])
    if not os.path.exists(unzipped_name):
        # unzip first
        with gzip.open(path, 'rb') as f_in:
            with open(unzipped_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if not os.path.exists(os.path.join(save_dir, f'{story}_-_wav2vec_{wav2vec_layers_to_extract[0]}.npy')):
        pkl_dict = get_hidden_output(path, layers=wav2vec_layers_to_extract, overlap=2, segment_length=8)

        for layer, value in pkl_dict.items():
            # Resample data
            number_of_samples = round(np.size(value, axis=0) * float(final_samplingate_hz) / 50)
            value = sps.resample(value, number_of_samples)

            # save layer per layer
            save_name = os.path.join(save_dir, f'{story}_-_wav2vec_{layer}.npy')
            np.save(value, save_name)

    print('done')























# predicted_ids = torch.argmax(logits, dim=-1)
# predicted_sentences = processor.batch_decode(predicted_ids)
#
# for i, predicted_sentence in enumerate(predicted_sentences):
#     print("-" * 100)
#     print("Reference:", test_dataset[i]["sentence"])
#     print("Prediction:", predicted_sentence)



class Wav2vecJalil(Wav2Vec2ForCTC):
    def forward_new(self, input_values):
        # write your function which will return output of a specific layer
        return input_values