from concurrent.futures import process
import os
import torch
import julius
import torchaudio
from torch.utils.data import Dataset
import random
import librosa
import numpy as np


# class twod_dataset(Dataset):
#     def __init__(self, process_config, train_config):
#         self.dataset_name = train_config["dataset"]
#         self.dataset_path = train_config["path"]["raw_path"]
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         self.wavs = self.process_meta()
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         audio_name = self.wavs[idx]
#         wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#         wav = wav[:,:self.max_len]
#         patch_num = wav.shape[1] // self.win_len
#         pad_num = (patch_num + 1)*self.win_len - wav.shape[1]
#         if pad_num == self.win_len:
#             pad_num = 0
#             wav_matrix = wav.reshape(-1, self.win_len)
#         else:
#             wav_matrix = torch.cat((wav,torch.zeros(1,pad_num)), dim=1).reshape(-1, self.win_len)
#         sample = {
#             "matrix": wav_matrix,
#             "sample_rate": sr,
#             "patch_num": patch_num,
#             "pad_num": pad_num,
#             "name": audio_name
#         }
#         return sample
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs
#
#
# class oned_dataset(Dataset):
#     def __init__(self, process_config, train_config):
#         self.dataset_name = train_config["dataset"]
#         self.dataset_path = train_config["path"]["raw_path"]
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         self.wavs = self.process_meta()
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         audio_name = self.wavs[idx]
#         wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#         wav = wav[:,:self.max_len]
#         sample = {
#             "matrix": wav,
#             "sample_rate": sr,
#             "patch_num": 0,
#             "pad_num": 0,
#             "name": audio_name
#         }
#         return sample
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs
#
#
# # pre-load dataset and resample to 22.05KHz
# class mel_dataset(Dataset):
#     def __init__(self, process_config, train_config):
#         self.dataset_name = train_config["dataset"]
#         self.dataset_path = train_config["path"]["raw_path"]
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         self.wavs = self.process_meta()
#         # n_fft = process_config["mel"]["n_fft"]
#         # hop_length = process_config["mel"]["hop_length"]
#         # self.stft = STFT(n_fft, hop_length)
#
#         sr = process_config["audio"]["or_sample_rate"]
#         self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
#         self.sample_list = []
#         for idx in range(len(self.wavs)):
#             audio_name = self.wavs[idx]
#             # import pdb
#             # pdb.set_trace()
#             wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#             if wav.shape[1] > self.max_len:
#                 cuted_len = random.randint(5*sr, self.max_len)
#                 wav = wav[:, :cuted_len]
#             wav = self.resample(wav[0,:].view(1,-1))
#             # wav = wav[:,:self.max_len]
#             sample = {
#                 "matrix": wav,
#                 "sample_rate": sr,
#                 "patch_num": 0,
#                 "pad_num": 0,
#                 "name": audio_name
#             }
#             self.sample_list.append(sample)
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         return self.sample_list[idx]
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs
#
#
#
# class mel_dataset_test(Dataset):
#     def __init__(self, process_config, train_config):
#         self.dataset_name = train_config["dataset"]
#         self.dataset_path = train_config["path"]["raw_path_test"]
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         self.wavs = self.process_meta()
#         self.resample = julius.ResampleFrac(22050, 16000)
#         # n_fft = process_config["mel"]["n_fft"]
#         # hop_length = process_config["mel"]["hop_length"]
#         # self.stft = STFT(n_fft, hop_length)
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         audio_name = self.wavs[idx]
#         wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#         # wav = self.resample(wav)
#         # wav = wav[:,:self.max_len]
#         # spect, phase = self.stft.transform(wav)
#         sample = {
#             "matrix": wav,
#             "sample_rate": sr,
#             "patch_num": 0,
#             "pad_num": 0,
#             "name": audio_name
#         }
#         return sample
#
#     def process_meta(self):
#         # wavs = os.listdir(self.dataset_path)
#         # return wavs
#         wav_files = []
#         for filename in os.listdir(self.dataset_path):
#             if filename.endswith('.wav'):
#                 wav_files.append(filename)
#         return wav_files
#
#
#
# class mel_dataset_test_2(Dataset):
#     def __init__(self, process_config, train_config):
#         self.dataset_name = train_config["dataset"]
#         self.dataset_path = train_config["path"]["raw_path_test"]
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         self.wavs = self.process_meta()
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         audio_name = self.wavs[idx]
#         wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#         wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
#         wav = torch.Tensor(wav).unsqueeze(0)
#         # wav = self.resample(wav)
#         # wav = wav[:,:self.max_len]
#         # spect, phase = self.stft.transform(wav)
#         sample = {
#             "matrix": wav,
#             "sample_rate": sr,
#             "patch_num": 0,
#             "pad_num": 0,
#             "name": audio_name
#         }
#         return sample
#
#     def process_meta(self):
#         # wavs = os.listdir(self.dataset_path)
#         # return wavs
#         wav_files = []
#         for filename in os.listdir(self.dataset_path):
#             if filename.endswith('.wav'):
#                 wav_files.append(filename)
#         return wav_files


# # pre-load dataset and resample to 22.05KHz
# class wav_dataset(Dataset):
#     def __init__(self, process_config, train_config, flag='train'):
#         self.dataset_name = train_config["dataset"]
#         raw_dataset_path = train_config["path"]["raw_path"]
#         self.dataset_path = os.path.join(raw_dataset_path, flag)
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         # self.wavs = self.process_meta()[:10]
#         self.wavs = self.process_meta()
#         # n_fft = process_config["mel"]["n_fft"]
#         # hop_length = process_config["mel"]["hop_length"]
#         # self.stft = STFT(n_fft, hop_length)
#
#         sr = process_config["audio"]["or_sample_rate"]
#         self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
#         self.sample_list = []
#         for idx in range(len(self.wavs)):
#             audio_name = self.wavs[idx]
#             wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#             if wav.shape[1] > self.max_len:
#                 cuted_len = random.randint(5*sr, self.max_len)
#                 wav = wav[:, :cuted_len]
#             wav = self.resample(wav[0,:].view(1,-1))
#             # wav = wav[:,:self.max_len]
#             sample = {
#                 "matrix": wav,
#                 "sample_rate": sr,
#                 "patch_num": 0,
#                 "pad_num": 0,
#                 "name": audio_name
#             }
#             self.sample_list.append(sample)
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         return self.sample_list[idx]
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs
#
#
#
# class wav_dataset_wopreload(Dataset):
#     def __init__(self, process_config, train_config, flag='train'):
#         self.dataset_name = train_config["dataset"]
#         raw_dataset_path = train_config["path"]["raw_path"]
#         self.dataset_path = os.path.join(raw_dataset_path, flag)
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         # self.wavs = self.process_meta()[:10]
#         self.wavs = self.process_meta()
#         # n_fft = process_config["mel"]["n_fft"]
#         # hop_length = process_config["mel"]["hop_length"]
#         # self.stft = STFT(n_fft, hop_length)
#
#         sr = process_config["audio"]["or_sample_rate"]
#         self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         audio_name = self.wavs[idx]
#         # import pdb
#         # pdb.set_trace()
#         wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#         if wav.shape[1] > self.max_len:
#             cuted_len = random.randint(5*sr, self.max_len)
#             wav = wav[:, :cuted_len]
#         wav = self.resample(wav[0,:].view(1,-1))
#         # wav = wav[:,:self.max_len]
#         sample = {
#             "matrix": wav,
#             "sample_rate": sr,
#             "patch_num": 0,
#             "pad_num": 0,
#             "name": audio_name
#         }
#         return sample
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs
#
#
# class wav_dataset_test(Dataset):
#     def __init__(self, process_config, train_config, flag='train'):
#         self.dataset_name = train_config["dataset"]
#         raw_dataset_path = train_config["path"]["raw_path"]
#         self.dataset_path = os.path.join(raw_dataset_path, flag)
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         # self.wavs = self.process_meta()[:2]
#         self.wavs = self.process_meta()
#         # n_fft = process_config["mel"]["n_fft"]
#         # hop_length = process_config["mel"]["hop_length"]
#         # self.stft = STFT(n_fft, hop_length)
#
#         sr = process_config["audio"]["or_sample_rate"]
#         self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
#         self.sample_list = []
#         for idx in range(len(self.wavs)):
#             audio_name = self.wavs[idx]
#             # import pdb
#             # pdb.set_trace()
#             wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#             # if wav.shape[1] > self.max_len:
#             #     cuted_len = random.randint(5*sr, self.max_len)
#             #     wav = wav[:, :cuted_len]
#             wav = self.resample(wav[0,:].view(1,-1))
#             # wav = wav[:,:self.max_len]
#             sample = {
#                 "matrix": wav,
#                 "sample_rate": sr,
#                 "patch_num": 0,
#                 "pad_num": 0,
#                 "name": audio_name
#             }
#             self.sample_list.append(sample)
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         return self.sample_list[idx]
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs
#
#
#
#
# class wav_dataset_librosa(Dataset):
#     def __init__(self, process_config, train_config, flag='train'):
#         self.dataset_name = train_config["dataset"]
#         raw_dataset_path = train_config["path"]["raw_path"]
#         self.dataset_path = os.path.join(raw_dataset_path, flag)
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         # self.wavs = self.process_meta()[:10]
#         self.wavs = self.process_meta()
#         # n_fft = process_config["mel"]["n_fft"]
#         # hop_length = process_config["mel"]["hop_length"]
#         # self.stft = STFT(n_fft, hop_length)
#
#         sr = process_config["audio"]["or_sample_rate"]
#         # self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
#         self.sample_list = []
#         for idx in range(len(self.wavs)):
#             audio_name = self.wavs[idx]
#             # import pdb
#             # pdb.set_trace()
#             # wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#             wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
#             wav = torch.Tensor(wav).unsqueeze(0)
#             if wav.shape[1] > self.max_len:
#                 cuted_len = random.randint(5*sr, self.max_len)
#                 wav = wav[:, :cuted_len]
#             # wav = self.resample(wav[0,:].view(1,-1))
#             # wav = wav[:,:self.max_len]
#             sample = {
#                 "matrix": wav,
#                 "sample_rate": sr,
#                 "patch_num": 0,
#                 "pad_num": 0,
#                 "name": audio_name
#             }
#             self.sample_list.append(sample)
#
#     def __len__(self):
#         return len(self.wavs)
#
#     def __getitem__(self, idx):
#         return self.sample_list[idx]
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs

def collate_fn(batch):
    # Find the maximum length for padding
    max_len = max(sample["matrix"].shape[0] for sample in batch)

    # Prepare one big padded tensor [batch_size, max_len]
    batch_size = len(batch)
    padded_matrix = torch.zeros(batch_size, max_len, dtype=batch[0]["matrix"].dtype)

    # Also collect other fields
    sample_rates = []
    names = []
    patch_nums = []
    pad_nums = []

    for i, sample in enumerate(batch):
        length = sample["matrix"].shape[0]
        padded_matrix[i, :length] = sample["matrix"]
        pad_size = max_len - length

        sample_rates.append(sample["sample_rate"])
        names.append(sample["name"])
        patch_nums.append(sample["patch_num"])
        pad_nums.append(pad_size)

    batched_sample = {
        "matrix": padded_matrix,  # shape: [B, max_len]
        "sample_rate": torch.tensor(sample_rates, dtype=torch.int64),
        "name": names,
        "patch_num": torch.tensor(patch_nums, dtype=torch.int64),
        "pad_num": torch.tensor(pad_nums, dtype=torch.int64),
    }

    return batched_sample


class WavDataset(Dataset):
    def __init__(self, process_config, train_config, flag="train"):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.flag = flag
        self.dataset_path = os.path.join(raw_dataset_path, flag)

        self.sample_rate = process_config["audio"]["sample_rate"]
        self.original_sample_rate = process_config["audio"]["or_sample_rate"]
        self.resample_needed = self.original_sample_rate != self.sample_rate

        if self.resample_needed:
            self.resample = torchaudio.transforms.Resample(
                self.original_sample_rate, self.sample_rate
            )

        self.future_amt_waveform = train_config["watermark"]["future_amt_waveform"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.data_percentage = process_config["audio"]["data_percentage"]

        # For example, 2.05s * 16000 = 32800 frames
        # plus future_amt_waveform*2 padding
        self.min_length = 32800 + self.future_amt_waveform * 2

        # Filter out short files
        self.wavs = self._filter_wavs()

        # If data_divider > 1, subsample the dataset
        if train_config["iter"]["data_divider"] > 1:
            random.seed(42)
            subset_size = len(self.wavs) // train_config["iter"]["data_divider"]
            self.wavs = random.sample(self.wavs, subset_size)

    def _filter_wavs(self):
        """
        Returns a list of valid .wav file paths that are longer than `min_length`
        *after* applying data_percentage factor.
        """
        all_files = os.listdir(self.dataset_path)
        valid_wavs = []
        for audio_name in all_files:
            audio_path = os.path.join(self.dataset_path, audio_name)
            if not audio_name.lower().endswith(".wav"):
                continue  # skip non-wav if present
            try:
                info = torchaudio.info(audio_path)
            except Exception as e:
                print(f"Could not read info for {audio_name}: {e}")
                continue

            # If data_percentage < 1.0, use that fraction to check minimal length
            chunk_frames = int(info.num_frames * self.data_percentage)
            if chunk_frames >= self.min_length:
                valid_wavs.append(audio_name)
            else:
                # Avoid spamming prints if many files are short
                # print(f"{audio_name} is too short.")
                pass
        return valid_wavs

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        """
        1. Load waveform
        2. Possibly resample
        3. Random cut to max_len
        4. Select random chunk via data_percentage
        5. Compute patch_num, pad_num AFTER final shape is decided
        """
        audio_name = self.wavs[idx]
        audio_path = os.path.join(self.dataset_path, audio_name)

        wav, sr = torchaudio.load(audio_path)  # shape: (channels, num_frames)

        wav = wav[0]  # now shape is [num_frames]

        # Resample if needed
        if self.resample_needed and sr != self.sample_rate:
            wav = self.resample(wav)
            sr = self.sample_rate

        # (Optional) random cut down to max_len
        if wav.shape[0] > self.max_len:
            cut_len = random.randint(5 * sr, self.max_len)
            cut_len = min(cut_len, wav.shape[0])
            wav = wav[:cut_len]

        # Then select random chunk (still shape = [L])
        wav = select_random_chunk(wav.unsqueeze(0), self.data_percentage).squeeze(0)

        # Now wav is shape [L], patch_num & pad_num can be calculated
        length = wav.shape[0]
        patch_num = length // self.win_len
        pad_num = (patch_num + 1) * self.win_len - length

        sample = {
            "matrix": wav,  # 1D tensor [L]
            "sample_rate": sr,
            "patch_num": patch_num,
            "pad_num": pad_num,
            "name": audio_name,
        }
        return sample


def select_random_chunk(audio_data, percentage=1.0):
    batch_size, total_length = audio_data.shape

    # Determine the length of the chunk to be selected
    chunk_length = int(total_length * percentage)

    # Randomly select start points for the batch
    start_points = torch.randint(
        0, total_length - chunk_length + 1, (batch_size,), device=audio_data.device
    )

    # Compute end points
    end_points = start_points + chunk_length

    # Create indices for the chunks
    offsets = torch.arange(chunk_length, device=audio_data.device).unsqueeze(
        0
    )  # (1, chunk_length)
    indices = start_points.unsqueeze(1) + offsets  # (batch_size, chunk_length)

    # Extract selected audio in a vectorized way
    selected_audio_wav = audio_data.gather(1, indices)

    return selected_audio_wav