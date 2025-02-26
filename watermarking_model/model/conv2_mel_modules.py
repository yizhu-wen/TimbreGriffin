from base64 import encode
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from .blocks import FCBlock, Conv2Encoder, WatermarkEmbedder, WatermarkExtracter, ReluBlock
from distortions.frequency2 import fixed_STFT
# from distortions.dl import distortion


# Optional: set up a small constant
EPS = 1e-9


def save_spectrum(spect, phase, flag='linear'):
    import numpy as np
    import os
    import librosa
    import librosa.display
    root = "draw_figure"
    import matplotlib.pyplot as plt
    spect = spect/torch.max(torch.abs(spect))
    spec = librosa.amplitude_to_db(spect.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)
    phase = phase/torch.max(torch.abs(phase))
    spec = librosa.amplitude_to_db(phase.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.clim(-40, 40)
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_phase_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)

def save_feature_map(feature_maps):
    import os
    import matplotlib.pyplot as plt
    import librosa
    import numpy as np
    import librosa.display
    feature_maps = feature_maps.cpu().numpy()
    root = "draw_figure"
    output_folder = os.path.join(root,"feature_map_or")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_channels = feature_maps.shape[0]
    for channel_idx in range(n_channels):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[channel_idx, :, :], cmap='gray')
        ax.axis('off')
        output_file = os.path.join(output_folder, f'feature_map_channel_{channel_idx + 1}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

def save_waveform(a_tensor, flag='original'):
    import os
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile
    root = "draw_figure"
    y = a_tensor.detach().cpu().numpy()
    soundfile.write(os.path.join(root, flag + "_waveform.wav"), y, samplerate=16000)
    # D = librosa.stft(y)
    # spectrogram = np.abs(D)
    # img=librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=22050, x_axis='time', y_axis='log', y_coords=None);
    # plt.axis('off')
    # plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram_from_waveform.png'), bbox_inches='tight', pad_inches=0.0)



# def get_vocoder(device):
#     with open("hifigan/config.json", "r") as f:
#         config = json.load(f)
#     config = hifigan.AttrDict(config)
#     vocoder = hifigan.Generator(config)
#     ckpt = torch.load("./hifigan/model/VCTK_V1/generator_v1")
#     vocoder.load_state_dict(ckpt["generator"])
#     vocoder.eval()
#     vocoder.remove_weight_norm()
#     vocoder.to(device)
#     freeze_model_and_submodules(vocoder)
#     return vocoder

# def freeze_model_and_submodules(model):
#     for param in model.parameters():
#         param.requires_grad = False
#
#     for module in model.children():
#         if isinstance(module, nn.Module):
#             freeze_model_and_submodules(module)


class Encoder(nn.Module):
    def __init__(self, process_config, model_config, train_config, msg_length):
        super(Encoder, self).__init__()
        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 3
        self.layers_EM = model_config["conv2"]["layers_EM"]
        self.future_amt = train_config["watermark"]["future_amt"]
        self.future = True
        self.power = 1.0

        self.vocoder_step = model_config["structure"]["vocoder_step"]
        #MLP for the input wm
        self.msg_linear_in = FCBlock(msg_length, win_dim, activation=LeakyReLU(inplace=False))

        #stft transform
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

        self.ENc = Conv2Encoder(input_channel=2, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_CE)

        self.EM = WatermarkEmbedder(input_channel=self.EM_input_dim, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_EM)

    def pad_w_zero_stft(self, input_stft, watermark_stft):
        """
        Pad the watermarked stft output with zeros on the left + right,
        respecting future_amt and chunk-based offsets.
        """
        B, C, F, T_in = input_stft.shape
        _, c, _, t_in = watermark_stft.shape
        T_wm = watermark_stft.shape[3]
        chunk_size = 206 if not self.future else (206 + self.future_amt)

        zeros_right_len = T_in - T_wm - chunk_size
        if zeros_right_len < 0:
            # Edge case: won't happen if chunking logic is correct, but just to be safe
            zeros_right_len = 0

        zeros_left = torch.zeros(B, c, F, chunk_size, device=watermark_stft.device)
        zeros_right = torch.zeros(
            B, c, F, zeros_right_len, device=watermark_stft.device
        )
        actual_watermark = torch.cat([zeros_left, watermark_stft, zeros_right], dim=3)
        return actual_watermark + EPS  # small offset

    def forward(self, x, msg, global_step):
        num_samples = x.shape[-1]
        spect, _, stft_result = self.stft.transform(x)
        B, freq_bin, time_frames = spect.shape  # [B, freq_bin, time_frames]
        spect = spect.unsqueeze(1)
        # Evaluate how many chunks we can process
        # 2s input + 0.05s calculation delay
        # 2.05*16000 = 32800
        # 32800 // hop_length + 1 = 206 center=True
        voice_prefilling = 206
        # Predict future 0.5s watermark
        # 0.5*16000 = 8000
        # 8000 // hop_length + 1 =51
        max_start = time_frames - (voice_prefilling + self.future_amt)
        if int(max_start / self.future_amt+1) <= 0:
            return None  # Not enough frames for a chunk

        list_of_watermarks = []
        for i in range(int((time_frames - (voice_prefilling + self.future_amt))/(self.future_amt+1))):
            carrier_encoded = self.ENc(stft_result[:, :, :, i*(self.future_amt+1):voice_prefilling + i*(self.future_amt+1)])
            # torch.Size([B, 1, 161])
            # torch.Size([B, 161, 1])
            # torch.Size([B, 1, 161, 1])
            # torch.Size([B, 1, 161, 206])
            watermark_encoded = self.msg_linear_in(msg).transpose(1, 2).unsqueeze(1).repeat(1, 1, 1,
                                                                                            carrier_encoded.shape[3])

            concatenated_feature = torch.cat((carrier_encoded, stft_result[:, :, :,
                                                               i*(self.future_amt+1):voice_prefilling + i*(self.future_amt+1)], watermark_encoded), dim=1)
            # [B, 2, bins, length]
            # Embed the watermark
            carrier_watermarked = self.EM(concatenated_feature)
            # Append both the watermark chunk and the pilot segment
            list_of_watermarks.append(carrier_watermarked)

        if len(list_of_watermarks) > 0:
            watermark = torch.cat(list_of_watermarks, dim=-1)
            all_watermark_stft = self.pad_w_zero_stft(
                spect, watermark
            )
            mask=spect!=0

            # scalar = 1.0  # Adjust if necessary
            # all_watermark_stft = self.power * torch.unsqueeze(scalar.cuda(), dim=1) * all_watermark_stft
            all_watermark_stft = all_watermark_stft*mask + 0.0000001

            self.stft.num_samples = num_samples

            # Compute the magnitude (spect)
            spect = torch.sqrt(all_watermark_stft[:, 0, :, :] ** 2 + all_watermark_stft[:, 1, :, :] ** 2)

            # Compute the phase using arctan2
            phase = torch.atan2(all_watermark_stft[:, 1, :, :], all_watermark_stft[:, 0, :, :])

            y = self.stft.inverse(spect, phase).squeeze(1)

            return y, all_watermark_stft
        else:
            print("Not enough watermarking!!!!")
            return None

    
    # def save_forward(self, x, msg):
    #     num_samples = x.shape[2]
    #     save_waveform(x.squeeze())
    #     spect, phase = self.stft.transform(x)
    #     # save spectrum
    #     save_spectrum(spect, phase, 'linear')
    #
    #     carrier_encoded = self.ENc(spect.unsqueeze(1))
    #     # save feature_map
    #     # save_feature_map(carrier_encoded[0])
    #     watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
    #     concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)
    #     carrier_wateramrked = self.EM(concatenated_feature)
    #     save_spectrum(carrier_wateramrked.squeeze(1), phase, 'wmed_linear')
    #
    #     self.stft.num_samples = num_samples
    #     y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
    #     save_waveform(y.squeeze().squeeze(), 'wmed')
    #     return y, carrier_wateramrked



class Decoder(nn.Module):
    def __init__(self, process_config, model_config, train_config, msg_length):
        super(Decoder, self).__init__()
        self.robust = model_config["robust"]
        # if self.robust:
        #     self.dl = distortion()

        # self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
        # self.vocoder = get_vocoder(device)
        # self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.hop_length = process_config["mel"]["hop_length"]
        self.block = model_config["conv2"]["block"]
        self.EX = WatermarkExtracter(input_channel=2, hidden_dim=model_config["conv2"]["hidden_dim"], block=self.block)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        self.msg_linear_out = FCBlock(win_dim, msg_length)

    def forward(self, y, global_step):
        y_identity = y.clone()
        y_d = y
        # if global_step > self.vocoder_step:
        #     y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
        #     # y = self.vocoder(y_mel)
        #     y_d = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        # else:
        #     y_d = y
        if self.robust:
            y_d_d = self.dl(y_d, self.robust)
        else:
            y_d_d = y_d
        _, _, stft_result = self.stft.transform(y_d_d)
        extracted_wm = self.EX(stft_result).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)

        _, _, stft_result_identity = self.stft.transform(y_identity)
        extracted_wm_identity = self.EX(stft_result_identity).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        msg_identity = self.msg_linear_out(msg_identity)
        return msg, msg_identity
    
    def test_forward(self, y):
        spect, phase, stft_result= self.stft.transform(y)
        extracted_wm = self.EX(stft_result).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg
    
    # def save_forward(self, y):
    #     # save mel_spectrum
    #     y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
    #     save_spectrum(y_mel, y_mel, 'mel')
    #     y, reconstruct_spec = self.mel_transform.griffin_lim(magnitudes=y_mel)
    #     y = y.unsqueeze(1)
    #     save_waveform(y.squeeze().squeeze(), 'distored')
    #     save_spectrum(reconstruct_spec, reconstruct_spec, 'recon')
    #     # y = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
    #     spect, phase = self.stft.transform(y)
    #     save_spectrum(spect, spect, 'distored')
    #     pdb.set_trace()
    #     extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
    #     msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
    #     msg = self.msg_linear_out(msg)
    #     return msg
    
    # def mel_test_forward(self, spect):
    #     extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
    #     msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
    #     msg = self.msg_linear_out(msg)
    #     return msg
        


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                ReluBlock(2,16,3,1,1),
                ReluBlock(16,32,3,1,1),
                ReluBlock(32,64,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Linear(64,1)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

    def forward(self, x):
        _, _, stft_result = self.stft.transform(x)
        x = self.conv(stft_result)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x


# def get_param_num(model):
#     num_param = sum(param.numel() for param in model.parameters())
#     return num_param