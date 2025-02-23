import math
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from optispeech.utils import denormalize, sequence_mask
from optispeech.utils.segments import get_segments, get_random_segments

from .modules.alignments import average_by_duration, maximum_path
from .loss import FastSpeech2Loss, duration_loss as f_duration_loss



class OptiSpeechGenerator(nn.Module):
    def __init__(
        self,
        dim: int,
        segment_size,
        text_embedding,
        encoder,
        duration_predictor,
        pitch_predictor,
        energy_predictor,
        decoder,
        vocoder,
        loss_coeffs,
        feature_extractor,
        num_speakers,
        num_languages,
        data_statistics,
        **kwargs
    ):
        super().__init__()

        self.segment_size = segment_size
        self.loss_coeffs = loss_coeffs
        self.n_feats = feature_extractor.n_feats
        self.n_fft = feature_extractor.n_fft
        self.hop_length = feature_extractor.hop_length
        self.sample_rate = feature_extractor.sample_rate
        self.data_statistics = data_statistics
        self.num_speakers = num_speakers
        self.num_languages = num_languages

        self.text_embedding = text_embedding(dim=dim)
        self.encoder = encoder(dim=dim)
        self.duration_predictor = duration_predictor(dim=dim)
        self.pitch_predictor = pitch_predictor(dim=dim)
        self.energy_predictor = energy_predictor(dim=dim)
        self.decoder = decoder(dim=dim)
        self.vocoder = vocoder(
            input_channels=dim,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        if self.num_speakers > 1:
            self.sid_embed = torch.nn.Embedding(self.num_speakers, dim)
        if self.num_languages > 1:
            self.lid_embed = torch.nn.Embedding(self.num_languages, dim)
        self.loss_criterion = FastSpeech2Loss()

    def forward(self, x, x_lengths, mel, mel_lengths, pitches, energies, sids, lids):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            pitches (torch.Tensor): phoneme-level pitch values.
                shape: (batch_size, max_text_length)
            energies (torch.Tensor): phoneme-level energy values.
                shape: (batch_size, max_text_length)
            sids (torch.LongTensor): list of speaker IDs for each input sentence.
                shape: (batch_size,)
            lids (torch.LongTensor): list of language IDs for each input sentence.
                shape: (batch_size,)

        Returns:
            loss: (torch.Tensor): scaler representing total loss
            duration_loss: (torch.Tensor): scaler representing durations loss
            pitch_loss: (torch.Tensor): scaler representing pitch loss
            energy_loss: (torch.Tensor): scaler representing energy loss
        """
        f0_real = pitches
        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).type_as(x)

        mel_max_length = mel_lengths.max()
        mel_mask = torch.unsqueeze(sequence_mask(mel_lengths, mel_max_length), 1).type_as(x)

        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)
        target_padding_mask = ~mel_mask.squeeze(1).bool().to(x.device)

        # text embedding
        x, __ = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, input_padding_mask)

        # Speaker and language embedding
        if sids is not None:
            sid_emb = self.sid_embed(sids.view(-1))
            x = x + sid_emb.unsqueeze(1)
        if lids is not None:
            lid_embs = self.lid_embed(lids.view(-1))
            x = x + lid_embs.unsqueeze(1)

        # alignment
        duration_hat = self.duration_predictor(x.detach(), input_padding_mask)
        attn_mask = x_mask.unsqueeze(-1) * mel_mask.unsqueeze(2)
        with torch.no_grad():
            # negative cross-entropy
            mu_x = x.clone().transpose(1, 2)
            s_p_sq_r = torch.ones_like(mu_x) # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi)- torch.zeros_like(mu_x), [1], keepdim=True
            )
            neg_cent2 = torch.einsum("bdt, bds -> bts", -0.5 * (mel**2), s_p_sq_r)
            neg_cent3 = torch.einsum("bdt, bds -> bts", mel, (mu_x * s_p_sq_r))
            neg_cent4 = torch.sum(
                -0.5 * (mu_x**2) * s_p_sq_r, [1], keepdim=True
            )  
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(mel_mask, -1)
            attn = (
                maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
            )
        durations = torch.log(1e-8 + attn.sum(2)) * x_mask
        duration_loss = f_duration_loss(duration_hat, durations, x_lengths, use_log=False)
        durations = durations.squeeze(1).detach()

        # Average pitch and energy values based on durations
        pitches = average_by_duration(durations, pitches.unsqueeze(-1), x_lengths, mel_lengths)
        energies = average_by_duration(durations, energies.unsqueeze(-1), x_lengths, mel_lengths)

        # variance predictors
        x, pitch_hat = self.pitch_predictor(x, input_padding_mask, pitches)
        x, energy_hat = self.energy_predictor(x, input_padding_mask, energies)

        # upsample to mel lengths
        attn = attn.squeeze(1).transpose(1,2)
        y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        # y = y.transpose(1, 2)

        # Decoder
        y = self.decoder(y, target_padding_mask)

        # get random segments
        segment_size = min(self.segment_size, y.shape[-2])
        num_frames = mel_lengths - 4 # mel-centered
        segment, start_idx = get_random_segments(
            y.transpose(1, 2),
            num_frames.type_as(y),
            segment_size,
        )
        f0_cond = get_segments(
            f0_real.unsqueeze(1),
            start_idxs=start_idx,
            segment_size=segment_size
        )

        # Generate wav
        wav_hat = self.vocoder(segment.detach(), f0=f0_cond.detach())

        # Losses
        loss_coeffs = self.loss_coeffs
        pitch_loss, energy_loss = self.loss_criterion(
            p_outs=pitch_hat.unsqueeze(-1),
            e_outs=energy_hat.unsqueeze(-1),
            ps=pitches.unsqueeze(-1),
            es=energies.unsqueeze(-1),
            ilens=x_lengths,
        )
        loss = (
            (duration_loss * loss_coeffs.lambda_duration)
            + (pitch_loss * loss_coeffs.lambda_pitch)
            + (energy_loss * loss_coeffs.lambda_energy)
        )

        return {
            "wav_hat": wav_hat,
            "start_idx": start_idx,
            "segment_size": segment_size,
            "loss": loss,
            "duration_loss": duration_loss.detach().cpu(),
            "pitch_loss": pitch_loss.detach().cpu(),
            "energy_loss": energy_loss.detach().cpu(),
        }

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, sids=None, lids=None, d_factor=1.0, p_factor=1.0, e_factor=1.0):
        """
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            sids (Optional[torch.LongTensor]): list of speaker IDs for each input sentence.
                shape: (batch_size,)
            lids (Optional[torch.LongTensor]): list of language IDs for each input sentence.
                shape: (batch_size,)
            d_factor (Optional[float]): scaler to control phoneme durations.
            p_factor (Optional[float]): scaler to control pitch.
            e_factor (Optional[float]): scaler to control energy.

        Returns:
            wav (torch.Tensor): generated waveform
                shape: (batch_size, T)
            durations: (torch.Tensor): predicted phoneme durations
                shape: (batch_size, max_text_length)
            pitch: (torch.Tensor): predicted pitch
                shape: (batch_size, max_text_length)
            energy: (torch.Tensor): predicted energy
                shape: (batch_size, max_text_length)
            rtf: (float): total Realtime Factor (inference_t/audio_t)
        """
        am_t0 = perf_counter()

        x_max_length = x_lengths.max()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_max_length), 1).to(x.dtype)
        x_mask = x_mask.to(x.device)
        input_padding_mask = ~x_mask.squeeze(1).bool().to(x.device)

        # text embedding
        x, __ = self.text_embedding(x)

        # Encoder
        x = self.encoder(x, input_padding_mask)

        # Set default speaker/language during inference when not specified
        if (self.num_speakers > 1) and sids is None:
            sids = torch.zeros(x.shape[0]).long().to(x.device)
        if (self.num_languages > 1) and lids is None:
            lids = torch.zeros(x.shape[0]).long().to(x.device)

        # Speaker and language embedding
        if sids is not None:
            sid_emb = self.sid_embed(sids.view(-1))
            x = x + sid_emb.unsqueeze(1)
        if lids is not None:
            lid_embs = self.lid_embed(lids.view(-1))
            x = x + lid_embs.unsqueeze(1)

        # duration predictor
        durations = self.duration_predictor.infer(x, input_padding_mask, factor=d_factor)

        # variance predictors
        x, pitch = self.pitch_predictor.infer(x, input_padding_mask, p_factor)
        if self.energy_predictor is not None:
            x, energy = self.energy_predictor.infer(x, input_padding_mask, e_factor)
        else:
            energy = None

        w = torch.exp(durations) * x_mask
        w_ceil = torch.ceil(w) * d_factor
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)
        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text
        y = torch.matmul(attn.squeeze(1).float().transpose(1, 2), x)
        # y = y.transpose(1, 2)

        # Decoder
        target_padding_mask = ~y_mask.squeeze(1).bool().to(x.device)
        y = self.decoder(y, target_padding_mask)
        am_infer = (perf_counter() - am_t0) * 1000

        v_t0 = perf_counter()
        # Generate wav
        wav = self.vocoder(
            y.transpose(1, 2),
            f0=None,
            padding_mask=target_padding_mask
        )
        wav_lengths = y_lengths * self.hop_length
        v_infer = (perf_counter() - v_t0) * 1000

        wav_t = wav.shape[-1] / (self.sample_rate * 1e-3)
        am_rtf = am_infer / wav_t
        v_rtf = v_infer / wav_t
        rtf = am_rtf + v_rtf
        latency = am_infer + v_infer

        return {
            "wav": wav.detach().cpu(),
            "wav_lengths": wav_lengths.detach().cpu(),
            "durations": durations.detach().cpu(),
            "pitch": pitch.detach().cpu(),
            "energy": energy.detach().cpu(),
            "am_rtf": am_rtf,
            "v_rtf": v_rtf,
            "rtf": rtf,
            "latency": latency,
        }


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path
def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape
