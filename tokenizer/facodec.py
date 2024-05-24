import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from .ns3_codec import FACodecEncoderV2, FACodecDecoderV2


class AudioFacodecTokenizer:
    def __init__(self, device="cpu"):
        self.device = device
        self.fa_encoder = FACodecEncoderV2(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )

        self.fa_decoder = FACodecDecoderV2(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        self.n_audio_tokens = (
            self.fa_decoder.vq_num_q_c
            + self.fa_decoder.vq_num_q_p
            + self.fa_decoder.vq_num_q_r
        )

        encoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin"
        )
        decoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin"
        )

        self.fa_encoder.load_state_dict(torch.load(encoder_ckpt))
        self.fa_decoder.load_state_dict(torch.load(decoder_ckpt))
        self.fa_encoder.eval()
        self.fa_decoder.eval()
        self.fa_encoder.to(self.device)
        self.fa_decoder.to(self.device)

    def encode(
        self,
        wav: torch.Tensor,
        n_seconds_chunk_size: int = 20,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Note that audio must be decoded in the same chunk size as it was encoded otherwise
        """
        assert len(wav.shape) == 3, "Wav shape should be: [BS, 1, N_SAMPLES]"
        # split into chunks
        wavs = wav.split(n_seconds_chunk_size*16000, dim=-1)
        # quantize
        tqdm_wavs = tqdm(wavs, 'Encoding Wavs: ') if show_progress else wavs
        vq_codes = []
        spk_embs = []
        with torch.no_grad():
            for i, wav in enumerate(tqdm_wavs):
                wav = wav.to(self.device)
                latents = self.fa_encoder(wav)
                prosody_feature = self.fa_encoder.get_prosody_feature(wav)
                min_len = min(latents.shape[-1], prosody_feature.shape[-1])
                latents = latents[:,:,:min_len]
                prosody_feature = prosody_feature[:,:,:min_len]
                vq_post_emb, vq_id, _, quantized, spk_emb = self.fa_decoder(
                    latents,
                    prosody_feature,
                    eval_vq=False,
                    vq=True,
                )
                vq_codes.append(vq_id)
                spk_embs.append(spk_emb)
            return {
                "vq_codes": torch.cat(vq_codes, dim=-1),  # [N_Quant, BS, Frames]
                "spk_embeddings": spk_embs[0],
                "prosody_codes": vq_id[:1],
                "content_codes": vq_id[1:3],
                "residual_codes": vq_id[3:],
            }

    def decode(
        self, tokens: torch.Tensor, spk_embeddings: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            tokens = tokens.to(self.device)
            spk_embeddings = spk_embeddings.to(self.device)
            vq_post_emb = self.fa_decoder.vq2emb(tokens)
            return self.fa_decoder.inference(vq_post_emb, spk_embeddings)


if __name__ == "__main__":
    import librosa
    import soundfile as sf

    tokenizer = AudioFacodecTokenizer(device="cuda")
    test_wav = librosa.load("data/full_audio/_2VtdSLr3Ck.mp3", sr=16000)[0]
    test_wav = torch.from_numpy(test_wav).float()
    test_wav = test_wav.unsqueeze(0).unsqueeze(0)
    token_data = tokenizer.encode(test_wav)

    recon_wav = tokenizer.decode(token_data["vq_codes"], token_data["spk_embeddings"])
    sf.write("recon.wav", recon_wav[0][0].cpu().numpy(), 16000)
    print("Done")
