import dac
import torch
import torchaudio
from tqdm import tqdm
from audiotools import AudioSignal
from transformers import AutoTokenizer
from audiolm_pytorch import EncodecWrapper


class AudioCodec:
    """https://github.com/descriptinc/descript-audio-codec"""

    def __init__(self, device="cuda", sample_rate: int = 16000):
        assert sample_rate in [16000, 24000, 44100]
        model_types = {
            16000: "16khz",
            24000: "24khz",
            44100: "44khz",
        }
        # Download and load the model
        model_path = dac.utils.download(model_type="16khz")
        self.model = dac.DAC.load(model_path)
        self.model.to(device).eval()
        self.device = device
        self.sample_rate = sample_rate

    def encode(self, y: torch.Tensor, chunk_size_seconds: int = 10):
        """
        Audio will be split into chunks of chunk_size_seconds and then encoded
        """
        # Load audio signal file
        signal = AudioSignal(y, sample_rate=self.sample_rate)
        signal = signal.resample(self.sample_rate)
        # If 2 channels take mean
        if signal.num_channels == 2:
            signal.audio_data = signal.audio_data.mean(axis=1).unsqueeze(1)
        # Split audio into chunks
        audio_data = signal.audio_data.squeeze(1)
        chunk_size = int(self.sample_rate * chunk_size_seconds)
        chunks = torch.split(audio_data, chunk_size, dim=1)

        z_chunks, codes_chunks = [], []
        for chunk in tqdm(chunks[:6]):
            chunk_signal = AudioSignal(chunk.unsqueeze(1), sample_rate=self.sample_rate)
            chunk_signal.to(self.device)
            x = self.model.preprocess(chunk_signal.audio_data, chunk_signal.sample_rate)
            with torch.no_grad():
                z, codes, latents, _, _ = self.model.encode(x)
                z_chunks.append(z.squeeze(0).cpu())
                codes_chunks.append(codes.squeeze(0).cpu())

        # Join and return
        latents = torch.cat(z_chunks, dim=1)
        codes = torch.cat(codes_chunks, dim=1)
        return latents, codes

    def decode_codes(self, codes):
        if len(codes.shape) == 2:
            codes = codes.unsqueeze(0).to(self.model.device)
        z_q, z_p, codes = self.model.quantizer.from_codes(codes)
        # Split into chunks
        z_q_chunks = torch.split(z_q, 2000, dim=2)
        y_chunks = []
        with torch.no_grad():
            for z_q_chunk in tqdm(z_q_chunks):
                y = self.model.decode(z_q_chunk)
                y_chunks.append(y.squeeze(0).cpu())
            y = torch.cat(y_chunks, dim=-1)
        return y


if __name__ == "__main__":
    codec = AudioCodec(sample_rate=16000)
    y, sr = torchaudio.load("test.mp3", 16000)
    z, codes = codec.encode(y, 10)
    print(codes.shape)
    y = codec.decode_codes(codes)
    torchaudio.save("output_recon.mp3", y, 44100)

    print("Meta Encodec")
    encodec = EncodecWrapper()
    print(encodec.model.target_bandwidths)
    encodec.model.set_target_bandwidth(6.0)
    _, codes, _ = encodec(y)
    y = encodec.decode_from_codebook_indices(codes)
    print(codes.shape)
    print(y.shape)
    torchaudio.save("output_recon_meta.mp3", y[0], 44100)
