import torch
from torch import nn
from transformers import AutoTokenizer
from .facodec import AudioFacodecTokenizer


class MultiModalTokenizer:
    def __init__(
        self,
        text_tokenizer_cls: AutoTokenizer,
        audio_tokenizer_cls: AudioFacodecTokenizer,
        start_of_audio_token: str = "<|start_of_audio|>",
        end_of_audio_token: str = "<|end_of_audio|>",
        audio_sample_rate: int = 16000,
    ):

        self.start_of_audio_token = start_of_audio_token
        self.end_of_audio_token = end_of_audio_token
        self.audio_tokenizer = audio_tokenizer_cls
        self.text_tokenizer = text_tokenizer_cls

        self.n_text_tokens = len(self.text_tokenizer)
        self.n_audio_tokens = self.audio_tokenizer.n_audio_tokens + 2
        self.n_tokens = self.n_text_tokens + self.n_audio_tokens

        self.audio_sos_token_idx = self.n_tokens - 1
        self.audio_eos_token_idx = self.n_tokens

    def __len__(self):
        return self.n_tokens

    def _encode_text(self, text: str):
        text_tokens = self.text_tokenizer(text, return_tensors="pt")
        text_tokens["input_ids"] = text_tokens["input_ids"].squeeze(0)
        text_tokens["attention_mask"] = text_tokens["attention_mask"].squeeze(0)
        return text_tokens

    def _decode_text(self, text_tokens: torch.Tensor):
        return self.text_tokenizer.decode(text_tokens)

    def _encode_audio(self, audio: torch.Tensor):
        audio_codes = self.audio_tokenizer.encode(audio)
        return audio_codes

    def _decode_audio(self, audio_tokens: torch.Tensor, spk_embedding: torch.Tensor):
        return self.audio_tokenizer.decode(audio_tokens, spk_embedding)

    def encode(self, text: str = None, audio: torch.Tensor = None):
        if text is not None:
            text_tokens = self._encode_text(text)
        else:
            text_tokens = None

        if audio is not None:
            audio_tokens = self._encode_audio(audio)
        else:
            audio_tokens = None

        return {
            "text_tokens": text_tokens,
            "audio_tokens": audio_tokens,
        }

    def decode(
        self,
        text_tokens: torch.Tensor = None,
        audio_tokens: torch.Tensor = None,
        spk_embedding: torch.Tensor = None,
    ):
        assert (
            text_tokens is not None or audio_tokens is not None
        ), "Must provide text or audio tokens"

        # Decode text tokens
        decoded_text = self._decode_text(text_tokens)

        # Decode audio tokens
        decoded_audio = self._decode_audio(audio_tokens, spk_embedding)

        return {
            "text": decoded_text,
            "audio": decoded_audio,
        }


if __name__ == "__main__":
    import librosa
    import torchaudio
    import soundfile as sf

    text_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    audio_tokenizer = AudioFacodecTokenizer("cuda")
    tokenizer = MultiModalTokenizer(
        text_tokenizer_cls=text_tokenizer,
        audio_tokenizer_cls=audio_tokenizer,
    )

    # Test the tokenizer
    text = "Hello, world!"
    audio, sr = librosa.load("data/samples/test.mp3", sr=16000)
    audio = torch.from_numpy(audio).float()
    audio = audio.unsqueeze(0).unsqueeze(0)
    tokens = tokenizer.encode(text, audio)
    print(tokens['audio_tokens']['vq_codes'].shape, tokens['audio_tokens']['spk_embeddings'].shape)
    decoded = tokenizer.decode(
        tokens["text_tokens"]["input_ids"],
        tokens["audio_tokens"]["vq_codes"],
        tokens["audio_tokens"]["spk_embeddings"],
    )
    print(decoded['audio'].shape)
    sf.write("decoded.wav", decoded['audio'][0][0].cpu().numpy(), 16000)

