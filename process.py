"""
Process dataset into .bin file
"""

import os
import glob
import torch
import fnmatch
import librosa
import numpy as np
from tqdm import tqdm
from tokenizer.tokenizer import AudioFacodecTokenizer


def process_audio_files(
    input_dir: str,
    output_dir: str,
    train_split: float = 0.9,
    val_split: float = 0.1,
):
    """
    Get all audio files in a directory.
    Tokenize using audio tokenizer
    Flatten into long list of tokens
    Save as .bin file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_tokenizer = AudioFacodecTokenizer(device=device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isdir(input_dir):
        audio_files = []
        for ext in ('*.wav', '*.mp3', '*.flac'):
            audio_files.extend(fnmatch.filter(os.listdir(input_dir), ext))
        audio_files = [os.path.join(input_dir, f) for f in audio_files]
        print(f"Found {len(audio_files)} audio files")
    else:
        audio_files = [input_dir]
    
    if len(audio_files) == 0:
        raise ValueError("No audio files found")

    if os.path.exists(output_dir + "/vq_codes.bin"):
        rm_cmd = input("Found existing vq_codes.bin, remove? (y/n) ")
        if rm_cmd == "y":
            os.remove(output_dir + "/vq_codes.bin")

    all_vqs = []
    for audio_file in tqdm(audio_files):
        wav = librosa.load(audio_file, sr=16000)[0]
        wav = torch.from_numpy(wav).float()
        wav = wav.unsqueeze(0).unsqueeze(0)
        token_data = audio_tokenizer.encode(wav, show_progress=False)
        vq_codes = token_data["vq_codes"].cpu().numpy()
        all_vqs.append(vq_codes)

    # Flatten and write all vq_codes to the file at once
    concatenated_vqs = np.concatenate(all_vqs, axis=-1)
    with open(output_dir + "/vq_codes.bin", "wb") as f:
        np.array(concatenated_vqs.reshape(-1), dtype="int32").tofile(f)

    # Read the entire file and reshape
    vq_codes_read = np.fromfile(output_dir + "/vq_codes.bin", dtype="int32")
    vq_codes_read = vq_codes_read.reshape(6, 1, -1)

    # Compare the concatenated all_vqs with the reshaped data from the file
    # as a sanity check
    assert np.array_equal(concatenated_vqs, vq_codes_read), 'Arrays are not equal!'
    
    # Split into train and val without loading the whole file
    vq_codes_path = output_dir + "/vq_codes.bin"
    vq_codes = np.memmap(vq_codes_path, dtype="int32", mode="r")

    # Calculate the split indices
    total_length = len(vq_codes)
    # Round to nearest multiple of 6 as vq_codes are 6 codes per sample
    train_length = int(np.ceil(total_length * train_split / 6) * 6)
    val_length = total_length - train_length

    # Create memmap files for train and val
    train_file = np.memmap(
        output_dir + "/train.bin", dtype="int32", mode="w+", shape=(train_length,)
    )
    val_file = np.memmap(
        output_dir + "/val.bin", dtype="int32", mode="w+", shape=(val_length,)
    )

    # Write the data to the memmap files
    train_file[:] = vq_codes[:train_length]
    val_file[:] = vq_codes[train_length:]

    # Flush changes to disk
    train_file.flush()
    val_file.flush()

    # Clean up
    del train_file
    del val_file
    del vq_codes


def test_read_bin():
    import soundfile as sf
    sample = np.memmap("data/vq_codes.bin", dtype="int32", mode="r")
    # Read 1500 * 6 samples, reshape and decode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_tokenizer = AudioFacodecTokenizer(device=device)

    spk_embedding = torch.load('data/samples/test_spk_embedding.pt')
    sample = sample.reshape(6, 1, -1)
    # Select a random section of the sample
    random_idx = (np.random.randint(0, sample.shape[-1]- 512))
    sample = sample[:,:,random_idx:random_idx + 512]
    sample = torch.from_numpy(sample).to(device).int()
    wav = audio_tokenizer.decode(sample, spk_embedding)
    print(wav.shape)
    sf.write("decoded.wav", wav[0][0].cpu().numpy(), 16000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # process_audio_files(args.input_dir, args.output_dir)
    test_read_bin()
