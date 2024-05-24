# Overview
The purpose of this repo is to try and adapt an LLM to be able to directly generate speech tokens.

# Approach
1. Audio tokenizer is critical. Want to compress audio into a small a token as possible. I found amphion/naturalspeech3_facodec to be the best specifically the V2 which had the best feature disentaglment and compression.
2. I replaced the head of a pretrained LLM with 6 linear layers which each generate a token for the relevant code book (1 code for prosody 2 codes for content, and 3 residual audio codes).



<!-- Pick up point -->
0. Add seperate embedding dicts for each token type: prosody, content, and residual audio
1. Get train loop working
2. Overfit a tiny model
