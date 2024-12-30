# SARJEPA
This is an unofficial copy of the SARJEPA implementation found at https://github.com/waterdisappear/SAR-JEPA/tree/main.

The reason this repo exists is because I found it difficult to follow their code and also believe that there are several suboptimal choices made in their implementation.

# TODO
* write docstrings
* finish code explanations
* investigate why a class token is added to the vit during pretraining. its odd that they add it in for pretraining and then opt for the smarter global pool during finetuning
* add back in some sort of positional encoding to the vit. they copied code from Cream iRPE but they didnt add in the positional encoding
* add in regularization such as attention dropout, projection dropout, and drop path
* investigate maintaining token masking throughout the encoder blocks and not just after the initial patchification. i have a hunch that by allowing masked tokens to be updated throughout the encoder it is learning to reconstruct the missing tokens in the encoder and not forcing that to be done in the decoder
* group the lomar vit constructor into vit, lomar/mae, and sarjepa pieces
* fix comments in the GF object since it says it accepts a RGB image but it really only accepts a single channel image (in_chans needs to be set to 1 everywhere)