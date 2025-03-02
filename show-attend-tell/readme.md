# Show, Attend and Tell - Image Captioning with Attention

## Overview
This repository contains an implementation of the paper ["Show, Attend and Tell"](https://arxiv.org/pdf/1502.03044), which enhances the idea that information contained in Convolutional Neural Networks (CNNs) can be directly interpreted to generate captions describing the objects in an image.

## Concept
Instead of a simple CNN to RNN (LSTM) architecture for image captioning, this technique utilizes **attention mechanisms** to improve communication between tokens. The attention mechanism ensures that long-term dependencies are properly carried forward, enhancing the semantic understanding of the generated captions.

## Model Architecture
1. **CNN Encoder**: Extracts feature representations from the input image.
2. **Attention Mechanism**: Dynamically focuses on different regions of the image during caption generation.
3. **LSTM Decoder**: Generates captions token by token, influenced by the attention-weighted encoding.

![Architecture](https://github.com/user-attachments/assets/96680938-a388-40be-b4ee-f524b3ab5fd4)

## Caption Generation Code
The function `generate_sequence` produces a caption for a given image using a trained encoder-decoder model.

### Function Breakdown
```python
import torch

def generate_sequence(encoder, decoder, image, vocab, max_len=30, device="cpu"):
    """
    Generates a caption for a given image using the trained encoder and decoder.
    
    Args:
        encoder: Trained encoder model.
        decoder: Trained decoder model.
        image: Preprocessed image tensor.
        vocab: Vocabulary dictionary.
        max_len: Maximum length of the generated caption.
        device: Device to use (CPU or GPU).

    Returns:
        caption: Generated caption (string).
        alphas: Attention weights for each timestep.
    """

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        encoder_out = encoder(image)
        h, c = decoder.init_hidden_state(encoder_out)

        # Initialize caption list with <START> token ID
        start_token_id = vocab['<START>']
        caption = [start_token_id]

        alphas = []  # Initialize attention weights list

        for i in range(max_len):
            attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # Embed the last predicted token
            embed = decoder.embedding(torch.tensor([caption[-1]], device=device))

            # Concatenate embedding and attention-weighted encoding
            lstm_input = torch.cat([embed, attention_weighted_encoding], dim=1)

            h, c = decoder.decode_step(lstm_input, (h, c))

            preds = decoder.fc(decoder.dropout(h))

            # Get predicted token ID
            _, predicted_idx = torch.max(preds, 1)
            predicted_idx = predicted_idx.item()

            # Append predicted token ID to caption
            caption.append(predicted_idx)
            alphas.append(alpha.cpu().numpy()) # Store alpha for visualization

            # Stop if <END> token is predicted
            if predicted_idx == vocab['<END>']:
                break
```

### Explanation
- **Encoder processes the image** and extracts feature representations.
- **Attention mechanism** helps the decoder focus on different parts of the image at each step.
- **LSTM decoder** generates words sequentially while updating its hidden state.
- **Attention weights (`alphas`)** are stored for visualization.
- Stops when `<END>` token is predicted.

## Usage
1. Train the encoder-decoder model on an image-caption dataset.
2. Use `generate_sequence` to generate captions for new images.
3. Visualize attention weights to understand where the model is focusing.

## References
- [Show, Attend and Tell Paper](https://arxiv.org/pdf/1502.03044)

