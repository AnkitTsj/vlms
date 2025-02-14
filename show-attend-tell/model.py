from torchvision.transforms import transforms
import torch.nn as nn
import torch
import torch
import torch.nn.functional as F

class SoftAttention(nn.Module):
    """
    Soft Attention mechanism for image captioning.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim,device):
        """
        :param encoder_dim: Feature size of encoded images (D).
        :param decoder_dim: Size of decoder's hidden state (n).
        :param attention_dim: Size of the attention network.
        """
        super(SoftAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim).to(device)  # Linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim).to(device)  # Linear layer to transform decoder's hidden state
        self.full_att = nn.Linear(attention_dim, 1).to(device)  # Linear layer to calculate attention scores
        self.relu = nn.ReLU().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)  # Softmax layer to calculate attention weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.

        :param encoder_out: Encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: Previous decoder hidden state, a tensor of dimension (batch_size, decoder_dim)
        :return: attention-weighted encoding, attention weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class HardAttention(nn.Module):
    """
    Stochastic Hard Attention mechanism for image captioning.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim,device):
        """
        :param encoder_dim: Feature size of encoded images (D).
        :param decoder_dim: Size of decoder's hidden state (n).
        :param attention_dim: Size of the attention network.
        """
        super(HardAttention, self).__init__(device)
        self.encoder_att = nn.Linear(encoder_dim, attention_dim).to(device)  # Linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim).to(device)  # Linear layer to transform decoder's hidden state
        self.full_att = nn.Linear(attention_dim, 1).to(device)  # Linear layer to calculate attention scores
        self.relu = nn.ReLU().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)  # Softmax layer to calculate probabilities

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.

        :param encoder_out: Encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: Previous decoder hidden state, a tensor of dimension (batch_size, decoder_dim)
        :return: attention-weighted encoding, attention weights (probabilities)
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)  These are now *probabilities*

        # Sampling
        sampled_indices = torch.multinomial(alpha, 1).squeeze(1)  # (batch_size)
        # Expand sampled_indices to have the same dimensions as encoder_out
        # Create s_t - one hot encoded matrix.
        s_t = torch.zeros_like(alpha, device=alpha.device).scatter_(1, sampled_indices.unsqueeze(1), 1)
        attention_weighted_encoding = (encoder_out * s_t.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha



from torch.nn.utils.rnn import pack_padded_sequence
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models  # Import torchvision.models
import time
import os
import json
from PIL import Image


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab

def build_vocab(captions):
    """Builds a character-level vocabulary."""
    char_dict = {}
    for caption in captions:
        for char in str(caption).lower():
            if char not in char_dict:
                char_dict[char] = len(char_dict)
    char_dict['<PAD>'] = len(char_dict)
    char_dict['<START>'] = len(char_dict)
    char_dict['<END>'] = len(char_dict)
    char_dict['<UNK>'] = len(char_dict)
    return char_dict

def pad_sequences(sequences, vocab):
    """Pads sequences to the same length."""
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padded = seq + [vocab['<PAD>']] * (max_len - len(seq))
        padded_sequences.append(padded)
    return torch.tensor(padded_sequences, dtype=torch.long)


class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14,device = "cpu"):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = models.resnet101(weights="DEFAULT").to(device)  # Use pre-trained weights
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules).to(device)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)).to(device)
        self.fine_tune()

    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune



class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, device,encoder_dim=2048, dropout=0.5, attention_type='soft'):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention_type = attention_type

        if attention_type == 'soft':
            self.attention = SoftAttention(encoder_dim, decoder_dim, attention_dim,device= device)
        elif attention_type == 'hard': #  Hard attention
            self.attention = HardAttention(encoder_dim, decoder_dim, attention_dim,device = device)
        else:
            raise ValueError("Invalid attention_type. Choose 'soft' or 'hard'.")

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.dropout = nn.Dropout(p=self.dropout).to(device)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True).to(device)
        self.init_h = nn.Linear(encoder_dim, decoder_dim).to(device)
        self.init_c = nn.Linear(encoder_dim, decoder_dim).to(device)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim).to(device)
        self.sigmoid = nn.Sigmoid().to(device)
        self.fc = nn.Linear(decoder_dim, vocab_size).to(device)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        # print(f"Encoder output device: {encoder_out.device}")
        # print(f"Decoder hidden device: {h.device}")
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


# --- Main Training Script ---

import random
from torch.utils.data import Subset
  # Example

# Get a list of all image indices (or create a list of indices if needed)
  # Assuming 'dataset' is your CocoCaptions dataset

# Shuffle the indices and select the first 'num_samples_to_train_on'


# Create a Subset of your dataset using the selected indices



def train(main_dir,ann_file,batch_size,vocab_path,attention_dim,embed_dim,decoder_dim,fine_tune_encoder,learning_rate,num_epochs,checkpoint_path,best_loss,num_samples_to_train_on = 3000):
    # Hyperparameters

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # # 1. Load COCO dataset using YOUR original method:
    # dataset = datasets.CocoCaptions(
    #     root=train_img_dir,
    #     annFile=train_ann_file,
    #     transform=transform
    # )
    # # Collate function
    def custom_collate_fn(batch):
        """
        batch: list of tuples (image, captions)
        We'll select the first caption for each image.
        """
        images, caption_lists = zip(*batch)  # Unpack batch
        # Stack images (assuming they are already tensors of the same shape)
        images = torch.stack(images, dim=0)  # Stack images

        # Select the first caption for each image
        captions = [caps[0] for caps in caption_lists]  # Select the first caption

        return images, captions  # Return images and captions

    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    dataset = datasets.CocoCaptions(
        root=main_dir,
        annFile=ann_file,
        transform=transform
    )

    all_indices = list(range(len(dataset)))

    selected_indices = random.sample(all_indices, num_samples_to_train_on)
    # print(len(selected_indices))
    subset_dataset = Subset(dataset, selected_indices)
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    # 2. Build or load vocabulary (character-level):
    if os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
    else:
        print("Building vocabulary...")
        # Get all captions.  Note: We're getting captions directly from the dataset.
        captions = [ann['caption'] for img_id, anns in dataset.coco.imgToAnns.items() for ann in anns]
        vocab = build_vocab(captions)  # build_vocab now creates a character-level vocab
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        print(f"Vocabulary saved to {vocab_path}")
    vocab_size = len(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 3. Initialize models
    encoder = EncoderCNN(device=device)
    decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size, device, attention_type="soft")

    # Move models to device

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # 4. Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(decoder.parameters())
    if fine_tune_encoder:
        params += list(encoder.parameters())
        encoder.fine_tune(fine_tune_encoder)
    optimizer = optim.Adam(params, lr=learning_rate)

    # 5. Training loop (adapted for character-level and your data loading)
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0.0
        encoder.train()
        decoder.train()
        for i, (images, captions) in enumerate(data_loader):
            images = images.to(device)
            # print(i)
            # Convert captions to character sequences of IDs
            char_sequences = []
            lengths = []
            for cap in captions:
                # Character-level tokenization:
                chars = ['<START>'] + list(str(cap).lower()) + ['<END>']
                char_ids = [vocab[c] if c in vocab else vocab['<UNK>'] for c in chars]  # Added the condition.
                char_sequences.append(char_ids)
                lengths.append(len(char_ids))
            # print(f"Images device: {images.device}")


            # Pad sequences
            padded_sequences = pad_sequences(char_sequences, vocab)  # pad the sequence
            lengths = torch.tensor(lengths, dtype=torch.long)  # lengths
            padded_sequences = padded_sequences.to(device)  # move to device.

            # Forward pass
            features = encoder(images)
            outputs, captions_sorted, decode_lengths, alphas, sort_ind = decoder(features, padded_sequences,
                                                                                 lengths.unsqueeze(1))  # Use lengths
            targets = captions_sorted[:, 1:]

            # Pack padded sequence
            outputs = pack_padded_sequence(outputs, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(outputs, targets)
            # print(f"Loss device: {loss.device}")
            # Add doubly stochastic attention regularization
            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Backward pass and optimization
            decoder.zero_grad()
            if fine_tune_encoder:
                encoder.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Print training statistics
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}, Perplexity: {torch.exp(loss).item():.4f}')

        epoch_loss = running_loss / len(data_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {time.time() - start:.2f}s')

        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    print("Training complete")

from torchvision.transforms import transforms
import torch.nn as nn
import torch



def generate_sequence(encoder, decoder, image, vocab, max_len=30, device="cpu"):
    """
    Generates a caption for a given image using the trained encoder and decoder.

    Args:
        encoder: Trained encoder model.
        decoder: Trained decoder model.
        image: Preprocessed image tensor (output of transform).
        vocab: Vocabulary dictionary.
        max_len: Maximum length of the generated caption.
        device: Device to use (CPU or GPU).

    Returns:
        caption: Generated caption (string).
        alphas: Attention weights for each timestep (optional).
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
            alphas.append(alpha.cpu().numpy()) # Store alpha for visualization if needed

            # If <END> token is predicted, stop generating
            if predicted_idx == vocab['<END>']:
                break

    # Convert token IDs to characters
    char_caption = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in caption[1:-1]] # Remove <START> and <END>
    caption_str = "".join(char_caption)
    return caption_str, alphas


