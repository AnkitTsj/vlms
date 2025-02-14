import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
import json
from torch.nn.utils.rnn import pad_sequence

# Load vocabulary and reverse vocabulary from JSON files
vocab = json.load(open('vocab.json', 'r'))
rev_vocab = json.load(open('vocab_rev.json', 'r'))
VSIZE = len(vocab)  # Vocabulary size


# Define the Captioner model
class Captioner(nn.Module):
    def __init__(self, vocab_size, hidden_state, batch_size, word_embedding_size, embedding_size, vocab, max_len,
                 rev_vocab, device):
        super(Captioner, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_state = hidden_state
        self.word_embedding_size = word_embedding_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.hooked_out = []  # List to store hooked outputs from intermediate layers
        self.embeddings = nn.Embedding(vocab_size, word_embedding_size).to(device)  # Word embedding layer
        self.justify_img_feats = nn.Linear(1024, embedding_size).to(device)  # Linear layer to adjust image features
        self.init_state_cell = nn.Linear(embedding_size, hidden_state).to(device)  # Linear layer for initial cell state
        self.init_state_hidden = nn.Linear(embedding_size, hidden_state).to(
            device)  # Linear layer for initial hidden state
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True).to(
            device)  # Pre-trained GoogLeNet model
        self.lstm = nn.LSTM(word_embedding_size, hidden_state, batch_first=True).to(device)  # LSTM layer
        self.final_layer = nn.Linear(hidden_state, vocab_size).to(device)  # Linear layer for final output
        self.rev_vocab = rev_vocab  # Reverse vocabulary mapping
        self.vocab = vocab  # Vocabulary mapping
        self.max_length = max_len  # Maximum caption length

    # Hook function to capture intermediate layer outputs
    def __append_hook(self, module, input, output):
        self.hooked_out.append(output)

    # Register a forward hook to a specific layer
    def __layer_hook(self, layer_id):
        for id, layer in self.model.named_modules():
            if id == layer_id:
                layer.register_forward_hook(self.__append_hook)

    # Forward pass for image embedding
    def embed_forward(self, X):
        with torch.no_grad():  # Disable gradient calculation for image embedding
            self.model.eval()  # Set model to evaluation mode
            self.hooked_out = []  # Reset hook outputs for each forward pass
            self.__layer_hook(layer_id="avgpool")  # Hook to the average pooling layer of GoogLeNet
            output = self.model(X)  # Forward pass through GoogLeNet

        image_feats = self.hooked_out[0].squeeze(-1).squeeze(-1)  # Extract image features
        lin_embeddings = self.justify_img_feats(image_feats)  # Linear transformation of image features
        hidden_state = self.init_state_hidden(lin_embeddings).unsqueeze(0)  # Initialize hidden state
        cell_state = self.init_state_cell(lin_embeddings).unsqueeze(0)  # Initialize cell state
        return hidden_state, cell_state, output  # Return hidden state, cell state, and GoogLeNet output

    # Forward pass for caption generation
    def forward(self, X, token_sequences):
        batch_size = X.size(0)  # Get batch size
        hidden_state, cell_state, _ = self.embed_forward(X)  # Get initial hidden and cell states
        token_embeddings = self.embeddings(
            token_sequences)  # [batch_size, sequence_length, embedding_dim] # Embed token sequences
        rnn_output, (hidden_state, cell_state) = self.lstm(token_embeddings,
                                                           (hidden_state, cell_state))  # LSTM forward pass
        logit_probs = self.final_layer(rnn_output)  # [batch_size, sequence_length, vocab_size] # Get logits
        return logit_probs  # Return logits for loss calculation

    # Generate a caption for a given image
    def generate_caption(self, image, start_token="<START>"):
        device = next(self.parameters()).device  # Get device of the model parameters
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension if necessary
        image = image.to(device)  # Move image to the device

        hidden_state, cell_state, _ = self.embed_forward(image)  # Get initial hidden and cell states
        # Start with start token embedding:
        token = torch.tensor([[self.vocab[start_token]]], dtype=torch.long, device=device)  # Create start token tensor
        caption = [start_token]  # Initialize caption list
        for i in range(self.max_length):  # Iterate up to max_length
            embedding = self.embeddings(token)  # shape [1, 1, embedding_dim] # Embed the current token
            rnn_output, (hidden_state, cell_state) = self.lstm(embedding,
                                                               (hidden_state, cell_state))  # LSTM forward pass
            logits = self.final_layer(rnn_output)  # [1, 1, vocab_size] # Get logits
            word_idx = torch.argmax(logits.squeeze(1), dim=-1).item()  # Get predicted word index
            word = self.rev_vocab[str(word_idx)]  # Get predicted word from reverse vocabulary
            caption.append(word)  # Add word to caption
            if word == "<END>":  # Break if end token is generated
                break
            token = torch.tensor([[word_idx]], dtype=torch.long, device=device)  # Create tensor for the predicted word
        return caption  # Return the generated caption


# Custom collate function for data loading
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


# Pad sequences to the same length
def pad_sequences(sequences, vocab):
    """Pad sequences to the same length"""
    max_len = max(len(seq) for seq in sequences)  # Get maximum sequence length
    padded_sequences = []

    for seq in sequences:
        padded = seq.copy()  # Copy the sequence
        while len(padded) < max_len:  # Pad until max_len is reached
            padded.append(vocab['<PAD>'])  # Append padding token
        padded_sequences.append(padded)  # Add padded sequence to the list

    return torch.tensor(padded_sequences, dtype=torch.long)  # Return padded sequences as a tensor


# Training function
def train(model, epochs, data_loader, optimizer, criterion, max_go=20000, ):
    model.train()  # Set model to training mode
    device = next(model.parameters()).device  # Get device of the model parameters

    for epoch in range(epochs):  # Iterate through epochs
        total_loss = 0  # Initialize total loss
        batch_count = 0  # Initialize batch count

        for imgs, captions in data_loader:  # Iterate through data loader
            if batch_count >= max_go:  # Break if max_go batches are processed
                break

            imgs = imgs.to(device)  # Move images to device
            optimizer.zero_grad()  # Zero gradients

            # Convert captions to character sequences
            char_sequences = []
            for cap in captions:  # Iterate through captions
                mid_list = [c for c in cap if c in vocab]  # Filter out-of-vocabulary characters
                chars = ['<START>'] + mid_list + ['<END>']  # Add start and end tokens
                char_sequences.append([vocab[c] for c in chars])  # Convert characters to indices

            # Pad sequences
            padded_sequences = pad_sequences(char_sequences, vocab)
            padded_sequences = padded_sequences.to(device)

            outputs = model(imgs, padded_sequences[:, :-1]) # Get Logits from Model
            targets = padded_sequences[:, 1:]

            # Reshape for CrossEntropyLoss
            # print(outputs.shape,targets.shape)
            outputs = outputs.view(-1, model.vocab_size) # [batch_size * seq_length, vocab_size]
            targets = targets.reshape(-1) # [batch_size * seq_length]


            loss = criterion(outputs, targets) # Calculate loss with logits and long targets
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_count % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_count}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), f'captioner_epoch_{epoch + 1}.pth')

    print("Training complete!")


def generate_caption(model,image):
    model.eval()
    with torch.no_grad():
        caption = model.generate_caption(image)
        return caption

