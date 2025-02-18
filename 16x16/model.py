import torch
from torch import nn


class Patch_Embedding_Projection(nn.Module):
    """
    Module for patch embedding projection.

    Divides the input image into patches, flattens them, and applies a linear projection.
    """
    def __init__(self, patch_size, in_channels, embedding_dim): # Added embedding_dim parameter
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim # Store embedding dimension
        self.patch_former = nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=0) # Extracts non-overlapping patches
        self.lin_proj = nn.Linear(in_features=patch_size ** 2 * self.in_channels, out_features=embedding_dim) # Linear projection layer

    def forward(self, x):
        """Forward pass for patch embedding projection.

        Args:
            x: Input image tensor of shape (batch_size, in_channels, H, W)

        Returns:
            Tensor of patch embeddings of shape (batch_size, num_patches, embedding_dim)
        """
        patches = self.patch_former(x) # Shape: (batch_size, patch_size*patch_size*in_channels, num_patches)
        x = patches.transpose(1, 2) # Shape: (batch_size, num_patches, patch_size*patch_size*in_channels) - Make num_patches the sequence dimension
        return self.lin_proj(x) # Apply linear projection and return embeddings


class POS_ENC(nn.Module):
    """
    Module for positional encoding.

    Adds learnable positional embeddings to the input sequence.
    """
    def __init__(self, seq_len, enc_dim): # seq_len is now a parameter
        super(POS_ENC, self).__init__()
        self.enc_dim = enc_dim
        self.pos_embeddings = nn.Parameter(torch.randn(1, seq_len, enc_dim)) # Learnable positional embeddings

    def forward(self, x):
        """Forward pass for positional encoding.

        Args:
            x: Input tensor of shape (batch_size, seq_len, enc_dim)

        Returns:
            Tensor with positional embeddings added, shape (batch_size, seq_len, enc_dim)
        """
        # x shape: (batch_size, seq_len, enc_dim)
        # pos_embeddings shape: (1, seq_len, enc_dim) - broadcasted along batch dimension
        return x + self.pos_embeddings # Element-wise addition of positional embeddings


class Encoder(nn.Module):
    """
    Transformer Encoder Block.

    Consists of LayerNorm, Multi-Head Self-Attention, Residual Connection,
    LayerNorm, MLP, and another Residual Connection.
    """
    def __init__(self, enc_dim, num_heads):
        super(Encoder, self).__init__()
        self.enc_dim = enc_dim
        self.num_heads = num_heads
        self.head_dim = enc_dim // num_heads # Dimension of each attention head

        assert enc_dim % num_heads == 0, "enc_dim must be divisible by num_heads" # Ensure divisibility

        self.norm1 = nn.LayerNorm(enc_dim) # LayerNorm before MHSA

        # Linear layers for Multi-Head Self-Attention
        self.wq = nn.Linear(in_features=enc_dim, out_features=enc_dim) # Query projection
        self.wk = nn.Linear(in_features=enc_dim, out_features=enc_dim) # Key projection
        self.wv = nn.Linear(in_features=enc_dim, out_features=enc_dim) # Value projection

        self.softmax = nn.Softmax(dim=-1) # Softmax for attention weights
        self.norm2 = nn.LayerNorm(enc_dim) # LayerNorm before MLP
        self.mlp = nn.Sequential( # MLP block
            nn.Linear(in_features=enc_dim, out_features=enc_dim*4), # First linear layer (expansion)
            nn.GELU(), # GELU activation
            nn.Linear(in_features=enc_dim*4, out_features=enc_dim)) # Second linear layer (projection back)
        self.proj_o = nn.Linear(in_features=enc_dim, out_features=enc_dim) # Output projection for MHSA

    def attention(self, x):
        """Multi-Head Self-Attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, enc_dim)

        Returns:
            Attention output tensor of shape (batch_size, seq_len, enc_dim)
        """
        batch_size, seq_len, enc_dim = x.shape

        q = self.wq(x) # Project to Queries
        k = self.wk(x) # Project to Keys
        v = self.wv(x) # Project to Values

        # Reshape for multi-head attention: (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        k_q = torch.matmul(q, k.transpose(-2, -1)) # Q @ K^T
        s_kq = self.softmax(k_q / (self.head_dim**0.5)) # Scale and apply softmax
        attn_output_heads = torch.matmul(s_kq, v) # Attention weights @ V

        # Concatenate heads and project: (batch_size, seq_len, enc_dim)
        attn_output_concat = attn_output_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, enc_dim)
        attn_output_proj = self.proj_o(attn_output_concat) # Output projection

        return attn_output_proj

    def forward(self, x):
        """Forward pass for the Encoder block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, enc_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, enc_dim)
        """
        x_residual = x # Store input for residual connection
        x =  self.norm1(x) # LayerNorm before MHSA
        attn_output = self.attention(x) # Multi-Head Self-Attention
        x = attn_output + x_residual # Residual connection for MHSA

        x_residual_mlp = x # Store input for residual connection to MLP
        x = self.norm2(x) # LayerNorm before MLP
        mlp_output = self.mlp(x) # MLP block
        x = mlp_output + x_residual_mlp # Residual connection for MLP
        return x


class ViTClassifierHead(nn.Module): # Renamed to ViTClassifierHead for clarity
    """
    Classification head for Vision Transformer.

    Simple linear layer to project the [class] token representation to class logits.
    """
    def __init__(self, enc_dim, num_classes): # num_classes parameter
        super().__init__()
        self.fc = nn.Linear(in_features=enc_dim, out_features=num_classes) # Single Linear Layer to num_classes

    def forward(self, x):
        """Forward pass for the classification head.

        Args:
            x: Input tensor from Transformer Encoder (batch_size, seq_len, enc_dim)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # x is expected to be the output from Transformer Encoder (batch_size, seq_len, enc_dim)
        # We extract the [class] token (usually the first token in the sequence - index 0)
        class_token = x[:, 0, :] # Shape: (batch_size, enc_dim) - Extract [class] token representation
        logits = self.fc(class_token) # Linear projection to num_classes
        return logits # Shape: (batch_size, num_classes)


class ViTForImageClassification(nn.Module): # Renamed to ViTForImageClassification
    """
    Vision Transformer (ViT) model for image classification.

    Combines patch embedding, positional encoding, Transformer Encoder layers, and a classification head.
    """
    def __init__(self, image_size, patch_size, in_channels, num_classes, embedding_dim, depth, num_heads): # Added num_classes, depth, renamed embedding_dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes # Number of classes for classification
        self.embedding_dim = embedding_dim # Embedding dimension for patches and tokens
        self.depth = depth # Number of Transformer Encoder layers
        self.num_heads = num_heads # Number of attention heads in MHSA

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size." # Check patch size validity
        self.num_patches = (image_size // patch_size) ** 2 # Calculate number of patches
        self.seq_len = self.num_patches + 1 # Sequence length = num_patches + class token

        self.patch_embed_proj = Patch_Embedding_Projection(patch_size=patch_size, in_channels=in_channels, embedding_dim=embedding_dim) # Patch embedding layer
        self.pos_embedding = POS_ENC(seq_len=self.seq_len, enc_dim=self.embedding_dim) # Positional encoding layer
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) # Learnable [class] token

        # Stack Transformer Encoder layers
        self.encoder_layers = nn.Sequential(*[Encoder(enc_dim=embedding_dim, num_heads=num_heads) for _ in range(depth)]) # Stack Encoder blocks
        self.norm_head = nn.LayerNorm(embedding_dim) # LayerNorm before classification head
        self.classifier_head = ViTClassifierHead(enc_dim=embedding_dim, num_classes=num_classes) # Classification head

    def forward(self, x):
        """Forward pass for the Vision Transformer model.

        Args:
            x: Input image tensor of shape (batch_size, in_channels, image_size, image_size)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]

        # 1. Patch Embedding Projection
        patch_embeddings = self.patch_embed_proj(x) # Shape: (batch_size, num_patches, embedding_dim)

        # 2. Positional Embedding and Class Token
        class_token = self.class_token.expand(batch_size, -1, -1) # Expand class token to batch size
        embeddings = torch.cat((class_token, patch_embeddings), dim=1) # Prepend class token to patch embeddings
        embeddings_with_pos = self.pos_embedding(embeddings) # Add positional embeddings

        # 3. Transformer Encoder
        encoded_sequence = self.encoder_layers(embeddings_with_pos) # Pass through Encoder layers
        pooled = self.norm_head(encoded_sequence) # LayerNorm before head

        # 4. Classification Head
        logits = self.classifier_head(pooled) # Get class logits

        return logits # Shape: (batch_size, num_classes)


# # Example Usage: (Removed from main code for clarity, can be in a separate example file or in README)
# if __name__ == '__main__':
#     image_size = 224
#     patch_size = 16
#     in_channels = 3
#     num_classes = 1000
#     embedding_dim = 768
#     depth = 12 # Number of Encoder layers
#     num_heads = 12
#
#     model = ViTForImageClassification(image_size=image_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes,
#                                         embedding_dim=embedding_dim, depth=depth, num_heads=num_heads)
#
#     batch_size = 2
#     dummy_input = torch.randn(batch_size, in_channels, image_size, image_size)
#     output_logits = model(dummy_input)
#
#     print("Output Logits Shape:", output_logits.shape) # Expected: torch.Size([2, 1000])