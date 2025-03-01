VLMS: Vision Language Models
This project implements the paper "An Image is Worth 16x16 Words".
This is the third paper in the VLMS series that I am exploring, and it represents a milestone in vision-language research. It is particularly significant because it lays the foundation for applying transformer 
architectures—originally designed for language tasks—to vision problems.

![image](https://github.com/user-attachments/assets/43b5bc6c-792d-4bfe-8dd3-e5018b1f1592)


Overview
Paper Background:
The paper introduces the concept of dividing an image into 16×16 patches and treating these patches as “words” for a transformer. This approach revolutionized vision tasks by leveraging the self-attention mechanisms of transformers.

Series Context:
This implementation is the third installment in the VLMS series. Each paper in the series builds upon previous insights, and this one marks the first time a transformer—traditionally used in natural language processing—is applied directly to image classification tasks.

Implementation Highlights:
Due to the heavy data requirements for training transformers, I validated the model using dummy inputs to ensure that all components of the model are correctly implemented. The diagram above provides a quick visual summary of the model’s workflow.

Dummy Validation Workflow
To verify the implementation, I followed these steps:

Model Instance Creation:
Initialized the Vision Transformer (ViT) model with parameters similar to ViT-Base.

Dummy Input Preparation:
Created a dummy input tensor simulating a batch of images.

Patch Embedding & Projection:
Processed the dummy input through the patch embedding module and verified the shape of the output.

Positional Encoding:
Prepended a class token to the patch embeddings and applied positional encoding. Shape assertions ensure correctness.

Transformer Encoder Layers:
Passed the encoded embeddings through the transformer encoder layers.

Classification Head:
Processed the encoder’s output with a classifier head that extracts the class token output for prediction.

Full Forward Pass:
Ran a complete forward pass through the model and asserted the final output shape.

```python 
# Example parameters (ViT-Base like)
image_size = 224
patch_size = 16
in_channels = 3
num_classes = 1000
embedding_dim = 768
depth = 12
num_heads = 12

# Instantiate the ViT model
model = ViTForImageClassification(
    image_size=image_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes,
    embedding_dim=embedding_dim,
    depth=depth,
    num_heads=num_heads
)

print("ViT Model Instantiated!")

# Create dummy input
batch_size = 2  # Example batch size
dummy_input = torch.randn(batch_size, in_channels, image_size, image_size)
print("Dummy Input Image Shape:", dummy_input.shape)

# Patch embedding and projection
patch_embeddings = model.patch_embed_proj(dummy_input)
print("Patch Embeddings Shape:", patch_embeddings.shape)
expected_num_patches = (image_size // patch_size) ** 2
assert patch_embeddings.shape == (batch_size, expected_num_patches, embedding_dim), \
       f"Patch Embeddings shape is incorrect. Expected {(batch_size, expected_num_patches, embedding_dim)}, but got {patch_embeddings.shape}"

# Positional Encoding
class_token = model.class_token.expand(batch_size, -1, -1)
embeddings_with_class_token = torch.cat((class_token, patch_embeddings), dim=1)
embeddings_with_pos = model.pos_embedding(embeddings_with_class_token)
print("Embeddings with Positional Encoding Shape:", embeddings_with_pos.shape)
expected_seq_len = expected_num_patches + 1  # +1 for class token
assert embeddings_with_pos.shape == (batch_size, expected_seq_len, embedding_dim), \
       f"Embeddings with Positional Encoding shape is incorrect. Expected {(batch_size, expected_seq_len, embedding_dim)}, but got {embeddings_with_pos.shape}"

# Transformer encoder layers
encoded_sequence = model.encoder_layers(embeddings_with_pos)
print("Output Shape after Transformer Encoder Layers:", encoded_sequence.shape)
assert encoded_sequence.shape == (batch_size, expected_seq_len, embedding_dim), \
       f"Encoder Output shape is incorrect. Expected {(batch_size, expected_seq_len, embedding_dim)}, but got {encoded_sequence.shape}"

# Transformer decoder layer with class token output (Classification Head)
output_logits = model.classifier_head(encoded_sequence)
print("Output Logits Shape from Classification Head:", output_logits.shape)
assert output_logits.shape == (batch_size, num_classes), \
       f"Classification Head output shape is incorrect. Expected {(batch_size, num_classes)}, but got {output_logits.shape}"

# Full forward pass
full_output_logits = model(dummy_input)
print("Output Logits Shape from Full Forward Pass:", full_output_logits.shape)
assert full_output_logits.shape == (batch_size, num_classes), \
       f"Full Forward Pass output shape is incorrect. Expected {(batch_size, num_classes)}, but got {full_output_logits.shape}"
