# VLMS: Vision Language Models

This project involves the implementation of the paper *An Image is Worth 16x16 Words*. This is the third paper in the series of VLM (Vision-Language Model) papers I am learning about.

![Model Diagram](https://github.com/user-attachments/assets/43b5bc6c-792d-4bfe-8dd3-e5018b1f1592)

### Importance of this Paper
This paper is significant in VLM history as it establishes the foundation of transformer-based models for vision tasks. The diagram below provides an overview of the model's architecture and workflow.

### Implementation and Validation
I have implemented the model and validated it using dummy inputs, as training transformers require a large amount of data. Below are the steps I followed to ensure the correctness of the dummy tests:

#### Creating an Instance of the Model
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
```

#### Dummy Input
```python
batch_size = 2  # Example batch size
dummy_input = torch.randn(batch_size, in_channels, image_size, image_size)
print("Dummy Input Image Shape:", dummy_input.shape)
```

#### Patch Embedding and Projection
```python
patch_embeddings = model.patch_embed_proj(dummy_input)
print("Patch Embeddings Shape:", patch_embeddings.shape)
expected_num_patches = (image_size // patch_size) ** 2
assert patch_embeddings.shape == (batch_size, expected_num_patches, embedding_dim), \
       f"Patch Embeddings shape is incorrect. Expected {(batch_size, expected_num_patches, embedding_dim)}, but got {patch_embeddings.shape}"
```

#### Positional Encoding
```python
# Create class token and prepend, then apply positional encoding
class_token = model.class_token.expand(batch_size, -1, -1)
embeddings_with_class_token = torch.cat((class_token, patch_embeddings), dim=1)
embeddings_with_pos = model.pos_embedding(embeddings_with_class_token)

print("Embeddings with Positional Encoding Shape:", embeddings_with_pos.shape)
expected_seq_len = expected_num_patches + 1  # +1 for class token
assert embeddings_with_pos.shape == (batch_size, expected_seq_len, embedding_dim), \
       f"Embeddings with Positional Encoding shape is incorrect. Expected {(batch_size, expected_seq_len, embedding_dim)}, but got {embeddings_with_pos.shape}"
```

#### Transformer Encoder Layers
```python
encoded_sequence = model.encoder_layers(embeddings_with_pos)
print("Output Shape after Transformer Encoder Layers:", encoded_sequence.shape)
assert encoded_sequence.shape == (batch_size, expected_seq_len, embedding_dim), \
       f"Encoder Output shape is incorrect. Expected {(batch_size, expected_seq_len, embedding_dim)}, but got {encoded_sequence.shape}"
```

#### Transformer Decoder Layer with Class Token Output
```python
output_logits = model.classifier_head(encoded_sequence)
print("Output Logits Shape from Classification Head:", output_logits.shape)
assert output_logits.shape == (batch_size, num_classes), \
       f"Classification Head output shape is incorrect. Expected {(batch_size, num_classes)}, but got {output_logits.shape}"
```

#### Full Forward Pass
```python
full_output_logits = model(dummy_input)
print("Output Logits Shape from Full Forward Pass:", full_output_logits.shape)
assert full_output_logits.shape == (batch_size, num_classes), \
       f"Full Forward Pass output shape is incorrect. Expected {(batch_size, num_classes)}, but got {full_output_logits.shape}"
```



