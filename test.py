import torch
from lit_gpt.diffmodel import TransEncoder, Config

# Load the model configuration
config = Config.from_name("Diff_LLaMA_761M")  # Replace with your specific model config if different

# Instantiate the model
model = TransEncoder(config)
model.to("cuda", dtype=torch.bfloat16)  # Match the rope cache dtype

# Create a random input tensor (batch size 1, sequence length 128, values between 0 and 31999)
input_tensor = torch.randint(0, 32000, (4, 4096))
input_tensor = input_tensor.to("cuda")

# Create target labels (shifted input for next token prediction)
target_labels = torch.randint(0, 32000, (1, 128))
target_labels = target_labels.to("cuda")

# Run the forward pass
output_logits = model(input_tensor)

# Compute cross-entropy loss
# Reshape logits and labels for loss computation
logits_flat = output_logits.view(-1, output_logits.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
labels_flat = target_labels.view(-1)  # Shape: [batch_size * seq_len]

# Compute the loss
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits_flat, labels_flat)

# Move to CPU for printing
output_logits = output_logits.to("cpu")
loss = loss.to("cpu")

# Print the output shape and loss
print(f"Output logits shape: {output_logits.shape}")  # Expected: torch.Size([1, 128, 32000])
print(f"Loss: {loss.item():.4f}")