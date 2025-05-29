import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalTransformerEncoder(nn.Module):
    def __init__(self, d_model=32, nhead=4, dim_feedforward=64, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)  # Single output for gradient analysis

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        encoded = self.transformer_encoder(x)
        # Take the first token's output (like a [CLS] token)
        cls_token = encoded[:, 0, :]
        out = self.output_layer(cls_token)
        return out.squeeze(-1)  # (batch_size,) or scalar if batch_size=1

# Example usage for gradient analysis:
if __name__ == "__main__":
    model = MinimalTransformerEncoder()
    x = torch.randn(2, 10, 32, requires_grad=True)  # (batch, seq_len, d_model)
    output = model(x)
    print("Output shape:", output.shape)
    output.sum().backward()
    print("Input gradient shape:", x.grad.shape)
