from transformers import PreTrainedModel, PretrainedConfig
import torch
GRID_SIZE = 48



class ControlConfig(PretrainedConfig):
    def __init__(self, control_dim=1, embed_dim=768, nhead=4, nlayer=2, intermediate_dim=3072, grid_size=48,ntarget=1, **kwargs):
        super().__init__(**kwargs)
        self.control_dim = control_dim
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.intermediate_dim = intermediate_dim
        self.grid_size = grid_size
        self.ntarget = ntarget
        self.nlayer=nlayer




class ControlTransformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_proj = torch.nn.Linear(config.control_dim, config.embed_dim)
        self.pos_emb = torch.nn.Parameter(torch.randn(config.grid_size, config.embed_dim))
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.nhead, dim_feedforward=config.intermediate_dim)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=config.nlayer)
        
    def forward(self, control_seq):
        print(control_seq.shape)
        # control_seq: [B, 48, control_dim]
        x = self.input_proj(control_seq)               # [B, 48, embed_dim]
        x = x + self.pos_emb.unsqueeze(0)              # add positional encoding
        x = x.transpose(0, 1)                          # [48, B, embed_dim]
        x = self.encoder(x)
        x = x.transpose(0, 1)                          # [B, 48, embed_dim]
        pooled = x.mean(dim=1)                          # [B, embed_dim]
        return pooled
    
class ControlTransformerWrapperPTM(PreTrainedModel):
    config_class = ControlConfig
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ControlTransformer(
            config
        )
        self.output_head = torch.nn.Linear(config.embed_dim, config.ntarget)

    def forward(self, control_seq):
        # control_seq: [B, 48, control_dim]
        embedding = self.encoder(control_seq)
        output = self.output_head(embedding)
        return output



if __name__=="__main__":
        
    config = ControlConfig()
    model = ControlTransformerWrapperPTM(config)

    # Suppose control_dim = 1, batch size = 4
    batch_size = 4
    grid_size = 48
    control_dim = 1

    # Create a dummy batch of control inputs (e.g., metric weights per bar)
    dummy_controls = torch.randn(batch_size, grid_size, control_dim)  # [4, 48, 1]

    # Dummy ground truth syncopation scores (targets)
    dummy_targets = torch.randn(batch_size, 1)  # [4, 1]
    # Forward pass: predict syncopation scores

    
    pred_syncopation = model(dummy_controls)  # shape [4, 1]

    print("Predicted syncopation shape:", pred_syncopation.shape)
    print(pred_syncopation)

    # Define loss function
    loss_fn = torch.nn.MSELoss()

    # Compute loss between prediction and target
    loss = loss_fn(pred_syncopation, dummy_targets)

    print("Loss:", loss.item())

