from model.models import PointNetFeatureExtractor, Generator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ViewTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=2, num_heads=4, dim_feedforward=512, dropout=0.1):
        super(ViewTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tensor of shape (batch_size, input_dim)
        """
        # Transpose to shape (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        # Pool over sequence length
        pooled = x.mean(dim=0)  # Shape: (batch_size, input_dim)
        return pooled

class MultiViewHRTFPredictionModel(nn.Module):
    def __init__(self):
        super(MultiViewHRTFPredictionModel, self).__init__()
        self.feat = PointNetFeatureExtractor()
        self.view_transformer = ViewTransformer(input_dim=1024)
        
        # Embedding for 129 frequency bins
        self.freq_enc = nn.Embedding(129, 16)
        
        # Single regression pathway for both magnitude and phase
        self.mid_channel = 2048 + 16  # ear features + frequency embedding
        self.regression = nn.Sequential(
            nn.Linear(self.mid_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, 793 * 2 * 2)  # [positions, ears, mag/phase]
        )
        
    def forward(self, point_clouds):
        """
        Process point clouds and predict HRTF for all frequencies.
        
        Args:
            point_clouds: [batch_size, 2, num_views, num_points, 3]
        Returns:
            HRTF predictions: [batch_size, 793, 2, 258]
            (batch, positions, ears, frequencies[mag+phase])
        """
        batch_size = point_clouds.shape[0]
        num_freqs = 129  # Number of frequency bins
        
        # Extract ear features once
        left_pc = point_clouds[:, 0]
        right_pc = point_clouds[:, 1]
        both_point_clouds = [left_pc, right_pc]
        features = []
        
        for ear_point_clouds in both_point_clouds:
            view_features = []
            for i in range(ear_point_clouds.shape[1]):
                pc = ear_point_clouds[:, i].permute(0, 2, 1).float()
                view_feat = self.feat(pc)
                view_feat = view_feat.squeeze(-1)
                view_features.append(view_feat)
            
            ear_features = torch.stack(view_features, dim=1)
            pooled_features = self.view_transformer(ear_features)
            features.append(pooled_features)
        
        # Combine ear features
        ear_features = torch.cat(features, dim=1)  # [batch_size, 2048]
        
        # Process each frequency
        all_freq_predictions = []
        for freq_idx in range(num_freqs):
            # Get frequency embedding
            frequency = torch.full((batch_size,), freq_idx, 
                                device=point_clouds.device, 
                                dtype=torch.long)
            freq_encoding = self.freq_enc(frequency)
            
            # Combine features
            combined_features = torch.cat([ear_features, freq_encoding], dim=1)
            # Predict magnitude and phase together
            predictions = self.regression(combined_features)
            predictions = predictions.view(batch_size, 793, 2, 2)  # [batch, positions, ears, mag/phase]
            all_freq_predictions.append(predictions)
        
        # Stack all frequency predictions
        stacked_predictions = torch.stack(all_freq_predictions, dim=3)  # [batch, 793, 2, num_freqs, 2]
        
        # Reshape to desired output format [batch, 793, 2, 258]
        hrtf_predictions = stacked_predictions.permute(0, 1, 2, 4, 3).reshape(batch_size, 793, 2, -1)
        
        return hrtf_predictions

if __name__ == "__main__":
    model = MultiViewHRTFPredictionModel()
    point_clouds = torch.randn(4, 2, 5, 1000, 3)
    hrtf_predictions = model(point_clouds)
    print(hrtf_predictions.shape)  # Should output: torch.Size([4, 793, 2, 258])