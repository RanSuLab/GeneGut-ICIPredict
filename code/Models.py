import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from torch import nn, optim
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.attention(query, key, value, need_weights=True)
        return self.layer_norm(attn_output), attn_weights

class VAE_prior_knowledge(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, reference_feature_dim):
        super(VAE_prior_knowledge, self).__init__()

   
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, latent_dim) 

        self.decoder_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_bn1 = nn.BatchNorm1d(hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
        self.decoder_bn2 = nn.BatchNorm1d(input_dim)

        self.class_fc1 = nn.Linear(latent_dim, 128)
        self.class_bn1 = nn.BatchNorm1d(128)
        self.class_fc2 = nn.Linear(128, 64)
        self.class_bn2 = nn.BatchNorm1d(64)
        self.fc_classifier = nn.Linear(64, 1)

        self.reference_fc = nn.Linear(reference_feature_dim, latent_dim)
        self.biology_constraint = nn.Linear(68170, 1)
        self.attention = MultiHeadAttention(embed_dim=latent_dim, num_heads=4)

    def encode(self, x, reference_features):
        reference_h = F.relu(self.reference_fc(reference_features)) 

        x_h1 = self.encoder_bn1(self.encoder_fc1(x))  
        x_h2 = self.encoder_fc2(x_h1)  
        atten_in = x_h2.unsqueeze(1).permute(1, 0, 2)  
        reference_h = reference_h.unsqueeze(1).expand(-1, x.shape[0], -1) 

        attn_output, attn_weights = self.attention(atten_in, reference_h, reference_h)
        attn_infor = attn_output.permute(1, 0, 2).squeeze(1)

        return x_h2, attn_infor, reference_h

    def reparameterize(self, mu):
        epsilon = torch.randn(mu.size(0), mu.size(1), device=mu.device)  
        latent = mu + epsilon
        return latent

    def decode(self, z, attn_infor):
        lambda_ = 0.7
        output = torch.cat([lambda_ * z, (1 - lambda_) * attn_infor], dim=1)   
        deco_h1 = F.relu(self.decoder_bn1(self.decoder_fc1(output)))
        deco_h2 = self.decoder_bn2(self.decoder_fc2(deco_h1))
        return torch.sigmoid(deco_h2)

    def classify(self, classify_x):
        classify1 = torch.relu(self.class_bn1(self.class_fc1(classify_x)))
        classify2 = torch.relu(self.class_bn2(self.class_fc2(classify1)))
        y_pred = torch.sigmoid(self.fc_classifier(classify2)).squeeze(-1)
        return y_pred

    def forward(self, x, reference_features):
        mu, attn_infor, reference_h = self.encode(x, reference_features.detach())
        z = self.reparameterize(mu)  
        recon_x = self.decode(z, attn_infor)  
        clssify_output = self.classify(mu)

        return recon_x, mu, clssify_output