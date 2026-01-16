import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from diffusion_policy.common.pytorch_util import dict_apply
class ImageSeqVAE(nn.Module):
    def __init__(self,obs_horizon,pre_horizon,key_horizon,obs_encoder, latent_dim):
        super(ImageSeqVAE, self).__init__()
        img_channels=3
        self.hidden_dim=latent_dim
        latent_dim=512
        self.obs_horizon=obs_horizon
        self.pre_horizon=pre_horizon
        self.key_horizon=key_horizon
        self.obs_encoder=obs_encoder
        self.img_channels = img_channels
        self.latent_dim = latent_dim
        self.num_layers=2
        self.cnn_features=None
        self.latent_channels=3
        self.latent_H=4
        self.latent_W=4
        self.H, self.W=96,96
        # Encoder (3D CNN + LSTM)

        # Latent space projection
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)  # Log variance

        # Decoder (LSTM + 3D Transposed CNN)
        self.key_decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.pre_decoder_lstm= nn.LSTM(
            input_size=latent_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.key_decoder_fc = nn.Linear(self.hidden_dim, self.latent_channels * self.H * self.W)  # Reshape to [B, 128, 4, 4]
        self.pre_decoder_fc = nn.Linear(self.hidden_dim, self.latent_channels * self.H * self.W)
        self.key_decoder_cnn = nn.Sequential(
            nn.ConvTranspose3d(self.latent_channels, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, img_channels, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.Sigmoid()  # Pixel values in [0, 1]
        )
        self.pre_decoder_cnn = nn.Sequential(
            nn.ConvTranspose3d(self.latent_channels, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, img_channels, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), output_padding=(0, 1, 1)),
            nn.Sigmoid()  # Pixel values in [0, 1]
        )

    # def encode(self, x):
    #     # x: [B, C, T, H, W]
    #     # import pdb;pdb.set_trace()
        
    #     # self.target_shape=
    #     cnn_features = self.encoder_cnn(x)  # [B, 128, seq_len, 4, 4]
    #     batch_size = cnn_features.shape[0]
    #     cnn_features = cnn_features.permute(0, 2, 1, 3, 4).contiguous()  # [B, seq_len, 128, 4, 4]
    #     self.cnn_features=cnn_features
    #     cnn_features = cnn_features.view(batch_size, self.obs_horizon, -1)  # [B, seq_len, 128*4*4]

    #     # LSTM to aggregate temporal features
    #     _, (h_n, _) = self.encoder_lstm(cnn_features)  # h_n: [4, B, hidden_dim] (2 layers, bidirectional)
    #     # h_n = h_n.view(2, 1, batch_size, -1)  # [2, 2, B, hidden_dim] (layers, directions)
    #     # h_n = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)  # [B, 2*hidden_dim] (last layer, bidirectional)
    #     h_n=h_n.view(batch_size,-1)
    #     mu = self.fc_mu(h_n)  # [B, latent_dim]
    #     logvar = self.fc_logvar(h_n)  # [B, latent_dim]
    #     return mu, logvar
    def encode(self, x):
        # x: [B, C, T, H, W]
        
        
        obs_feature=self.obs_encoder(x)
        mu=self.fc_mu(obs_feature)
        logvar=self.fc_logvar(obs_feature)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def key_decode(self, z):
        # z: [B, latent_dim]
        # import pdb;pdb.set_trace()
        
        z = z.unsqueeze(1).repeat(1, self.key_horizon, 1)  # [B, seq_len, latent_dim]

        # LSTM to generate temporal features
        lstm_out, _ = self.key_decoder_lstm(z)  # [B, seq_len, hidden_dim]
        cnn_features = self.key_decoder_fc(lstm_out)  # [B, seq_len, 128*4*4]
        cnn_features = cnn_features.view(8, self.key_horizon, self.latent_channels, self.H, self.W)
        # cnn_features = cnn_features.permute(0, 2, 1, 3, 4)  # [B, 128, seq_len, 4, 4]
        
        # 3D CNN to reconstruct frames
        # recon = self.key_decoder_cnn(cnn_features)  # [B, C, T, H, W]
        # recon = F.interpolate(recon, size=(self.key_horizon,self.H,self.W), mode='trilinear', align_corners=False)
        return cnn_features

    # def pre_decode(self, z):
    #     # z: [B, latent_dim]
    #     batch_size = z.shape[0]
    #     z = z.unsqueeze(1).repeat(1, self.pre_horizon, 1)  # [B, seq_len, latent_dim]

    #     # LSTM to generate temporal features
    #     lstm_out, _ = self.pre_decoder_lstm(z)  # [B, seq_len, hidden_dim]
    #     cnn_features = self.pre_decoder_fc(lstm_out)  # [B, seq_len, 128*4*4]
    #     cnn_features = cnn_features.view(batch_size, self.pre_horizon, self.latent_channels, self.H, self.W)
    #     cnn_features = cnn_features.permute(0, 2, 1, 3, 4)  # [B, 128, seq_len, 4, 4]

    #     # 3D CNN to reconstruct frames
    #     recon = self.pre_decoder_cnn(cnn_features)  # [B, C, T, H, W]
    #     recon= F.interpolate(recon, size=(self.pre_horizon,self.H, self.W), mode='trilinear', align_corners=False)
    #     return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        key_img = self.key_decode(z)
        pre_img = self.pre_decode(z)
        return key_img, pre_img,mu, logvar