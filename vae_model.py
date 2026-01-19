from libs import *

class ResidualConvolutionNet(nn.Module):
    def __init__(self, in_channels, out_channels, device=None):
        super().__init__()
        in_group, out_group = ((32 if in_channels > 32 else 1), (32 if in_channels > 32 else 1))
        self.gn1 = nn.GroupNorm(in_group, in_channels, device=device)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True, device=device)
        self.silu1 = nn.SiLU()
        self.gn2 = nn.GroupNorm(out_group, out_channels, device=device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True, device=device)
        self.silu2 = nn.SiLU()
        self.residual_projection = nn.Identity() if in_channels == out_channels else \
            nn.Conv2d(in_channels, out_channels, kernel_size=1, device=device)

    def forward(self, x: torch.Tensor):
        x_skip = x
        x = self.silu1(self.conv1(self.gn1(x)))
        x = self.silu2(self.conv2(self.gn2(x)))
        x = x + self.residual_projection(x_skip)
        return x


class ConvolutionEncoderVAE(nn.Module):
    def __init__(self, channels_sequence=[3, 64, 128, 256], num_resnet_layers=3, latent_project_channels=3*2, device=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels_sequence)-1):
            self.layers.append(nn.Conv2d(channels_sequence[i], channels_sequence[i+1], kernel_size=2, stride=2, bias=True, device=device))
            for _ in range(num_resnet_layers):
                self.layers.append(ResidualConvolutionNet(channels_sequence[i+1], channels_sequence[i+1], device=device))
        self.latent_projection = nn.Conv2d(channels_sequence[-1], latent_project_channels, kernel_size=1, bias=True, device=device)
        self.channels_sequence = channels_sequence
        self.num_resnet_layers = num_resnet_layers
        self.latent_project_channels = latent_project_channels
        self.device = device

    def forward(self, x: torch.Tensor, gaussian_noise: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        x = self.latent_projection(x)
        mean, log_var = x.split(self.latent_project_channels//2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        std = torch.exp(0.5 * log_var)
        z_latent = mean + (std * gaussian_noise)
        return z_latent, mean, log_var


class ConvolutionDecoderVAE(nn.Module):
    def __init__(self, channels_sequence=[3, 64, 64, 3], num_resnet_layers=3, device=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(channels_sequence)-1):
            self.layers.append(nn.Upsample(scale_factor=2))
            self.layers.append(nn.Conv2d(channels_sequence[i], channels_sequence[i+1], kernel_size=3, padding=1, bias=True, device=device))
            for _ in range(num_resnet_layers):
                self.layers.append(ResidualConvolutionNet(channels_sequence[i+1], channels_sequence[i+1], device=device))
        self.channels_sequence = channels_sequence
        self.num_resnet_layers = num_resnet_layers
        self.device = device

    def forward(self, z_latent: torch.Tensor):
        for layer in self.layers:
            z_latent = layer(z_latent)
        return z_latent


class VAE(nn.Module):
    def __init__(self, down_latent_channels_sequence=[3, 64, 128, 256, 512],
                 up_latent_channels_sequence=[3, 64, 128, 256, 3], num_resnet_layers=3, latent_project_channels=3*2, device=None):
        super().__init__()
        self.vae_encoder = ConvolutionEncoderVAE(channels_sequence=down_latent_channels_sequence, num_resnet_layers=num_resnet_layers,
                                                 latent_project_channels=latent_project_channels, device=device)
        self.latent_scaler_n01 = nn.Parameter(torch.tensor(1.0, device=device))
        self.vae_decoder = ConvolutionDecoderVAE(channels_sequence=up_latent_channels_sequence, num_resnet_layers=num_resnet_layers,
                                                 device=device)
        self.latent_project_channels = latent_project_channels
        self.device = device

    def encode_with_scale_n01(self, x: torch.Tensor, epsilon=1e-6, in_forward_pass=False):
        with torch.no_grad():
            gaussian_noise_wh_compute = [x.shape[-1], x.shape[-2]]
            for _ in range(len(self.vae_encoder.channels_sequence)-1):
                gaussian_noise_wh_compute[0] //= 2
                gaussian_noise_wh_compute[1] //= 2

        gaussian_noise = torch.randn(
            size=(x.shape[0], self.latent_project_channels//2, gaussian_noise_wh_compute[0], gaussian_noise_wh_compute[1]),
            device=x.device
        )

        z_latent, mean, log_var = self.vae_encoder(x, gaussian_noise)
        if in_forward_pass:
            z_latent *= (self.latent_scaler_n01.detach() + epsilon)
        else:
            z_latent *= (self.latent_scaler_n01 + epsilon)
        return z_latent, mean, log_var

    def unscale_n01(self, x: torch.Tensor, epsilon=1e-6, in_forward_pass=False):
        if in_forward_pass:
            return x / (self.latent_scaler_n01.detach() + epsilon)
        else:
            return x / (self.latent_scaler_n01 + epsilon)

    def forward(self, x: torch.Tensor):
        z_latent, mean, log_var = self.encode_with_scale_n01(x, in_forward_pass=True)
        z_latent = self.unscale_n01(z_latent, in_forward_pass=True)
        x = self.vae_decoder(z_latent)
        return x, mean, log_var
    

if __name__ == "__main__":
    model = VAE(
        down_latent_channels_sequence=[3, 64, 128, 256, 512],
        up_latent_channels_sequence=[16, 64, 128, 256, 3],
        num_resnet_layers=2,
        latent_project_channels=16*2,
        device=device
    )
    x = torch.rand(size=(2, 3, 256, 256), device=device)
    out, mean, log_var = model(x)
    print("Out.shape:", out.shape)
    print("Mean.shape:", mean.shape)
    print("Log_var.shape:", log_var.shape)