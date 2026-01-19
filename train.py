from libs import *
from vae_model import VAE

IMG_SIZE = 256
batch_size = 32
lr = 0.0001
num_epochs = 2



# --------------------- Load Data --------------------------

def get_batch(datasets, batch_size, idx=0):
    batch = [datasets[i] for i in range(idx*batch_size, (idx+1)*batch_size)]
    return torch.stack([x[0] for x in batch], dim=0)


def show_images(images, timeout=10):  # timeout tính bằng giây
    cols = images.shape[0]
    N = len(images)
    rows = math.ceil(N / cols)

    fig = plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(N):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.tight_layout()

    # Hiển thị ảnh không blocking
    plt.show(block=False)

    # Chờ timeout giây
    plt.pause(timeout)

    # Đóng cửa sổ tự động
    plt.close(fig)


def unscale_image_one_to_neg_one(x):
    x = (x + 1) / 2
    x = x * 255
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)

# Download latest version
path = kagglehub.dataset_download("splcher/animefacedataset")
print(f"Images path -> {path}")
data_transforms = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Scales data into [0,1]
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
]
anime_datasets = ImageFolder(root=path, transform=transforms.Compose(data_transforms))
images_loaded = get_batch(anime_datasets, 5, 5)
print(f"Images shape -> {images_loaded.shape}")
show_images(unscale_image_one_to_neg_one(images_loaded.permute(0, 2, 3, 1).cpu().numpy()), timeout=10)



# ------------------------ Training ----------------------------

# test and init model
model = VAE(
    down_latent_channels_sequence=[3, 64, 128, 256],
    up_latent_channels_sequence=[8, 64, 128, 3],
    num_resnet_layers=2,
    latent_project_channels=8*2,
    device=device
)
x = torch.randn(size=(2, 3, IMG_SIZE, IMG_SIZE), device=device)
out, mean, log_var = model(x)
print("Out.shape:", out.shape)
print("Mean.shape:", mean.shape)
print("Log_var.shape:", log_var.shape)

train_loader = torch.utils.data.DataLoader(
    anime_datasets,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True
)
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=lr,
    weight_decay=0.01
)
optimizer_for_scaler = torch.optim.SGD(
    params=[model.latent_scaler_n01],
    lr=1e-5,
    weight_decay=0.0
)

kl_loss_beta = 0.00025
for epoch in range(num_epochs):
    for step, (x, _) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        out, mean, log_var = model(x)
        reconstructed_loss = torch.nn.functional.mse_loss(out, x, reduction="sum")
        kl_loss = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=(1, 2, 3))).sum()
        loss = reconstructed_loss + (kl_loss * kl_loss_beta)
        loss.backward()
        optimizer.step()

        # huấn luyện hệ số scale n01
        model.vae_encoder.eval()
        model.vae_encoder.requires_grad_(False)
        optimizer_for_scaler.zero_grad()
        z_latent_with_scale_n01, mean, log_var = model.encode_with_scale_n01(x)
        mean, var = torch.mean(z_latent_with_scale_n01, dim=(1, 2, 3)), torch.var(z_latent_with_scale_n01, dim=(1, 2, 3))
        mean_loss = torch.nn.functional.l1_loss(mean, torch.zeros(size=mean.shape, device=device), reduction="sum")
        var_loss = torch.nn.functional.l1_loss(var, torch.ones(size=var.shape, device=device), reduction="sum")
        scale_loss = mean_loss + var_loss
        scale_loss.backward()
        optimizer_for_scaler.step()
        model.vae_encoder.train()
        model.vae_encoder.requires_grad_(True)

        print(f"\rEpoch {epoch+1}/{num_epochs} and step {step+1}/{len(train_loader)} with loss {loss.item()}, N(0,1)=[{torch.mean(mean)},{torch.mean(var)}],loss={scale_loss.item()},scale_val={model.latent_scaler_n01.item()}", end="")

        if (step+1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                out = model(x[:1, :, :, :])[0]
            pair = torch.cat([x[:1, :, :, :], out], dim=0)
            show_images(unscale_image_one_to_neg_one(pair.permute(0, 2, 3, 1).cpu().numpy()), timeout=1)
            model.train()