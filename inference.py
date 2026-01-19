from libs import *
from vae_model import VAE

imgae_path = "./keqing.jpg"
model_path = "vae_model.pt"
optim_path = "./vae_optimizer.pt"
device = device


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


def scale(x):
    return ((x / 255.0) * 2) - 1

vae_model = VAE(
    down_latent_channels_sequence=[3, 64, 128, 256],
    up_latent_channels_sequence=[8, 64, 128, 3],
    num_resnet_layers=2,
    latent_project_channels=8*2,
    device=device
)
vae_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
vae_model.eval()

image = cv2.imread(imgae_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.expand_dims(image, axis=0)
image = torch.tensor(image, device=device)
image = scale(image)
image = image.permute(0, 3, 1, 2)
print("Image.shape:", image.shape)

show_images(
    unscale_image_one_to_neg_one(image.permute(0, 2, 3, 1).cpu().numpy()),
    timeout=3600
)

with torch.no_grad():
    x, mean, log_var = vae_model(image)
    z_latent, _, _ = vae_model.encode_with_scale_n01(x)

print("X.shape:", x.shape)
print("Mean of Z Latent ->", torch.mean(z_latent, dim=(1, 2, 3)))
print("Variance of Z Latent ->", torch.var(z_latent, dim=(1, 2, 3)))

show_images(
    unscale_image_one_to_neg_one(x.permute(0, 2, 3, 1).cpu().numpy()),
    timeout=3600
)