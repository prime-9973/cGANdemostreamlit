import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 定数定義
LATENT_DIM = 10
N_CLASSES = 10
IMG_SIZE = 28

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator クラス（学習時と一致させる必要あり）
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# モデルのロード（weights_only=False を明示）
@st.cache_resource
def load_generator_model():
    model = torch.load("GANgenerator.pth", map_location=device, weights_only=False)
    model.eval()
    return model

generator = load_generator_model()

# Streamlit インターフェース
st.title("Conditional GAN Image Generator")
st.write("手書き数字 (0～9) を選択して、画像を生成します。")

label = st.selectbox("生成したい数字を選んでください (0〜9):", list(range(N_CLASSES)))
n_images = st.slider("生成する画像の数", min_value=1, max_value=10, value=5)

# 生成ボタン
if st.button("画像を生成"):
    # ノイズとラベルの準備
    z = torch.randn(n_images, LATENT_DIM, device=device)
    labels = torch.full((n_images,), label, dtype=torch.long, device=device)

    # 画像生成
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    gen_imgs = (gen_imgs + 1) / 2  # [-1,1] → [0,1]

    # 画像表示
    fig, axs = plt.subplots(1, n_images, figsize=(2 * n_images, 2))
    if n_images == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        img = gen_imgs[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    st.pyplot(fig)
