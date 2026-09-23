import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA



class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10000 ),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(10000, 3000),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000 ,3000),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            nn.Linear(3000 , 10000),
            nn.BatchNorm1d(10000),
            nn.ReLU(),
            nn.Linear(10000 , output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.enc = Encoder(input_dim, latent_dim, dropout)
        self.dec = Decoder(latent_dim, output_dim)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z


def split_data(data, val_ratio=0.2, seed=42, return_indices=False):
    n = data.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)

    n_val = int(n * val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    if return_indices:
        return train_idx, val_idx
    return data[train_idx], data[val_idx]


def mse_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)



def compute_loss(x_true, x_recon):
    mse = mse_loss(x_recon, x_true)
    total_loss = mse 
    return total_loss


def evaluate(model, data_input, data_target, device):
    model.eval()
    with torch.no_grad():
        data_input = data_input.to(device)
        data_target = data_target.to(device)
        x_hat, _ = model(data_input)
        loss = compute_loss(data_target, x_hat)
    return loss.item()
  

def train(model, train_input, train_target, val_input, val_target,
          epochs, batch_size, lr, device="cuda"):
    model.to(device)

    dataset = torch.utils.data.TensorDataset(train_input, train_target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_in, batch_target in loader:
            batch_in = batch_in.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch_in)
            loss = compute_loss(batch_target, x_hat)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * batch_in.size(0)

        train_loss = total_loss / len(dataset)
        val_loss = evaluate(model, val_input, val_target, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - "
            f"val_loss: {val_loss:.4f} - lr: {current_lr:.6f}"
        )

    return model



def load_gene_matrix(
    expr_path=r"C:\Users\tolga\Desktop\CGGA.mRNAseq_693.Read_Counts-genes.20220620.txt"
):
    rna = pd.read_csv(expr_path, index_col=0, sep="\t").fillna(0)

    valid_genes = (rna.std(axis=1).values > 0)
    rna = rna.iloc[valid_genes]

    gene_names = rna.index
    sample_names = np.array(rna.columns)  # Hasta/sample isimleri

    return rna.values.T, gene_names, sample_names  


def zscore_normalize(train_data, val_data, min_std=1e-2):
    gene_std = train_data.std(dim=0, keepdim=True)
    keep_mask = (gene_std.squeeze() > min_std)

    train_data = train_data[:, keep_mask]
    val_data = val_data[:, keep_mask]
    gene_mean = train_data.mean(dim=0, keepdim=True)
    gene_std = train_data.std(dim=0, keepdim=True)

    train_norm = (train_data - gene_mean) / (gene_std + 1e-8)
    val_norm = (val_data - gene_mean) / (gene_std + 1e-8)
    return train_norm, val_norm, keep_mask


print(device := "cuda" if torch.cuda.is_available() else "cpu")

# Çıktı klasörü: Desktop
OUT_DIR = r"C:\Users\tolga\Desktop"


def save_fig(path, fig=None, **kwargs):
    """
    Grafiği kaydeder. Hedef dosya zaten varsa (önceki çalıştırmadan kalmış, kilitli
    veya OneDrive placeholder olabilir) 'w+b' ile üzerine yazma [Errno 22] Invalid
    argument verebildiğinden, önce mevcut dosyayı silmeyi deneyip taze yazar.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
    (fig or plt).savefig(path, **kwargs)






if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 500
    dropout = 0
    lr = 1e-4
    epochs = 100
    batch_size = 35

    gene_matrix, gene_names, sample_names = load_gene_matrix()
    print("Gene matrix shape (hasta x gen):", gene_matrix.shape)
    print("Sample isimleri kaydedildi")

    # Log transform
    gene_matrix = np.log2(gene_matrix + 1)
    print("After log transform shape:", gene_matrix.shape)

    data = torch.tensor(gene_matrix, dtype=torch.float32)

    train_indices, val_indices = split_data(data, val_ratio=0.2, return_indices=True)
    train_data = data[train_indices]
    val_data = data[val_indices]
    train_samples = sample_names[train_indices.numpy()]
    val_samples = sample_names[val_indices.numpy()]

    train_data, val_data, keep_mask = zscore_normalize(train_data, val_data)
    gene_names = np.asarray(gene_names[keep_mask.numpy()])
    print(f"Train: {train_data.shape[0]} hasta, Val: {val_data.shape[0]} hasta, Gen: {train_data.shape[1]}")

    # ------------------------------------------------------------------
    # CGGA genlerinden rastgele 20000 gen seç (sabit seed)
    # ------------------------------------------------------------------
    N_GENES = 20000
    rng = np.random.default_rng(42)
    sel_idx = rng.choice(train_data.shape[1], size=N_GENES, replace=False)
    sel_idx_t = torch.tensor(sel_idx)

    train_data = train_data[:, sel_idx_t]
    val_data = val_data[:, sel_idx_t]
    gene_names = gene_names[sel_idx]
    print(f"Rastgele seçilen gen sayısı: {train_data.shape[1]}")

    train_input = train_data
    train_target = train_data

    val_input = val_data
    val_target = val_data

    input_dim = train_data.shape[1]
    print(f"Input şekli (train): {train_input.shape}")   # (hasta_sayısı, gen_sayısı)
    print(f"Output şekli (train): {train_target.shape}") # (hasta_sayısı, gen_sayısı)

    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout, output_dim=input_dim)
    train(
        model, train_input, train_target, val_input, val_target,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device,
    )

    # Modeli kaydet
    weights_path = os.path.join(OUT_DIR, "autoencoder_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print("Model weights kaydedildi:", weights_path)

    model.eval()
    with torch.no_grad():
        full_input = train_input  
        reconstructed, latent = model(full_input.to(device))
        print("Latent shape:", latent.shape)
        print("Reconstructed shape:", reconstructed.shape)

        # MSE hesapla
        final_mse = F.mse_loss(reconstructed, full_input.to(device))
        print(f"Final MSE Loss: {final_mse.item():.6f}")

        # Reconstruction scatter plot
        x_true = full_input.cpu().numpy().flatten()
        y_true = reconstructed.cpu().numpy().flatten()
        r2 = r2_score(x_true, y_true)
        print(f"R² Score: {r2:.4f}")

        value_dir = os.path.join(OUT_DIR, "value")
        os.makedirs(value_dir, exist_ok=True)
        full_input_flatten_path = os.path.join(value_dir, "full_input_flatten.csv")
        reconstructed_flatten_path = os.path.join(value_dir, "reconstructed_flatten.csv")
        pd.DataFrame({"value": x_true}).to_csv(full_input_flatten_path, index=False)
        pd.DataFrame({"value": y_true}).to_csv(reconstructed_flatten_path, index=False)

        print("Full input flatten kaydedildi:", full_input_flatten_path)
        print("Reconstructed flatten kaydedildi:", reconstructed_flatten_path)

        # Çok fazla nokta varsa plot için alt-örnekle (plot hızlı olsun; CSV'ler tam veriyi içerir)
        x_plot, y_plot = x_true, y_true
        max_points = 200000
        if x_plot.size > max_points:
            sel = np.random.default_rng(42).choice(x_plot.size, size=max_points, replace=False)
            x_plot, y_plot = x_plot[sel], y_plot[sel]

        recon_plot_path = os.path.join(OUT_DIR, "reconstruction_plot.png")
        plt.figure(figsize=(6, 6))
        plt.scatter(x_plot, y_plot, alpha=0.5, s=10)
        plt.xlabel("input değeri")

        plt.ylabel("reconstructed değeri")
        plt.tight_layout()
        save_fig(recon_plot_path, dpi=150)
        plt.close()
        print("Grafik kaydedildi:", recon_plot_path)
        latent_np = latent.cpu().numpy()

    # Latent temsilleri kaydet
    full_samples = train_samples
    latent_path = os.path.join(OUT_DIR, "latent_representations.csv")
    latent_df = pd.DataFrame(
        latent_np, index=full_samples,
        columns=[f"z{i}" for i in range(latent_np.shape[1])],
    )
    latent_df.to_csv(latent_path)
    print("Latent temsiller kaydedildi:", latent_path)

    # ------------------------------------------------------------------
    # Aynı latent boyutta PCA ile reconstruction (autoencoder ile kıyaslama)
    # ------------------------------------------------------------------
    train_input_np = full_input.cpu().numpy()

    pca = PCA(n_components=latent_dim, random_state=42)
    pca_latent = pca.fit_transform(train_input_np)
    pca_reconstructed = pca.inverse_transform(pca_latent)

    pca_mse = np.mean((train_input_np - pca_reconstructed) ** 2)
    pca_x_true = train_input_np.flatten()
    pca_y_true = pca_reconstructed.flatten()
    pca_r2 = r2_score(pca_x_true, pca_y_true)
    print(f"PCA Final MSE Loss: {pca_mse:.6f}")
    print(f"PCA R² Score: {pca_r2:.4f}")

    pca_full_input_flatten_path = os.path.join(value_dir, "pca_full_input_flatten.csv")
    pca_reconstructed_flatten_path = os.path.join(value_dir, "pca_reconstructed_flatten.csv")
    pd.DataFrame({"value": pca_x_true}).to_csv(pca_full_input_flatten_path, index=False)
    pd.DataFrame({"value": pca_y_true}).to_csv(pca_reconstructed_flatten_path, index=False)
    print("PCA full input flatten kaydedildi:", pca_full_input_flatten_path)
    print("PCA reconstructed flatten kaydedildi:", pca_reconstructed_flatten_path) 

    pca_x_plot, pca_y_plot = pca_x_true, pca_y_true
    if pca_x_plot.size > max_points:
        sel = np.random.default_rng(42).choice(pca_x_plot.size, size=max_points, replace=False)
        pca_x_plot, pca_y_plot = pca_x_plot[sel], pca_y_plot[sel]

    pca_recon_plot_path = os.path.join(OUT_DIR, "pca_reconstruction_plot.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(pca_x_plot, pca_y_plot, alpha=0.5, s=10)
    plt.xlabel("input değeri")
    plt.ylabel("PCA reconstructed değeri")
    plt.tight_layout()
    save_fig(pca_recon_plot_path, dpi=150)
    plt.close()
    print("PCA grafik kaydedildi:", pca_recon_plot_path)

    pca_latent_path = os.path.join(OUT_DIR, "pca_latent_representations.csv")
    pca_latent_df = pd.DataFrame(
        pca_latent, index=full_samples,
        columns=[f"pc{i}" for i in range(pca_latent.shape[1])],
    )
    pca_latent_df.to_csv(pca_latent_path)
    print("PCA latent temsiller kaydedildi:", pca_latent_path)





class FClayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

