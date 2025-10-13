import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# === Parametry ===
pixel_size_mm = 1.0
z_step_mm = 1.0

# === Output folder ===
output_dir = 'przekroje_boczne_all'
os.makedirs(output_dir, exist_ok=True)

# === Przetwarzaj wszystkie pliki .npy w podfolderach ===
for file_path in glob.glob('**/*.npy', recursive=True):
    # Skip files in /old folder
    if '/old/' in file_path or file_path.startswith('old/'):
        continue
    print(f"Przetwarzam plik: {file_path}")
    data = np.load(file_path)  # shape: (Z, Y, X)
    num_z, height, width = data.shape
    mid_x = width // 2
    profile = np.abs(data[:, :, mid_x])  # shape: (Z, Y)
    z = np.arange(0, num_z * z_step_mm, z_step_mm)
    y = np.arange(0, height * pixel_size_mm, pixel_size_mm)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(profile.T, cmap='hot', extent=[z[0], z[-1], y[-1], y[0]], aspect='auto')
    plt.title(f"Przekrój boczny pola (środek X = {mid_x})\n{file_path}")
    plt.xlabel("Z [mm]")
    plt.ylabel("Y [mm]")
    plt.colorbar(im, label="Amplituda")
    plt.tight_layout()
    # Build respectful output name (use only file name, no folders)
    base = os.path.splitext(os.path.basename(file_path))[0]
    # Remove 'przekroj_boczny_' prefix if already present
    if base.startswith('przekroj_boczny_'):
        base = base[len('przekroj_boczny_'):]
    out_path = os.path.join(output_dir, f"przekroj_boczny_{base}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Zapisano: {out_path}")
