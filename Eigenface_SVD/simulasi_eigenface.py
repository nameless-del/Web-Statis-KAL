import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

print("1. Mengunduh dataset wajah Olivetti (AT&T)...")
# Mengambil 400 foto wajah dari 40 orang berbeda (ukuran 64x64 piksel)
dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
faces = dataset.data  # Matriks berukuran (400 gambar, 4096 piksel)
h, w = 64, 64         # Resolusi tinggi dan lebar gambar

# Dalam Aljabar Linear, setiap gambar biasanya dijadikan vektor kolom.
# Jadi kita transpose matriksnya menjadi (4096 baris piksel x 400 kolom wajah)
A = faces.T 
print(f"   Bentuk Matriks Data (A): {A.shape} -> (Piksel x Jumlah Wajah)")

print("\n2. Menghitung Mean Face (Wajah Rata-rata)...")
# Menghitung rata-rata untuk setiap baris piksel
mean_face = np.mean(A, axis=1, keepdims=True)

# Mengurangi setiap wajah dengan wajah rata-rata (Centering)
A_centered = A - mean_face

print("\n3. Melakukan Dekomposisi SVD (A = U * Sigma * V^T)...")
# Kita gunakan Thin SVD (full_matrices=False) agar tidak memakan RAM raksasa
U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)

# KOLOM-KOLOM PADA MATRIKS U ADALAH EIGENFACES!
eigenfaces = U
print(f"   Bentuk Matriks Eigenfaces (U): {U.shape}")

# ==============================================================================
# BAGIAN VISUALISASI MATPLOTLIB
# ==============================================================================
print("\n4. Menampilkan Visualisasi...")

fig = plt.figure(figsize=(12, 6))
fig.canvas.manager.set_window_title('Simulasi Eigenface menggunakan SVD')

# -- Baris 1: Menampilkan Mean Face --
ax = plt.subplot(2, 5, 1)
ax.imshow(mean_face.reshape(h, w), cmap='gray')
ax.set_title("Wajah Rata-rata (\u03A8)")
ax.axis('off')

# -- Baris 1: Menampilkan 4 Eigenface Utama --
# Ini adalah representasi visual dari kolom-kolom pertama matriks U
for i in range(4):
    ax = plt.subplot(2, 5, i + 2)
    # Ambil kolom ke-i dari matriks U, lalu kembalikan ke bentuk 2D (64x64)
    eigenface_img = eigenfaces[:, i].reshape(h, w)
    ax.imshow(eigenface_img, cmap='bone') # Gunakan cmap 'bone' agar terlihat efek "hantu"
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')

# -- Baris 2: Simulasi Rekonstruksi Wajah --
# Mari kita ambil satu wajah acak dari dataset untuk direkonstruksi
target_face_centered = A_centered[:, 0]
target_face_asli = A[:, 0]

ax = plt.subplot(2, 5, 6)
ax.imshow(target_face_asli.reshape(h, w), cmap='gray')
ax.set_title("Wajah Target Asli")
ax.axis('off')

# Kita coba susun ulang wajah tersebut menggunakan jumlah Eigenface (k) yang berbeda-beda
# Semakin banyak Eigenface yang dipakai, semakin mirip dengan wajah asli!
k_values = [5, 20, 50, 150] 

for i, k in enumerate(k_values):
    # 1. Proyeksi: Cari bobot wajah ini di ruang Eigenface (W = U^T * x)
    U_k = eigenfaces[:, :k] 
    bobot = U_k.T @ target_face_centered
    
    # 2. Rekonstruksi: Susun kembali wajahnya (x' = U * W + Mean)
    rekonstruksi = (U_k @ bobot) + mean_face.flatten()
    
    ax = plt.subplot(2, 5, i + 7)
    ax.imshow(rekonstruksi.reshape(h, w), cmap='gray')
    ax.set_title(f"Rekonstruksi (k={k})")
    ax.axis('off')

plt.tight_layout()
plt.show()