import os
import cv2
import mediapipe as mp
import pandas as pd

# 1. Inisialisasi Mediapipe Hands dengan max_num_hands=2
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)

DATASET_PATH = 'dataset/'  # Pastikan folder dataset ada di sini
data_list = []

print("Memulai ekstraksi landmark (Mode 2 Tangan)...")

# 2. Loop melalui setiap folder alfabet (A-Z)
for label in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, label)
    
    if os.path.isdir(folder_path):
        print(f"Memproses folder: {label}")
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            # Ubah ke RGB untuk Mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            # Logika Ekstraksi 2 Tangan
            if results.multi_hand_landmarks:
                all_coords = []
                
                # Ambil koordinat dari semua tangan yang terdeteksi
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        all_coords.extend([lm.x, lm.y, lm.z])
                
                # Jika hanya 1 tangan terdeteksi (63 angka), 
                # tambahkan 63 angka nol (padding) untuk tangan kedua
                if len(all_coords) == 63:
                    all_coords.extend([0.0] * 63)
                
                # Pastikan hanya mengambil maksimal 2 tangan (126 koordinat)
                # Jika lebih atau kurang dari 126 karena eror deteksi, abaikan baris ini
                if len(all_coords) == 126:
                    all_coords.append(label)
                    data_list.append(all_coords)

# 3. Simpan ke CSV
# 126 kolom koordinat (21 titik * 3 sumbu * 2 tangan) + 1 kolom label
columns = [f'coord_{i}' for i in range(126)] + ['label']
df = pd.DataFrame(data_list, columns=columns)
df.to_csv('data_processed.csv', index=False)

print(f"\nSelesai! Berhasil mengekstrak {len(df)} data ke data_processed.csv")
print(f"Jumlah kolom: {len(df.columns)} (126 koordinat + 1 label)")