import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Tentukan path file
CSV_PATH = 'data_processed.csv'
MODEL_DIR = 'models'
MODEL_NAME = 'model_bisindo.pkl'

def train_bisindo_model():
    # 2. Cek apakah file CSV hasil ekstraksi ada
    if not os.path.exists(CSV_PATH):
        print(f"Eror: File '{CSV_PATH}' tidak ditemukan!")
        print("Pastikan kamu sudah menjalankan 'python src/extract_features.py' terlebih dahulu.")
        return

    # 3. Load data
    print("Membaca data dari CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # X adalah fitur (126 kolom koordinat), y adalah label (kolom terakhir)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 4. Bagi data menjadi Training dan Testing (80:20)
    # stratify=y memastikan distribusi huruf A-Z seimbang di kedua data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Inisialisasi dan Latih Model
    print(f"Sedang melatih model dengan {len(X_train)} data...")
    # Random Forest cocok untuk pola koordinat yang kompleks
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Akurasi Model: {accuracy * 100:.2f}%")
    print("-" * 30)
    print("Laporan Klasifikasi per Huruf:")
    print(classification_report(y_test, y_pred))

    # 7. Simpan Model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    model_full_path = os.path.join(MODEL_DIR, MODEL_NAME)
    with open(model_full_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Sukses! Model berhasil disimpan di: {model_full_path}")

if __name__ == "__main__":
    train_bisindo_model()