import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import pandas as pd
import warnings

# Mengabaikan peringatan agar terminal bersih
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="BISINDO Pro Detector", layout="wide")

@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'model_bisindo.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# --- 2. INISIALISASI MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands_live = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
hands_upload = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- 3. SIDEBAR ---
st.sidebar.title("⚙️ Kontrol Panel")
mode = st.sidebar.radio("Pilih Mode Input:", ("Live Webcam", "Upload Foto"))

st.sidebar.divider()
st.sidebar.subheader("✨ Image Enhancement")
brightness = st.sidebar.slider("Kecerahan (Brightness)", -100, 100, 0)
contrast = st.sidebar.slider("Kontras (Contrast)", 0.5, 3.0, 1.0)

# --- 4. TAMPILAN UTAMA ---
st.title("🤟 BISINDO Recognition System")
st.write("---")

if model is None:
    st.error("⚠️ Model tidak ditemukan! Harap jalankan 'train_model.py' terlebih dahulu.")
else:
    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("📊 Hasil Prediksi")
        prediction_placeholder = st.empty()
        
        st.write("---")
        with st.expander("📖 Panduan & Tips", expanded=True):
            st.markdown("""
            * **Brightness**: Gunakan jika kamera terasa gelap.
            * **Posisi**: Jarak tangan ke kamera idealnya ±70cm.
            * **Mode 2 Tangan**: Mendukung deteksi satu atau dua tangan sekaligus.
            """)
        st.success("Model Status: **Active**")

    with col_left:
        if mode == "Live Webcam":
            run_cam = st.toggle('Nyalakan Kamera', value=True)
            video_frame_placeholder = st.empty()
            cap = cv2.VideoCapture(0)

            while run_cam:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_live.process(rgb_frame)
                
                # VARIABEL PENAMPUNG HASIL
                display_label = "-" 
                
                if results.multi_hand_landmarks:
                    all_coords = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        for lm in hand_landmarks.landmark:
                            all_coords.extend([lm.x, lm.y, lm.z])
                    
                    if len(all_coords) == 63:
                        all_coords.extend([0.0] * 63)
                    
                    if len(all_coords) == 126:
                        feature_names = [f'coord_{i}' for i in range(126)]
                        coords_df = pd.DataFrame([all_coords], columns=feature_names)
                        pred = model.predict(coords_df)
                        display_label = pred[0] # Update variabel hasil

                # UPDATE DISPLAY
                prediction_placeholder.markdown(f"""
                    <div style="background-color:#1E1E1E; padding:30px; border-radius:15px; text-align:center; border: 2px solid #ff4b4b;">
                        <p style="color:white; font-size: 18px; margin-bottom:5px;">Terdeteksi:</p>
                        <h1 style="color:#ff4b4b; font-size: 100px; margin:0;">{display_label}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                video_frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()

        else:
            # --- MODE UPLOAD FOTO ---
            uploaded_file = st.file_uploader("Upload Foto", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                
                # Resize otomatis
                h_orig, w_orig = img.shape[:2]
                if w_orig > 1000:
                    img = cv2.resize(img, (1000, int(h_orig * 1000 / w_orig)))

                img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands_upload.process(rgb_img)
                
                final_label_up = "-"
                
                if results.multi_hand_landmarks:
                    coords_up = []
                    for h_lms in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(img, h_lms, mp_hands.HAND_CONNECTIONS)
                        for lm in h_lms.landmark:
                            coords_up.extend([lm.x, lm.y, lm.z])
                    
                    if len(coords_up) == 63: coords_up.extend([0.0] * 63)
                    
                    if len(coords_up) == 126:
                        f_names = [f'coord_{i}' for i in range(126)]
                        coords_df_up = pd.DataFrame([coords_up], columns=f_names)
                        pred_up = model.predict(coords_df_up)
                        final_label_up = pred_up[0] # Update variabel hasil
                else:
                    st.warning("⚠️ Tangan tidak terdeteksi.")

                prediction_placeholder.markdown(f"""
                    <div style="background-color:#1E1E1E; padding:30px; border-radius:15px; text-align:center; border: 2px solid #ff4b4b;">
                        <p style="color:white; font-size: 18px; margin-bottom:5px;">Terdeteksi:</p>
                        <h1 style="color:#ff4b4b; font-size: 100px; margin:0;">{final_label_up}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)