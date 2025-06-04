# 💓 Human Vital Detection using Facial Skin Color Variation

This project uses **computer vision** and **signal processing** techniques to non-invasively detect **heart rate** and **oxygen saturation (SpO₂)** from facial video footage, by analyzing **skin color variation** caused by blood flow.

## 🧠 Project Overview

Using only a webcam or video input, this system detects a person's face, extracts specific **Regions of Interest (ROIs)** (like cheeks or forehead), and processes the subtle color changes over time to estimate vital signs.

### 👨‍⚕️ Vital Parameters Detected
- 💓 **Heart Rate (BPM)**
- 🫁 **Oxygen Saturation (SpO₂%)**

> Future Scope: Blood Pressure, Stress Level, Heart Rate Variability (HRV), and Body Temperature.

---

## 🔍 How It Works

1. **Face Detection**  
   Uses OpenCV’s **Viola-Jones algorithm** (`haar cascade`) to detect the face in each frame.

2. **Skin Region Extraction**  
   Extracts the **cheek area** or other regions of interest using facial landmarks or fixed ROI logic.

3. **Color Signal Extraction**  
   - Tracks color variation (typically in the **green** channel) over frames.
   - Applies **frame difference** or average pixel intensity calculation.

4. **Signal Processing**  
   - Applies a **Bandpass Filter** to isolate the frequency band of interest (typically 0.7–4 Hz).
   - Uses **Fast Fourier Transform (FFT)** to convert the signal to the frequency domain.

5. **Vital Estimation**
   - Detects peak frequency in the FFT result to estimate **Heart Rate** (HR).
   - Uses Red/Green ratio and waveform analysis for **SpO₂** estimation.

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: OpenCV, NumPy, SciPy, Matplotlib,PyQt5
- **Tools**: Jupyter Notebook / Streamlit (for GUI)

---
## Output Images
![WhatsApp Image 2025-06-04 at 19 03 58_3d1cc06a](https://github.com/user-attachments/assets/9896ae45-ea4c-4549-9023-e89f0845bd1c)
