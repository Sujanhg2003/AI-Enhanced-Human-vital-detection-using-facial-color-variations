import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import time

# Initialize webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    if len(data) < (3 * order * 2):  # Ensure enough data points
        return data  # Return unfiltered if insufficient data
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Extract ROI (cheek region)
def extract_roi(frame, face):
    x, y, w, h = face
    cheek_roi = frame[y + int(h * 0.6):y + int(h * 0.8), x + int(w * 0.2):x + int(w * 0.8)]
    return cv2.cvtColor(cheek_roi, cv2.COLOR_BGR2RGB)

# Extract green and red channel signals
def extract_channel_signals(roi_images):
    green_signals = [np.mean(frame[:, :, 1]) for frame in roi_images]
    red_signals = [np.mean(frame[:, :, 2]) for frame in roi_images]
    return np.array(green_signals), np.array(red_signals)

# Calculate Heart Rate (BPM)
def calculate_heart_rate(green_signal, fs):
    if len(green_signal) < 60:  # Ensure enough samples
        return None
    
    # FFT to find dominant frequency
    N = len(green_signal)
    freqs = fftfreq(N, 1/fs)
    fft_values = np.abs(fft(green_signal))
    
    # Find peak frequency in physiological range (0.8 - 3 Hz)
    min_idx = np.where(freqs >= 0.8)[0][0]
    max_idx = np.where(freqs <= 3)[0][-1]
    dominant_freq = freqs[min_idx:max_idx][np.argmax(fft_values[min_idx:max_idx])]

    bpm = int(dominant_freq * 60)  # Convert Hz to BPM
    return bpm if 40 <= bpm <= 180 else None  # Valid BPM range

# Calculate SpO₂
def calculate_spo2(green_signals, red_signals):
    ac_green = np.std(green_signals)
    ac_red = np.std(red_signals)
    if ac_green == 0 or ac_red == 0:
        return None
    spo2 = 95 + 5 * (ac_red / ac_green)
    return max(90, min(100, spo2))

# PyQtGraph setup for real-time plotting
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Live Heart Rate & SpO₂ Monitoring")
plot = win.addPlot(title="Green Signal Intensity (Zoomed ECG-style)")
curve = plot.plot(pen='g')

# **New Scale Adjustment**: Set a tighter y-range for zooming in
plot.setRange(yRange=(100, 180))  # Previously (0, 255), now zoomed-in

win.show()

# Parameters
fs = 30  # Frames per second
roi_images = []
green_signal_buffer = []

try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi = extract_roi(frame, face)
            roi_images.append(roi)

            if len(roi_images) >= 5 * fs:  # Process every 5 seconds
                green_signals, red_signals = extract_channel_signals(roi_images)

                if len(green_signals) >= 33:  # Prevent 'padlen' error
                    filtered_green = bandpass_filter(green_signals, 0.8, 3, fs)

                    heart_rate = calculate_heart_rate(filtered_green, fs)
                    spo2 = calculate_spo2(green_signals, red_signals)

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] HR: {heart_rate if heart_rate else 'N/A'} BPM, SpO₂: {spo2 if spo2 else 'N/A'}%")

                    green_signal_buffer.extend(filtered_green[-50:])  # Keep adding new data for smooth plotting

                    # Trim buffer to maintain smooth scrolling effect
                    if len(green_signal_buffer) > 300:
                        green_signal_buffer = green_signal_buffer[-300:]

                    roi_images = []

        # **Dynamically adjust the Y-axis range for zoom effect**
        if len(green_signal_buffer) > 0:
            min_val = min(green_signal_buffer)
            max_val = max(green_signal_buffer)
            plot.setYRange(min_val - 5, max_val + 5)  # Adjust scale dynamically

        # Update PyQtGraph live plot
        curve.setData(green_signal_buffer)

        # Display webcam feed
        cv2.imshow('Webcam Feed', frame)
        QtWidgets.QApplication.processEvents()  # PyQtGraph UI update

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Monitoring stopped by user.")

finally:
    webcam.release()
    cv2.destroyAllWindows()
