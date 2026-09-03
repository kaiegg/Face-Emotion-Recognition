import base64
import os
import time
import urllib.request
import cv2
from IPython.display import Javascript, display
from google.colab.output import eval_js
import numpy as np
from tensorflow.keras.models import load_model

# --- PREPARATION: Download Haar Cascade XML ---
xml_filename = 'haarcascade_frontalface_default.xml'
if not os.path.exists(xml_filename):
  print(f'Downloading {xml_filename}...')
  url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{xml_filename}'
  urllib.request.urlretrieve(url, xml_filename)

# --- 1. LOAD MODEL & CATEGORIES ---
data_dir = 'data/BTFER7'
model_path = 'models/model_batch7.h5'

if not os.path.exists(model_path):
  raise FileNotFoundError(f'Model not found at {model_path}')

print('Loading 8-class model...')
model = load_model(model_path)
face_detector = cv2.CascadeClassifier(xml_filename)

# Automatically read and sort category names from dataset folder
categories = sorted(
    [
        d
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    ]
)
target_size = (64, 64)


# --- 2. JAVASCRIPT WEBCAM & OVERLAY BRIDGE ---
def init_webcam_overlay():
  js = Javascript('''
        async function initWebcamOverlay() {
            if (document.getElementById('webcam_container')) return;

            const div = document.createElement('div');
            div.id = 'webcam_container';
            div.style.position = 'relative';
            div.style.width = '640px';
            div.style.height = '480px';

            const video = document.createElement('video');
            video.id = 'webcam_video';
            video.width = 640;
            video.height = 480;
            video.autoplay = true;

            const canvas = document.createElement('canvas');
            canvas.id = 'overlay_canvas';
            canvas.width = 640;
            canvas.height = 480;
            canvas.style.position = 'absolute';
            canvas.style.top = '0';
            canvas.style.left = '0';

            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            
            div.appendChild(video);
            div.appendChild(canvas);
            document.body.appendChild(div);

            await new Promise((resolve) => video.onloadedmetadata = resolve);
            window.colabWebcamStream = stream;
        }
        initWebcamOverlay();
    ''')
  display(js)


def draw_overlay(detections):
  """Sends detection data (boxes + labels) to JavaScript to draw on screen."""
  js = Javascript(f'''
        function updateCanvas() {{
            const canvas = document.getElementById('overlay_canvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const detections = {detections};
            detections.forEach(det => {{
                // Draw bounding box
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                ctx.strokeRect(det.x, det.y, det.w, det.h);

                // Draw label background and text
                ctx.fillStyle = '#00FF00';
                ctx.font = 'bold 18px Arial';
                const text = det.label + " (" + det.conf + "%)";
                const textWidth = ctx.measureText(text).width;
                ctx.fillRect(det.x, det.y - 25 > 0 ? det.y - 25 : det.y, textWidth + 10, 25);

                ctx.fillStyle = '#000000';
                ctx.fillText(text, det.x + 5, det.y - 25 > 0 ? det.y - 7 : det.y + 18);
            }});
        }}
        updateCanvas();
    ''')
  display(js)


def stop_webcam():
  js = Javascript('''
        async function stopWebcam() {
            const div = document.getElementById('webcam_container');
            if (window.colabWebcamStream) {
                window.colabWebcamStream.getTracks().forEach(track => track.stop());
                window.colabWebcamStream = null;
            }
            if (div) div.remove();
        }
        stopWebcam();
    ''')
  display(js)


def capture_frame():
  js = Javascript('''
        async function captureFrame() {
            const video = document.getElementById('webcam_video');
            if (!video || video.videoWidth === 0) return null;

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            return canvas.toDataURL('image/jpeg', 0.8);
        }
        captureFrame();
    ''')
  display(js)
  try:
    data = eval_js('captureFrame()')
    if not data or ',' not in data:
      return None
    binary = base64.b64decode(data.split(',')[1])
    return cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
  except Exception:
    return None


# --- 3. MAIN EXECUTION LOOP ---
print('\n--- Starting Real-Time Emotion Detection ---')
init_webcam_overlay()
time.sleep(3)  # Allow camera initialization

duration_seconds = 30
start_time = time.time()

try:
  while True:
    if time.time() - start_time > duration_seconds:
      print(f'\nTime limit of {duration_seconds}s reached.')
      break

    frame = capture_frame()
    if frame is None:
      continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5
    )

    detections = []
    for x, y, w, h in faces:
      # Extract face region
      roi_bgr = frame[y : y + h, x : x + w]
      roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

      # Preprocess image
      resized_face = cv2.resize(roi_rgb, target_size) / 255.0
      face_array = np.expand_dims(resized_face, axis=0)

      # Inference
      prediction = model.predict(face_array, verbose=0)
      predicted_class = np.argmax(prediction)
      confidence = float(prediction[0][predicted_class] * 100)

      # Build JSON payload for JS overlay renderer
      detections.append({
          'x': int(x),
          'y': int(y),
          'w': int(w),
          'h': int(h),
          'label': str(categories[predicted_class]),
          'conf': f'{confidence:.1f}',
      })

    # Render bounding boxes and text directly onto webcam canvas layer
    draw_overlay(detections)

except KeyboardInterrupt:
  print('\nInterrupted by user.')
finally:
  stop_webcam()
  print('Camera closed successfully.')
