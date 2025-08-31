# src/camera.py
"""
camera.py
- start_camera() returns an OpenCV VideoCapture object.
- capture_image(cap) captures one frame and returns image path + PIL image or numpy array.
- describe_image(image) tries to use BLIP (Hugging Face Salesforce checkpoint) to generate caption; 
  if not available, falls back to simple detection/simulation.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Transformers import
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch
    _has_blip = True
except Exception:
    _has_blip = False

# Load BLIP (Salesforce Hugging Face model) once
processor, model, device = None, None, None
if _has_blip:
    try:
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    except Exception as e:
        print("[camera] Could not load Hugging Face BLIP model:", e)
        _has_blip = False


def start_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check camera index/permissions.")
    return cap


def capture_image(cap, save_dir="captures/captures"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from camera.")
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    filename = save_dir / f"capture.jpg"
    Image.fromarray(img_rgb).save(filename)
    return filename, img_rgb


def describe_image(image, processor=processor, model=model, device=device):
    """
    Input:
      - image: numpy array (RGB), PIL image, or path (str/Path)
    Output:
      - text description (string)
    """
    try:
        if not _has_blip or processor is None or model is None:
            raise RuntimeError("BLIP model not available")

        # Normalize input → PIL RGB
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        # Run through BLIP
        inputs = processor(images=img, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_length=50)
        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return caption

    except Exception as e:
        print("[camera.describe_image] BLIP failed:", e)

        # Fallback: simple OpenCV DNN (MobileNet-SSD)
        try:
            proto = Path("models/mobilenet_ssd_deploy.prototxt")
            weights = Path("models/mobilenet_ssd.caffemodel")
            if proto.exists() and weights.exists():
                net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
                if isinstance(image, (str, Path)):
                    frame = cv2.imread(str(image))
                elif isinstance(image, np.ndarray):
                    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                             0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                found = {}
                for i in range(detections.shape[2]):
                    conf = float(detections[0, 0, i, 2])
                    if conf > 0.5:
                        idx = int(detections[0, 0, i, 1])
                        label = str(idx)
                        found[label] = found.get(label, 0) + 1

                if found:
                    items = ", ".join(f"{k} (x{v})" for k, v in found.items())
                    return f"Detected objects (IDs): {items} — (fallback detection)"
        except Exception as e2:
            print("[camera.describe_image] DNN fallback failed:", e2)

        # Final fallback
        return "I see a scene. (No caption model available — BLIP failed.)"
