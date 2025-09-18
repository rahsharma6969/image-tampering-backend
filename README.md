# 🖼️ Image Forgery Detection Web App – Powered by MantraNet

A **deep learning-based web application** to detect digital image tampering using the **MantraNet neural network**.  
The system visually highlights forged areas and provides a confidence score to help verify the authenticity of images and documents.

---

## 🚀 Features

- **Image Upload:** Supports JPEG, PNG, TIFF, BMP, and multi-page PDF files.
- **Forgery Detection:** Runs MantraNet to generate:
  - Tampering heatmap
  - Overlay image with highlighted forged regions
  - Forgery score (percentage of suspected tampering)
- **Metadata Extraction:** Displays EXIF metadata, resolution, and image details.
- **Verdict System:**
  - Forgery Score `< 20%` → **CLEAN**
  - Forgery Score `≥ 20%` → **SUSPICIOUS**

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, PyTorch
- **Model:** Pretrained [MantraNet](https://github.com/ISICV/ManTraNet) (`MantraNetv4.pt`)
- **Frontend:** React (consuming backend APIs)
- **Visualization:** Pillow + NumPy for overlays and heatmaps

---

## 📌 Use Cases

- Digital forensics & legal document verification  
- Detecting tampered media in journalism and social platforms  
- Educational demonstrations of image forgery detection  

---

## ⚙️ Installation & Setup

### 1. Clone the repositories
```bash
# Backend
git clone https://github.com/rahsharma6969/image-tampering-backend.git
cd image-tampering-backend

# Frontend
git clone https://github.com/rahsharma6969/image-tampered-Frontend.git
cd image-tampered-Frontend
