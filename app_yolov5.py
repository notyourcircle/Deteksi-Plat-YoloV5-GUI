import torch
import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
import easyocr
import re

# Inisialisasi model YOLO dan EasyOCR
model_path = "best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)

reader = easyocr.Reader(['en'])

def compute_iou(boxA, boxB):
    """
    Menghitung Intersection over Union (IoU) dari dua bounding box.
    Masing-masing box dalam format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(areaA + areaB - interArea + 1e-5)
    return iou

def non_max_suppression(detections, iou_threshold=0.4):
    """
    Filter bounding box yang tumpang tindih.
    Jika dua atau lebih bbox tumpang tindih (IoU >= threshold),
    hanya ambil bbox dengan area terbesar.
    
    detections: list/array dengan format [x1, y1, x2, y2, conf, cls]
    """
    if len(detections) == 0:
        return []
    
    areas = [(det[2] - det[0]) * (det[3] - det[1]) for det in detections]
    idxs = sorted(range(len(detections)), key=lambda i: areas[i], reverse=True)
    
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(detections[i])
        idxs = [j for j in idxs if compute_iou(detections[i][:4], detections[j][:4]) < iou_threshold]
    return keep

def clear_panels():
    panel_original.config(image='')
    panel_original.image = None
    panel_bbox.config(image='')
    panel_bbox.image = None
    panel_crop.config(image='')
    panel_crop.image = None
    panel_gray.config(image='')
    panel_gray.image = None
    label_ocr.config(text="Hasil OCR: ")

def upload_image():
    clear_panels()
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image(image_rgb, panel_original, resize=True)
    
    # Deteksi objek dengan YOLO
    results = model(image_rgb)
    _,preds = results.xyxy[0][:, -1], results.xyxy[0][:, :-1]
    
    # Terapkan NMS untuk menghilangkan bbox yang tumpang tindih
    filtered_preds = non_max_suppression(preds, iou_threshold=0.4)
    print(preds)
    print(filtered_preds)

    # Salin gambar untuk menampilkan bounding box
    image_bbox = image_rgb.copy()
    
    for detection in filtered_preds:
        x1, y1, x2, y2 = map(int, detection[:4])
       
        # Gambar bounding box dengan koordinat yang sudah disesuaikan
        cv2.rectangle(image_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Crop plat nomor menggunakan bounding box yang disesuaikan
        plate_image = image[y1:y2, x1:x2]
        display_image(plate_image, panel_crop)
        
        # Konversi ke grayscale
        plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        display_image(plate_gray, panel_gray, is_gray=True)
        
        # OCR dengan EasyOCR dengan mode detail
        ocr_results = reader.readtext(plate_gray, detail=1)
        if ocr_results:
            # Hitung tinggi tiap kandidat berdasarkan bounding box OCR
            heights = []
            for res in ocr_results:
                bbox = res[0]
                y_coords = [pt[1] for pt in bbox]
                height = max(y_coords) - min(y_coords)
                heights.append(height)
            max_height = max(heights)
            # Filter kandidat dengan tinggi >= 50% dari tinggi maksimum
            filtered_results = [res for res, h in zip(ocr_results, heights) if h >= 0.5 * max_height]
            
            # Filter hasil OCR: abaikan teks yang sesuai pola masa berlaku (misalnya "12.34")
            pattern_expiration = re.compile(r'^\d{2}\.\d{2}$')
            filtered_texts = [res[1] for res in filtered_results if not pattern_expiration.match(res[1])]
            final_text = " ".join(filtered_texts) if filtered_texts else "Tidak terdeteksi"
        else:
            final_text = "Tidak terdeteksi"
       
        # Ubah hasil OCR: hanya angka dan huruf besar, hapus semua spasi terlebih dahulu
        final_text = final_text.upper()
        final_text = re.sub(r'\s+', '', final_text)
        final_text = re.sub(r'[^A-Z0-9]', '', final_text)

        # Sisipkan spasi berdasarkan pola: huruf, angka, huruf
        # Jika huruf diikuti angka, sisipkan spasi
        final_text = re.sub(r'(?<=[A-Z])(?=\d)', ' ', final_text)
        # Jika angka diikuti huruf, sisipkan spasi
        final_text = re.sub(r'(?<=\d)(?=[A-Z])', ' ', final_text)
        cv2.putText(image_bbox, final_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        label_ocr.config(text=f"Hasil OCR: {final_text}")

    display_image(image_bbox, panel_bbox, resize=True)

def display_image(image, panel, is_gray=False, max_width=500, max_height=400, resize=False):
    """
    Konversi gambar ke format PIL, resize agar sesuai dengan max_width dan max_height,
    lalu tampilkan pada panel Tkinter.
    """
    if is_gray:
        img_pil = Image.fromarray(image)
    else:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Resize gambar dengan menjaga aspect ratio
    if resize==True:
        orig_width, orig_height = img_pil.size
        ratio = min(max_width / orig_width, max_height / orig_height)
        new_size = (int(orig_width * ratio), int(orig_height * ratio))
        img_pil = img_pil.resize(new_size, Image.LANCZOS)
        
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk


# -------------------- GUI Layout --------------------
root = tk.Tk()
root.title("Deteksi Plat Nomor Kendaraan")
root.geometry("850x700")

# Atur grid agar kolom 0 dan 1 memiliki bobot yang sama (untuk center alignment)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Row 0: Tombol Upload di bagian atas
btn_upload = Button(root, text="Upload Gambar", command=upload_image)
btn_upload.grid(row=0, column=0, columnspan=2, pady=10)

# Row 1: Judul untuk Gambar Asli dan Deteksi Bounding Box
lbl_original = Label(root, text="Gambar Asli")
lbl_original.grid(row=1, column=0, padx=5, pady=5)
lbl_bbox = Label(root, text="Deteksi Bounding Box")
lbl_bbox.grid(row=1, column=1, padx=5, pady=5)

# Row 2: Panel untuk Gambar Asli dan Deteksi Bounding Box
panel_original = Label(root, width=400, height=300)
panel_original.grid(row=2, column=0, padx=5, pady=5)
panel_bbox = Label(root, width=400, height=300)
panel_bbox.grid(row=2, column=1, padx=5, pady=5)

# Row 3: Judul untuk Cropped Plat Nomor dan Grayscale Plat Nomor
lbl_crop = Label(root, text="Cropped Plat Nomor")
lbl_crop.grid(row=3, column=0, padx=5, pady=5)
lbl_gray = Label(root, text="Grayscale Plat Nomor")
lbl_gray.grid(row=3, column=1, padx=5, pady=5)

# Row 4: Panel untuk Cropped dan Grayscale Plat Nomor
panel_crop = Label(root)
panel_crop.grid(row=4, column=0, padx=5, pady=5)
panel_gray = Label(root)
panel_gray.grid(row=4, column=1, padx=5, pady=5)

# Row 5: Hasil OCR sebagai sub header, spanning dua kolom
label_ocr = Label(root, text="Hasil OCR:", font=("Helvetica", 16, "bold"))
label_ocr.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
