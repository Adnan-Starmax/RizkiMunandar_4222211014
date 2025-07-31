# Indonesian License Plate Recognition with Visual Language Model (VLM)

Proyek ini bertujuan untuk melakukan **Optical Character Recognition (OCR)** pada gambar plat nomor kendaraan menggunakan **Visual Language Model (VLM)** yang dijalankan melalui **LMStudio**. Model multimodal seperti **llava-1.6-mistral-7b** digunakan untuk mengenali karakter pada plat nomor berdasarkan input gambar.

Dataset yang digunakan:  
[Indonesian License Plate Dataset (Kaggle)](https://www.kaggle.com/datasets/juanthomaswijaya/indonesianlicense-plate-dataset)  
Folder yang digunakan untuk evaluasi: `test`

---

## Cara Kerja

1. Gambar dari folder `test` dibaca satu per satu.
2. Gambar dikirim ke LMStudio yang menjalankan model multimodal (misalnya `llava-1.6-mistral-7b`) menggunakan HTTP API.
3. Prompt yang digunakan:
What is the license plate number shown in this image? Respond only with the plate number.

markdown
Copy
Edit
4. Hasil prediksi dibersihkan dari noise menggunakan regular expression.
5. Dilakukan evaluasi performa OCR menggunakan **Character Error Rate (CER)** dengan rumus:
CER = (S + D + I) / N

yaml
Copy
Edit
- S: jumlah karakter salah substitusi
- D: jumlah karakter yang dihapus
- I: jumlah karakter yang disisipkan
- N: jumlah karakter pada ground truth

---

## Struktur File

VLM_OCR/
│
├── data/
│ ├── test/ ← Folder gambar plat nomor uji
│ └── ground_truth.csv ← File ground truth format: image,ground_truth
│
├── vlm_ocr_license_plate.py ← Script utama
└── vlm_license_plate_results.csv ← Hasil prediksi & evaluasi

yaml
Copy
Edit

---

## Cara Menjalankan

1. Pastikan LMStudio aktif di `http://localhost:1234/` dan telah menjalankan model multimodal seperti `llava-1.6-mistral-7b`.
2. Install dependencies (jika belum):
   ```bash
   pip install pandas requests
Jalankan program:

bash
Copy
Edit
python vlm_ocr_license_plate.py
Hasil akan tersimpan di:

Copy
Edit
vlm_license_plate_results.csv
Format Output
File vlm_license_plate_results.csv akan berisi 4 kolom:

image	ground_truth	prediction	CER_score
001.jpg	B1234XYZ	B1234XYZ	0.0000
...	...	...	...

Evaluasi
Evaluasi dilakukan berdasarkan Character Error Rate (CER). Semakin kecil nilai CER, semakin akurat prediksi dari model terhadap label ground truth.

