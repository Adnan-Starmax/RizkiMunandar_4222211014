import os
import csv
import json
import time
import base64
import requests
from difflib import SequenceMatcher
import pandas as pd
import re

# ======== Konfigurasi ========
image_dir = r"C:\Users\Lenovo\Documents\VLM_OCR\data\test"
ground_truth_csv = r"C:\Users\Lenovo\Documents\VLM_OCR\data\ground_truth.csv"
output_csv = r"C:\Users\Lenovo\Documents\VLM_OCR\vlm_license_plate_results.csv"
LMSTUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

# ======== Prompt untuk VLM ========
PROMPT_TEMPLATE = "What is the license plate number shown in this image? Respond only with the plate number."

# ======== Fungsi Hitung CER ========
def compute_cer(gt, pred):
    matcher = SequenceMatcher(None, gt, pred)
    s = d = i = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            s += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            d += i2 - i1
        elif tag == 'insert':
            i += j2 - j1
    n = len(gt) if gt else 1
    return (s + d + i) / n

# ======== Bersihkan Output Model ========
def clean_prediction(pred):
    pred = pred.strip().upper()
    pred = re.sub(r"[^A-Z0-9]", "", pred)
    pred = re.sub(r"^THELICENSEPLATENUMBERSHOWNINTHEIMAGEIS", "", pred)

    match = re.search(r"[A-Z]{1,2}[0-9]{3,4}[A-Z]{1,3}", pred)
    return match.group(0) if match else pred

# ======== Kirim Gambar ke LMStudio ========
def send_to_lmstudio(image_path):
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
        img_b64 = base64.b64encode(image_data).decode("utf-8")

    payload = {
        "model": "llava-1.6-mistral-7b",  # Sesuaikan dengan model kamu
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 100
    }

    try:
        response = requests.post(LMSTUDIO_ENDPOINT, headers=HEADERS, data=json.dumps(payload))
        result = response.json()
        answer = result['choices'][0]['message']['content']
        return answer.strip()
    except Exception as e:
        print(f"❌ Error memproses {image_path}: {e}")
        return ""

# ======== Main Program ========
def main():
    df = pd.read_csv(ground_truth_csv)
    results = []

    print("image, ground_truth, prediction, CER_score")

    for _, row in df.iterrows():
        image_name = row["image"]
        gt = row["ground_truth"]
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"❌ File tidak ditemukan: {image_name}")
            continue

        print(f"🔍 Processing {image_name}...")

        pred_raw = send_to_lmstudio(image_path)
        pred = clean_prediction(pred_raw)
        cer = compute_cer(gt, pred)

        print(f"{image_name}, {gt}, {pred}, {cer:.4f}")

        results.append({
            "image": image_name,
            "ground_truth": gt,
            "prediction": pred,
            "CER_score": f"{cer:.4f}"
        })

        time.sleep(0.5)  # Hindari overload LMStudio

    # Simpan ke file CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "ground_truth", "prediction", "CER_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Semua hasil disimpan di: {output_csv}")

# ======== Run ========
if __name__ == "__main__":
    main()
