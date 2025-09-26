# Glass Bottle QC Suite

Aplikasi desktop PySide6 untuk quality control botol kaca end-to-end. Sistem ini menggabungkan inferensi Ultralytics YOLOv11, workflow review, analitik, serta manajemen siklus model dalam antarmuka yang responsif untuk operator pabrik.

## Ringkasan Fitur
- Deteksi real-time dengan metadata batch, ROI, aturan QC, pencatatan otomatis ke SQLite, dan kini mendukung jendela popup kamera.
- Orkestrasi pelatihan dengan start/stop, resume, shortcut hasil, serta generator notebook Colab.
- Review & relabel untuk mengkurasi deteksi, mengedit bounding box, dan menyiapkan dataset YOLO hasil koreksi.
- Analitik & laporan dengan KPI filterable, chart, ekspor CSV, serta PDF siap audit.
- Registri model dengan hashing, validasi, dan aktivasi satu klik.
- Panel pengaturan untuk kamera, path, ROI preset, JSON aturan QC, operator, dan backup basis data.

## Persyaratan
- Windows 10/11
- Python 3.11 ke atas
- Disarankan GPU dengan CUDA (pasang PyTorch CUDA sebelum Ultralytics jika ingin akselerasi GPU)

## Instalasi
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Menjalankan Aplikasi
Entry point GUI berada di `app/main.py`:
```powershell
python app/main.py
```

Helper script untuk konfigurasi CPU/GPU eksplisit:
```powershell
# CPU
python run_cpu.py

# GPU (atur index device lewat environment variable)
$env:APP_GPU_DEVICE = "0"
python run_gpu.py
```

## Alur Deteksi
1. Pilih kamera lokal/IP stream dan load model dari registri.
2. Isi metadata batch (lot, shift, line, operator) dan pilih preset ROI bila ada.
3. Mulai deteksi. Aturan QC (`reject_if`, `min_conf`) menentukan status pass/fail otomatis.
4. Pantau counter, tabel deteksi, log real-time, dan gunakan popup kamera untuk tampilan penuh. Hentikan deteksi untuk menutup batch.

## Review & Relabel
- Telusuri frame gagal atau sampel, edit bounding box/kelas, dan simpan ke truth table ataupun direktori dataset reviewed (format YOLO).
- Tombol ekspor menyiapkan dataset siap latih berikut YAML.

## Analitik & Laporan
- Filter berdasarkan tanggal, lot, line, shift, operator, atau model.
- KPI menampilkan jumlah inspeksi, pass/fail ratio, frekuensi defect, hingga tren.
- Chart dirender ke PNG dan dapat diekspor ke CSV/PDF (ReportLab).

## Siklus Model
- Registrasikan bobot baru melalui tab Training/Registry (hash memastikan tidak duplikat).
- Tombol `Set Active` memperbarui konfigurasi deteksi dan me-load model secara otomatis.
- Validasi ringan dapat dijalankan terhadap frame tersimpan.

## Pelatihan
- Jalankan Ultralytics training dari GUI, pantau log, dan batalkan tanpa membekukan UI.
- `Create Colab Notebook` men-generate notebook dengan instruksi mount dataset dan sel pelatihan.

## Pengaturan & Pemeliharaan
- Simpan preset kamera/device, ROI rect/poly per line, dan atur JSON aturan QC.
- Kelola path default, operator, serta backup/restore SQLite.

## Penyimpanan Data
- Basis data SQLite `qc.db` menyimpan model, batch, inspeksi, deteksi, preset ROI, aturan QC, hasil review, dan operator.
- Snapshot & rekaman berada di `runs/snapshots/` dan `runs/records/`.
- Dataset hasil review ada di `dataset_reviewed/images` & `dataset_reviewed/labels`.
- Ekspor analitik disimpan di folder `exports/`.

## Pengujian
Jalankan smoke test dengan:
```powershell
python -m pytest
```

## Troubleshooting
- **Dependensi belum terpasang**: jalankan ulang `pip install -r requirements.txt` di dalam virtual environment.
- **Model gagal diload**: pastikan path `.pt` benar dan hash sesuai entry registri.
- **Chart kosong**: periksa filter tanggal/lot atau longgarkan rentangnya.
- **Ekspor PDF gagal**: pastikan ReportLab terinstal dan folder `exports/` writable.

Selamat menginspeksi dan pertahankan kualitas botol tanpa cacat!
