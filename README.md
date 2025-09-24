# YOLO Desktop Object Detector

A beginner-friendly Windows desktop application for real-time object detection using Ultralytics YOLO and PySide6. Supports local DirectShow webcams (including the "IP Camera Adapter" virtual camera) or direct HTTP/RTSP streams from a phone.

## Features
- Enumerates DirectShow camera devices with quick Connect/Disconnect controls.
- Accepts HTTP/RTSP stream URLs (e.g. `http://<phone-ip>:8080/video`).
- Lazy-loads Ultralytics YOLO models with presets (Nano, Small, Medium, Large) or a custom `.pt` file (auto-downloads if needed).
- Progress dialog menjelaskan proses unduhan model dan memberi tahu jika selesai atau terjadi kesalahan.
- Adjustable confidence threshold and per-class filtering.
- Responsive PySide6 interface backed by worker thread for capture + inference (no UI freezes).
- Annotated preview with FPS & inference time overlay.
- Snapshot and optional MP4/AVI recording of annotated frames (`runs/snapshots`, `runs/records`).
- Status bar and scrolling log for quick troubleshooting.
- Clean shutdown that releases capture devices and threads.

## Requirements
- Windows 10/11
- Python 3.11+
- Internet connection for the first model download (unless model file already present).

## Installation (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> **Tip:** If you plan to use a GPU, install the matching PyTorch build first via the [official selector](https://pytorch.org/get-started/locally/). CPU-only installs work out of the box.

## Phone Camera + IP Camera Adapter Setup
1. Ensure your phone and PC share the same Wi-Fi network.
2. Install the **IP Webcam** app on Android and start the server. Note the stream URL (`http://<phone-ip>:8080`).
3. On Windows, install **IP Camera Adapter (x64)** and configure it with the phone stream URL if you want it to appear as a DirectShow webcam.
4. In this desktop app, either pick the adapter from the camera dropdown or paste the URL `http://<phone-ip>:8080/video` into the Stream URL field.

## Running the Application
Install Ultralytics once (if it is not already in your environment):
```powershell
python -m pip install ultralytics
```

### CPU-only launch
Use the helper script to disable CUDA and force PyTorch onto the CPU:
```powershell
python run_cpu.py
```

### GPU launch
If you have a CUDA-capable GPU, point the launcher at the device index (defaults to `0` if omitted) and run the GPU script:
```powershell
$env:APP_GPU_DEVICE = "0"   # optional
python run_gpu.py
```
The script clears any CPU-forcing flags and sets `ULTRALYTICS_DEVICE` so the detector moves the model onto the requested GPU.

#### Installing CUDA-enabled PyTorch
Ultralytics relies on PyTorch. To run on the GPU you must install a CUDA-capable PyTorch build:
```powershell
# Visit https://pytorch.org/get-started/locally/ and pick the command
# that matches your CUDA toolkit version. Example for CUDA 12.1:
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
After installation, verify CUDA support inside your virtual environment:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
It should print `True`. If it prints `False`, double-check your NVIDIA drivers and selected wheel.

### Manual entry point
You can still launch the original module directly if you prefer to manage environment variables yourself:
```powershell
python app/main.py
```

### Model storage
Place any additional `.pt` weights you download into the `models` folder (ignored by git) so the repository stays clean. Use the **Model kustom** picker in the app to load them from there.

### Typical Workflow
1. **Select a camera** from the dropdown or paste a stream URL, then click **Connect**.
2. Pilih model YOLO dari dropdown (Nano/Small/Medium/Large) atau klik "Model kustom" untuk memilih file `.pt` milik Anda.
3. Load the model to populate the class list, adjust the confidence slider, and choose classes to detect.
4. Click **Start Detection** to begin inference. Use **Stop Detection** to pause while keeping the camera connected.
5. Use **Take Snapshot** to save the current annotated frame or enable **Record annotated video** before starting detection to log MP4/AVI clips.

### Output Folders
- Annotated snapshots: `runs/snapshots/` (timestamped JPEG files)
- Recorded videos: `runs/records/`

## Configuration Persistence
The app stores your last-used selections (camera source, model path, confidence, class filters, recording toggle) in `app/config.json` so they load automatically next time.

## Troubleshooting
- **Camera won't open:** Ensure the phone stream is running, your firewall allows local connections, and the IP Camera Adapter is configured. For USB cameras, close other apps that might be using them.
- **Model download fails:** Verify internet access or manually place the `.pt` file and set its path via the picker.
- **Low FPS:** Try lighter models (e.g., `yolo11n.pt`), lower the input resolution inside the Ultralytics model settings, or pause other CPU-intensive tasks.
- **Recording fails:** Install codecs such as [FFmpeg](https://ffmpeg.org/) if MP4 creation is unavailable on your system. The app automatically falls back to AVI.
- **Adapter not listed:** Click "Refresh Cameras" in the menu or restart the adapter service.
- **URL errors (404/401):** Double-check the phone app's streaming endpoint and authentication settings.

## Tests
Simple `unittest` checks are provided to validate camera enumeration, model loading, and the threaded frame loop (with mocked capture). Run all tests with:
```powershell
python -m unittest discover -s tests
```

## Safe Shutdown
Closing the application stops detection, flushes recordings, releases the camera, and persists your preferences.
