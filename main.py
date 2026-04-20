import os
import sys
import json
import asyncio
import webbrowser
import shutil
import uvicorn
import pyaudio
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
from io import BytesIO


# 1. Функція для визначення шляхів усередині EXE (PyInstaller)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


app = FastAPI()

# 2. Налаштування CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Логіка роботи з папкою зображень (Пресети + Кастомні)
BUNDLE_IMG_DIR = resource_path("images")
LOCAL_IMG_DIR = "images"

if not os.path.exists(LOCAL_IMG_DIR):
    os.makedirs(LOCAL_IMG_DIR)

# Копіюємо вбудовані пресети в локальну папку, якщо їх там немає
if os.path.exists(BUNDLE_IMG_DIR):
    for item in os.listdir(BUNDLE_IMG_DIR):
        s = os.path.join(BUNDLE_IMG_DIR, item)
        d = os.path.join(LOCAL_IMG_DIR, item)
        if not os.path.exists(d) and os.path.isfile(s):
            shutil.copy2(s, d)

app.mount("/images", StaticFiles(directory=LOCAL_IMG_DIR), name="images")

# 4. Аудіо ініціалізація
CHUNK = 2048
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

flux_history = []
prev_fft_data = np.zeros(CHUNK // 2 + 1)


def rotate_and_save_image(band_name, file_content):
    base_path = os.path.join(LOCAL_IMG_DIR, f"{band_name}.jpg")
    for i in range(4, 0, -1):
        old_file = os.path.join(LOCAL_IMG_DIR, f"{band_name}_old{i}.jpg")
        next_file = os.path.join(LOCAL_IMG_DIR, f"{band_name}_old{i + 1}.jpg")
        if os.path.exists(old_file):
            if os.path.exists(next_file): os.remove(next_file)
            os.rename(old_file, next_file)
    if os.path.exists(base_path):
        os.rename(base_path, os.path.join(LOCAL_IMG_DIR, f"{band_name}_old1.jpg"))

    img = Image.open(BytesIO(file_content))
    img = img.convert("RGB")
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (1000, 1000), (0, 0, 0))
    offset = ((1000 - img.width) // 2, (1000 - img.height) // 2)
    new_img.paste(img, offset)
    new_img.save(base_path, "JPEG", quality=90)


@app.get("/")
async def get_index():
    with open(resource_path("index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/upload")
async def upload_image(file: UploadFile = File(...), band: str = Form(...)):
    content = await file.read()
    rotate_and_save_image(band, content)
    return {"status": "success"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global flux_history, prev_fft_data

    session_modifiers = {
        "low_1": 8000.0, "low_2": 8000.0, "low_3": 8000.0,
        "high_1": 10000.0, "high_2": 10000.0, "high_3": 10000.0,
        "beat": 400000.0
    }

    async def receive_updates():
        try:
            while True:
                data = json.loads(await websocket.receive_text())
                if "update" in data: session_modifiers.update(data["update"])
        except:
            pass

    asyncio.create_task(receive_updates())

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            fft_data = np.abs(np.fft.rfft(audio_data))

            bands = {
                "low_1": np.mean(fft_data[1:15]) / session_modifiers["low_1"],
                "low_2": np.mean(fft_data[15:40]) / session_modifiers["low_2"],
                "low_3": np.mean(fft_data[40:100]) / session_modifiers["low_3"],
                "high_1": np.mean(fft_data[100:250]) / session_modifiers["high_1"],
                "high_2": np.mean(fft_data[250:500]) / session_modifiers["high_2"],
                "high_3": np.mean(fft_data[500:1000]) / session_modifiers["high_3"]
            }

            flux = np.sum(np.maximum(0, fft_data - prev_fft_data))
            prev_fft_data = fft_data
            flux_history.append(flux)
            if len(flux_history) > 10: flux_history.pop(0)

            is_beat = bool(flux > np.mean(flux_history) * 1.7 and flux > session_modifiers["beat"])
            payload = {"beat": is_beat, **{k: float(min(1.0, max(0.0, v)) ** 0.7) for k, v in bands.items()}}

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.01)
    except:
        pass
    finally:
        # 5. Самозавершення процесу при закритті вкладки (розриві WS)
        stream.stop_stream()
        stream.close()
        p.terminate()
        os._exit(0)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000")
    # log_config=None критично для уникнення помилки AttributeError в EXE
    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)