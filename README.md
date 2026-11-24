# Face Recognition Access System

An access-control system built on an ESP32-CAM and a laptop running OpenCV's
YuNet + SFace face pipeline. The board streams frames over WiFi; the laptop
decides who is at the door and sends the verdict back, which switches an
onboard LED — green for an authorised employee, off otherwise.

Built at Tech4Life Enterprises as an internship project (2026).

## How this code was written

My supervisor was fine with AI tools being used, on one
condition: that I understood the system — how the board and the laptop talk to
each other, and why each part is there.

So the code came together with AI help; the understanding is mine. The
interesting parts here are the architecture, the threshold and enrolment
choices, the refusal path, and the testing that found the issues below — and I
can explain any of it.

## How it works

1. `src/check_camera.py` - a smoke test: loads both models, streams frames from
   the board and draws detection boxes. Run this first when something breaks.
2. `src/database_setup.py` - creates the SQLite employee table.
3. `src/enroll-face.py` - captures several frames of a person, extracts a
   128-dimensional SFace embedding, averages it, and stores it as
   `database/embeddings/<employee_id>.npy` alongside their name, job title and
   authorisation flag.
4. `src/recognize_face.py` - polls the camera, detects a face with YuNet,
   extracts an embedding, compares it by **cosine similarity** against every
   enrolled vector, and grants or refuses at a threshold of `0.45`.

Nothing about a face is stored as an image. Only the embedding is kept, which
means the system cannot be used to reconstruct a photo of an employee.

## Setup

```bash
git clone <this repo> && cd face-recognition-access
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_models.sh          # YuNet + SFace ONNX models (~40 MB)
```

Set the board's IP in `src/recognize_face.py` (`ESP32_IP`), then, **from the
repo root**:

```bash
python src/database_setup.py   # create database/access.db
python src/enroll-face.py      # register one person at a time
python src/recognize_face.py   # run the access loop, Q to quit
```

That last step is preceded by `python src/check_camera.py`, which tells you in
five seconds whether the problem is the network, the board or the models.

## Design decisions

**Cosine similarity over Euclidean distance.** SFace embeddings are trained
with a cosine-margin loss, so the angle between vectors is the meaningful
quantity. Cosine also ignores embedding magnitude, which varies with lighting
and pose; normalising away that variation makes the threshold easier to set.

**Averaging multiple enrollment samples.** One frame per person is fragile -
eyes closed, a shadow, a bad angle. Averaging five embeddings produces a
centroid that is markedly more stable, at no extra cost at recognition time.

**Threshold at 0.45.** Tuned empirically: high enough to reject a different
person, low enough to tolerate glasses and indoor lighting changes.

**Unknown faces are refused before the database lookup.** The threshold check
happens on the score, so an impostor never reaches the employee table.

## Things I would change

- **The threshold is a single global number.** Per-person thresholds derived
  from enrollment spread would handle people whose embedding clusters are
  naturally tight versus spread.
- **Liveness is absent.** A printed photo held to the camera would pass. Blink
  detection or a depth/frame-difference check is the next step, and without it
  this is a demo of recognition, not of security.
- **The LED is the only actuator.** A relay on a door strike would make it a
  real access system.
- **The camera IP is hardcoded.** mDNS (`esp32cam.local`) or a config file
  would remove a common failure mode.

## Firmware

`firmware/ESP32CAM_Server.ino` is the sketch flashed to the board. It does four
things and nothing else:

- `GET /` - a small status page: live preview, plus LED ON / LED OFF buttons
- `GET /capture` - grabs one frame and returns it as JPEG, with no-cache headers
- `GET /led/on` - drives GPIO 4 high
- `GET /led/off` - drives GPIO 4 low

The preview page refreshes `/capture` every 250 ms, which is how the laptop's
frame loop was fed during testing.

Two details that came out of the hardware rather than from a tutorial. The board
has **no PSRAM**, so the config is pinned to `FRAMESIZE_HQVGA`, JPEG quality 15,
a single frame buffer in DRAM and `CAMERA_GRAB_WHEN_EMPTY`; the usual VGA +
PSRAM settings simply fail `esp_camera_init` on this module. And `ssid` /
`password` at the top are placeholders - set them before flashing.

## Known rough edges

- The three scripts were developed separately, so the board IP must be set in
  each if you change it.
- Frame size is fixed at 240x176 for latency; detection on larger frames is
  noticeably slower on a laptop CPU.

## Stack

Python, OpenCV (YuNet detector, SFace recogniser), SQLite, NumPy, requests,
ESP32-CAM (Arduino C++).

## Models

Neither ONNX model is mine. They are from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) and are fetched by
`download_models.sh` rather than committed here.
