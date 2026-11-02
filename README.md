# IDVerify — offline ID-card identity verification

An **offline, system-level desktop application** in Python: it reads a government
ID card with OCR, pulls the portrait printed on it, and verifies the person in
front of the machine against that portrait — plus the record stored locally for
that ID number. Think facial-recognition-system (FRS) workflow, but the ID card
is the source of truth and **nothing ever leaves the machine**.

```
ID card image ──OCR──▶ name / ID number / DOB / expiry + printed portrait
live capture  ──YuNet─▶ face ──SFace─▶ 128-D descriptor
                                    ├──▶ cosine vs printed portrait   (does this person hold the card?)
                                    └──▶ cosine vs enrolled record    (is this the card we issued?)
```

## Why it is genuinely offline

| Component | Model | Where it comes from |
|---|---|---|
| OCR | PP-OCRv6 det/cls/rec | **bundled inside the `rapidocr` wheel** (`site-packages/rapidocr/models/`) |
| Face detection | YuNet `face_detection_yunet_2023mar.onnx` (232 KB) | vendored in `./models` |
| Face recognition | SFace `face_recognition_sface_2021dec.onnx` (38.7 MB) | vendored in `./models` |
| Storage | SQLite + image files | `~/.idverify/` |

No API keys, no cloud calls, no model downloads at runtime. The audit log, the
enrolled descriptors and every captured image live in one local directory.

## Install

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
```

### 3. Ensure the face models are present

The two ONNX model files are already checked into this repository:
- `models/face_detection_yunet_2023mar.onnx` (232 KB)
- `models/face_recognition_sface_2021dec.onnx` (38.7 MB)

If they are ever missing, download them from:
- `face_detection_yunet`: https://huggingface.co/opencv/face_detection_yunet
- `face_recognition_sface`: https://huggingface.co/opencv/face_recognition_sface

Or set the `IDVERIFY_MODELS` environment variable to point at a directory
containing the two `.onnx` files.

### 4. (Linux/Ubuntu only) Install Tkinter for the GUI

```bash
sudo apt update
sudo apt install python3-tk
```

## Use

### Desktop application

```bash
python main.py            # Verify / Enroll / Records / History tabs
```

1. **Verify** — load the ID card (camera or file), load a live capture, press
   **VERIFY IDENTITY**. You get a green/red/amber verdict, both similarity
   scores, the parsed card fields and the reasons behind the decision.
2. **Enroll** — the live face must clearly match the printed portrait before
   anything is written to the database.
3. **Records** — enrolled identities, removable.
4. **History** — every verification attempt, timestamped.

### Command line

```bash
python main.py scan    --card card.jpg
python main.py enroll  --card card.jpg --face selfie.jpg
python main.py verify  --card card.jpg --face selfie.jpg
python main.py verify  --card card.jpg --camera 0      # live frame from the webcam
python main.py list
python main.py history --limit 20
python main.py forget  --id 4471-9938-2210
```

`verify` exit codes: `0` verified, `1` rejected, `2` needs review, `3` error —
so it drops straight into a shell script or an access-control hook.

Add `--home DIR` to any command to use a different data directory.

## How the decision is made

Both comparisons produce a cosine similarity between SFace descriptors:

| Score | Band |
|---|---|
| ≥ **0.45** | accept |
| **0.30 – 0.45** | needs manual review |
| < **0.30** | reject |

These were calibrated while building the system: the same person (card portrait
vs a full-resolution capture) scored **0.83**, different people scored
**0.05 – 0.10**. 0.45 sits above OpenCV's LFW-calibrated 0.363 floor with margin
for the lower quality of printed portraits; 0.30 stays far above the impostor
population. Thresholds and quality gates live in `idverify/config.py`.

Before any matching happens, captures must pass quality gates: detection
confidence ≥ 0.80 and a face at least 80 px across (60 px for the printed
portrait). A live capture containing more than one face is refused outright.

A card that is past its printed expiry date is rejected regardless of faces.

Policy when a record is missing: the check still runs against the printed
portrait, but the result is reported as `card portrait only (ID not in local
records)` so the operator can see that only possession was established.

## Layout

```
main.py                entry point (GUI by default, CLI via subcommands)
idverify/
  config.py            thresholds, quality gates, paths
  face.py              YuNet detection + SFace embedding + cosine matching
  idcard.py            OCR, reading-order sort, label parsing, date parsing, portrait crop
  store.py             SQLite: persons + verification audit log
  engine.py            IdentitySystem.enroll() / .verify() and the verdict logic
  camera.py            webcam capture (optional - every command also takes files)
  gui.py               Tkinter desktop interface
  cli.py               command line interface
models/                the two vendored ONNX face models
tests/                 pytest suite + synthetic ID-card fixture

## Liveness detection (printed-photo / phone-screen spoof guard)

A very simple but effective check: two consecutive webcam frames are captured
and their face-detection confidence scores are compared. A static photo or phone
screen will produce nearly identical confidence both times, while a live face
typically produces slightly different scores between frames (lighting, minor
movement, etc.).

- If the confidence delta is **below 0.10**, the check fails and the verdict is
  downgraded to **REVIEW** (the system asks the operator to confirm).
- If the delta is **0.10 or above**, the check passes normally.
- On headless CI or any environment without a webcam, the check is **automatically
  skipped** so existing tests and headless deploys are not broken.

This guard blocks simple printouts and phone-screen replays. For stronger
anti-spoofing (replay attacks, video playback), integrate a dedicated anti-spoof
model before deploying to a public-facing door-release hook.

**Limits worth knowing**
- **No liveness detection beyond the two-frame confidence-delta.** YuNet + SFace
  will happily match a photo held up to the camera without the two-frame check.
  Add a blink/head-turn challenge or an dedicated anti-spoof model before using
  this to open a door.
- **OCR quality depends on the photo.** Glare, skew and small fonts on real
  cards will read worse than the synthetic fixture; field parsing is label-driven
  (`NAME:`, `ID NUMBER:`, `DATE OF BIRTH:` …) so unusual layouts may need an
  extra pattern in `idverify/idcard.py:LABEL_RULES`.
- **Thresholds are a starting point.** Re-tune them on your own camera, lighting
  and card stock before trusting the verdicts.
- **Portrait selection:** when several faces appear in the card image, the
  highest-scoring one is used as the printed portrait and a warning is raised —
  confirm the crop in the preview.

## Use

### Desktop application

```bash
python main.py            # Verify / Enroll / Records / History tabs
```

1. **Verify** — load the ID card (camera or file), load a live capture, press
   **VERIFY IDENTITY**. You get a green/red/amber verdict, both similarity
   scores, the parsed card fields and the reasons behind the decision.
2. **Enroll** — the live face must clearly match the printed portrait before
   anything is written to the database.
3. **Records** — enrolled identities, removable.
4. **History** — every verification attempt, timestamped.

### Command line

```bash
python main.py scan    --card card.jpg
python main.py enroll  --card card.jpg --face selfie.jpg
python main.py verify  --card card.jpg --face selfie.jpg
python main.py verify  --card card.jpg --camera 0      # live frame from the webcam
python main.py list
python main.py history --limit 20
python main.py forget  --id 4471-9938-2210
```

`verify` exit codes: `0` verified, `1` rejected, `2` needs review, `3` error —
so it drops straight into a shell script or an access-control hook.

Add `--home DIR` to any command to use a different data directory.

## How the decision is made

Both comparisons produce a cosine similarity between SFace descriptors:

| Score | Band |
|---|---|
| ≥ **0.45** | accept |
| **0.30 – 0.45** | needs manual review |
| < **0.30** | reject |

These were calibrated while building the system: the same person (card portrait
vs a full-resolution capture) scored **0.83**, different people scored
**0.05 – 0.10**. 0.45 sits above OpenCV's LFW-calibrated 0.363 floor with margin
for the lower quality of printed portraits; 0.30 stays far above the impostor
population. Thresholds and quality gates live in `idverify/config.py`.

Before any matching happens, captures must pass quality gates: detection
confidence ≥ 0.80 and a face at least 80 px across (60 px for the printed
portrait). A live capture containing more than one face is refused outright.

A card that is past its printed expiry date is rejected regardless of faces.

Policy when a record is missing: the check still runs against the printed
portrait, but the result is reported as `card portrait only (ID not in local
records)` so the operator can see that only possession was established.

## Layout

```
main.py                entry point (GUI by default, CLI via subcommands)
idverify/
  config.py            thresholds, quality gates, paths
  face.py              YuNet detection + SFace embedding + cosine matching
  idcard.py            OCR, reading-order sort, label parsing, date parsing, portrait crop
  store.py             SQLite: persons + verification audit log
  engine.py            IdentitySystem.enroll() / .verify() and the verdict logic
  camera.py            webcam capture (optional - every command also takes files)
  gui.py               Tkinter desktop interface
  cli.py               command line interface
models/                the two vendored ONNX face models
tests/                 pytest suite + synthetic ID-card fixture
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite renders a synthetic ID card (real portrait, printed fields), then
exercises OCR field extraction, date/label parsing, face similarity, quality
gates, the store, enrolment refusal on mismatch, impostor rejection, expiry
handling, the audit log and the CLI exit codes.## Liveness detection (printed-photo / phone-screen spoof guard)

A very simple but effective check: two consecutive webcam frames are captured
and their face-detection confidence scores are compared. A static photo or phone
screen will produce nearly identical confidence both times, while a live face
typically produces slightly different scores between frames (lighting, minor
movement, etc.).

- If the confidence delta is **below 0.10**, the check fails and the verdict is
  downgraded to **REVIEW** (the system asks the operator to confirm).
- If the delta is **0.10 or above**, the check passes normally.
- On headless CI or any environment without a webcam, the check is **automatically
  skipped** so existing tests and headless deploys are not broken.

This guard blocks simple printouts and phone-screen replays. For stronger
anti-spoofing (replay attacks, video playback), integrate a dedicated anti-spoof
model before deploying to a public-facing door-release hook.

**Limits worth knowing**
- **No liveness detection beyond the two-frame confidence-delta.** YuNet + SFace
  will happily match a photo held up to the camera without the two-frame check.
  Add a blink/head-turn challenge or an dedicated anti-spoof model before using
  this to open a door.
- **OCR quality depends on the photo.** Glare, skew and small fonts on real
  cards will read worse than the synthetic fixture; field parsing is label-driven
  (`NAME:`, `ID NUMBER:`, `DATE OF BIRTH:` …) so unusual layouts may need an
  extra pattern in `idverify/idcard.py:LABEL_RULES`.
- **Thresholds are a starting point.** Re-tune them on your own camera, lighting
  and card stock before trusting the verdicts.
- **Portrait selection:** when several faces appear in the card image, the
  highest-scoring one is used as the printed portrait and a warning is raised —
  confirm the crop in the preview.