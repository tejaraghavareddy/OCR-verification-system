# IDVerify — Offline + Online ID Card Verification

A simple desktop app that checks whether the person standing in front of the camera is the same person whose ID card you have. It reads the card with OCR, finds the face printed on it, and compares it to the person in front of the camera. Everything runs **completely offline by default** — no internet, no servers, no API keys. All the data stays on your computer.

You can use it two ways:
1. A **visual window** with buttons and tabs (the "GUI").
2. **Typed commands** in a terminal (the "command line").

It also has a special **attendance mode** where the camera opens automatically and walks you through taking a card photo and a face photo, then saves the result to a daily attendance list.

---

## Table of Contents

1. [What the app does](#what-the-app-does)
2. [Before you start — what you need](#before-you-start--what-you-need)
3. [Step-by-step setup (first time only)](#step-by-step-setup-first-time-only)
4. [How to use it — the visual app](#how-to-use-it--the-visual-app)
5. [How to use it — typed commands](#how-to-use-it--typed-commands)
6. [Live attendance mode (camera opens automatically)](#live-attendance-mode-camera-opens-automatically)
7. [How the check works (in simple words)](#how-the-check-works-in-simple-words)
8. [The results explained](#the-results-explained)
9. [Where your data is saved](#where-your-data-is-saved)
10. [Offline vs online — how to make it work both ways](#offline-vs-online--how-to-make-it-work-both-ways)
11. [Things to keep in mind](#things-to-keep-in-mind)
12. [Quick cheat sheet](#quick-cheat-sheet)

---

## What the app does

| Task | What happens |
|---|---|
| **Verify a person** | You show the app a photo of an ID card and a photo (or live camera shot) of the person. The app says **VERIFIED**, **REJECTED**, or **NEEDS REVIEW**. |
| **Enroll a new person** | You add someone to the database for the first time. The app saves their ID number, name, and face details. It refuses if the live face does not match the portrait on the card. |
| **Take attendance** | The camera opens, you take a photo of the card and then the person, and the app records it in today's attendance list. You can also add new people automatically with `--enroll`. |
| **View records** | See everyone already enrolled, or see the history of every check that was done. |

---

## Before you start — what you need

| What you need | Why |
|---|---|
| **A computer** (Windows, Mac, or Linux) | The app runs on your computer. |
| **Python 3.10 or newer** | The app is written in Python. Download it from [python.org](https://www.python.org/downloads/) if you don't have it. |
| **A camera (webcam)** | Needed for the attendance mode and for taking live face photos. You can also use existing photos instead. |
| **ID card photos** | Photos of the ID cards you want to verify. You can take these with your phone or camera. |

That's it. No internet connection needed after setup. No accounts to create. No keys to get.

---

## Step-by-step setup (first time only)

Do these steps once. After this, you never need to do them again.

### Step 1: Open a terminal in the project folder

- **Windows:** Open Command Prompt or PowerShell, then go to the folder. For example:
  ```
  cd C:\Users\YourName\Desktop\OCR-verification-system
  ```
- **Mac/Linux:** Open the Terminal, then go to the folder. For example:
  ```
  cd ~/Desktop/OCR-verification-system
  ```

### Step 2: Create a clean workspace (a "virtual environment")

This keeps the app's packages separate from anything else on your computer.

```
python -m venv .venv
```

### Step 3: Turn on the workspace

- **Windows:**
  ```
  .venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```
  source .venv/bin/activate
  ```

You should see `(.venv)` appear at the start of your command line. That means it worked.

### Step 4: Install the app's packages

This downloads everything the app needs. The file `requirements.txt` lists all of them.

- **Windows:**
  ```
  .venv\Scripts\pip install -r requirements.txt
  ```
- **Mac/Linux:**
  ```
  .venv/bin/pip install -r requirements.txt
  ```

### Step 5 (Mac/Linux only): Install the visual toolkit

If you want to use the **window with buttons** (the GUI) on Linux, you also need one extra package called `tkinter`:

```
sudo apt update
sudo apt install python3-tk
```

On Windows and Mac, this is already included with Python.

### Step 6: Make sure the face models are there

The app needs two small files to recognize faces. They are already included in this project under the `models/` folder:

- `models/face_detection_yunet_2023mar.onnx` (finds faces — 232 KB)
- `models/face_recognition_sface_2021dec.onnx` (recognizes faces — 38.7 MB)

If those files are ever missing, you would download them from:
- Face detection: https://huggingface.co/opencv/face_detection_yunet
- Face recognition: https://huggingface.co/opencv/face_recognition_sface

But for normal use, they should already be there. You don't need to do anything.

### Done!

Now you can run the app. No internet needed. The OCR (text-reading) and face models are already included — they come with the installed packages.

---

## How to use it — the visual app

This is the version with windows, buttons, and tabs. It's the easiest for most people.

### Start it

In your terminal (with the workspace turned on), type:

```
python main.py
```

A window opens.

### The window layout

The window has **4 tabs** across the top:

| Tab | What it's for |
|---|---|
| **Verify** | Check a person against a card |
| **Enroll** | Add a new person to the database |
| **Records** | See everyone already enrolled |
| **History** | See every check ever made |

At the top-right of the window, there's a small box that says **Camera** with a number (usually 0). That's which camera to use. Change it if you have more than one camera.

At the bottom, there's a status bar that shows what's happening.

### How to verify a person (the Verify tab)

1. Click the **Verify** tab.
2. You'll see two panels: **"1 - ID card"** and **"2 - Live capture"**.
3. In the **ID card** panel:
   - Click **Camera** to take a photo of the card with your webcam, OR
   - Click **File** to pick a photo of the card from your computer.
4. In the **Live capture** panel:
   - Click **Camera** to take a photo of the person with your webcam, OR
   - Click **File** to pick a photo of the person from your computer.
5. Click the big **VERIFY IDENTITY** button.

The app shows you:
- The result: **IDENTITY VERIFIED** (green), **VERIFICATION FAILED** (red), or **NEEDS MANUAL REVIEW** (amber).
- Two scores: **Card portrait check** and **Local record check**.
- The details read from the card: ID number, name, date of birth, expiry date.
- A list of reasons and any warnings.

### How to enroll a new person (the Enroll tab)

1. Click the **Enroll** tab.
2. Load the ID card photo (Camera or File) in the first panel.
3. Load the person's live photo (Camera or File) in the second panel.
4. Click **ENROLL IDENTITY**.

The app checks that the live face matches the portrait on the card. If it does, it saves the person to the database. If it doesn't match, it refuses and tells you why.

### How to see your records (the Records tab)

1. Click the **Records** tab.
2. Click **Refresh** to see the latest list.
3. You'll see each person's ID number, name, date of birth, expiry date, and when they were last updated.
4. To remove someone, select their row and click **Forget selected**.

### How to see the history (the History tab)

1. Click the **History** tab.
2. Click **Refresh**.
3. You'll see every check ever made: when it happened, the result (green/red/amber), the ID number, the name, and the scores.

### Using the camera in any tab

In the **Verify**, **Enroll**, and other panels, each input box has two buttons:
- **Camera** — takes one photo from the webcam.
- **File** — loads a photo from your computer.

You can mix and match. For example, take the card photo with the camera and load the face photo from a file.

---

## How to use it — typed commands

If you prefer typing commands instead of clicking buttons, you can do everything from the terminal.

Make sure the workspace is turned on (`.venv\Scripts\activate` or `source .venv/bin/activate`), then use these commands.

### Check what's on a card (read the text)

```
python main.py scan --card photo_of_card.jpg
```

This reads the card and prints the name, ID number, date of birth, expiry date, sex, and nationality. It doesn't compare faces — it just reads the text.

### Add a new person to the database

```
python main.py enroll --card photo_of_card.jpg --face photo_of_person.jpg
```

This reads the card, checks that the face matches the portrait, and saves the person if everything is OK.

### Verify a person

```
python main.py verify --card photo_of_card.jpg --face photo_of_person.jpg
```

This checks the person and tells you VERIFIED, REJECTED, or NEEDS REVIEW.

### Verify using the live camera (instead of a face photo file)

```
python main.py verify --card photo_of_card.jpg --camera 0
```

This reads the card from a file, then opens the camera for the face. Press Enter when ready. Use `--camera 1` if you have a second camera.

### See everyone enrolled

```
python main.py list
```

### See the history of all checks

```
python main.py history
```

Add `--limit 20` to see only the last 20.

### Remove someone from the database

```
python main.py forget --id 1234-5678-9012
```

Replace `1234-5678-9012` with the actual ID number.

### Save your data in a different folder

By default, everything is saved in a folder called `.idverify` in your user home folder. To use a different folder, add `--home`:

```
python main.py attend --home C:\my-idverify-data
```

---

## Live attendance mode (camera opens automatically)

This is the mode where the app takes over the camera and walks you through the whole process step by step. It's made for situations like taking attendance in a classroom, office, or event.

### Start it

```
python main.py attend
```

The camera opens. You'll see prompts on the screen.

### What happens step by step

1. **Step 1 — Card photo:** The app asks you to point the camera at the ID card and press **Enter**. It takes the card photo.
2. **Step 2 — Face photo:** The app asks you to point the camera at the person and press **Enter**. It takes the face photo.
3. The app verifies the person automatically.
4. The result is saved to today's attendance list.

To quit at any step, type `q` and press Enter.

### Options

| Command | What it does |
|---|---|
| `python main.py attend` | Basic attendance — captures card + face, saves the result. |
| `python main.py attend --enroll` | Same, but also adds the person to the database if they're new (first time only). |
| `python main.py attend --camera 1` | Use a different camera (if you have more than one). |
| `python main.py attend --list` | Show today's attendance list. |
| `python main.py attend --export today.csv` | Save today's attendance list as a CSV file (opens in Excel). |
| `python main.py attend --clear` | Erase today's attendance list to start fresh. |

### Using attendance mode in a real setting

- **Classroom:** Run `python main.py attend --enroll`. For each student, point the camera at their ID card, press Enter, then point it at their face, press Enter. If they're new, they get added automatically. If they're already in the system, it just records their attendance.
- **Office / event:** Same flow. Use `--list` to see who attended today. Use `--export attendance.csv` to open the list in Excel.
- **Start of day:** Run `python main.py attend --clear` to wipe yesterday's list and start fresh.

### What gets saved

Every attendance attempt is saved, whether it passed or failed. For each attempt, the app saves:
- The time
- The result (VERIFIED, REJECTED, or NEEDS REVIEW)
- The ID number and name (if read from the card)
- The two face scores
- Whether the card was expired
- The reason for the result
- The photo of the card and the photo of the face (saved as image files)

This means you can look back later and see exactly what happened.

---

## How the check works (in simple words)

The app does three things, in order:

### 1. Reads the card (OCR)

The app looks at the photo of the ID card and reads the text on it. It finds:
- The name
- The ID number
- The date of birth
- The expiry date
- The sex
- The nationality

This uses an OCR engine (PP-OCRv6) that's already included in the installed packages.

### 2. Finds the face on the card

The app looks for a face in the card photo — usually the passport-style portrait. It crops it out so it can compare it later.

If there are multiple faces on the card, it picks the clearest one and warns you.

### 3. Compares faces

The app uses a face-recognition model (SFace) to turn faces into numbers (called "embeddings"). Then it compares:

| Comparison | What it answers |
|---|---|
| **Live face vs printed portrait** | "Is the person holding their own card?" |
| **Live face vs enrolled record** | "Is this the same person we saved before?" (only if the ID is already in the database) |

Both comparisons give a **similarity score** from 0 to 1. The higher the number, the more similar the faces are.

### Quality checks before comparing

Before any comparison happens, the app checks:
- The face was found clearly enough (detection confidence of at least 0.80).
- The face is big enough in the photo (at least 80 pixels across for a live face, 60 pixels for the printed portrait).
- There is only one face in the live photo (if there are multiple, it refuses).

If any of these fail, the app tells you.

### Expired cards

If the card has an expiry date and that date has passed, the app automatically says **REJECTED**, no matter how good the face match is.

---

## The results explained

### VERIFIED (green)

Everything matched. The person's face matches the portrait on the card, and (if they're in the database) it also matches their saved record. The card is not expired.

### REJECTED (red)

Something didn't match:
- The face didn't match the portrait on the card.
- The face didn't match the saved record (if enrolled).
- The card is expired.
- No face was found, or the face was too small or unclear.

### NEEDS REVIEW (amber)

The scores are in a middle zone — not clearly a match, not clearly a mismatch. A human should look at it. This can happen when:
- The face is a partial match.
- The lighting is poor.
- The card photo is low quality.

### What the scores mean

The app uses a **cosine similarity score**. In simple terms:

| Score | What it means |
|---|---|
| **0.45 or higher** | Match — accept it. |
| **Between 0.30 and 0.45** | Uncertain — needs a human to look. |
| **Below 0.30** | No match — reject it. |

These numbers were chosen based on testing: the same person's face usually scores 0.8 or higher, while different people usually score below 0.1. The cutoffs (0.45 and 0.30) sit safely between those two groups.

You can change these numbers in the file `idverify/config.py` if your camera or lighting is different.

---

## Where your data is saved

All your data is saved **on your computer**, in one place.

### Where is it?

By default, it's in a folder called `.idverify` in your user home folder:
- **Windows:** `C:\Users\YourName\.idverify\`
- **Mac/Linux:** `/Users/YourName/.idverify/` or `/home/YourName/.idverify/`

You can find your exact home folder by opening a terminal and typing `echo %USERPROFILE%` (Windows) or `echo ~` (Mac/Linux).

### What's inside?

```
.idverify/
  idverify.db          ← the database (all people, checks, attendance)
  images/
    ...card photos, face photos, and portraits saved during attendance...
```

### The database

The database is a single file called `idverify.db`. It's a SQLite database — a simple, self-contained database that's just one file.

It has three tables:

1. **persons** — everyone enrolled. Stores their ID number, name, date of birth, expiry date, sex, nationality, and their face data.
2. **verifications** — every check ever made. Stores the time, result, ID number, name, scores, reasons, and warnings.
3. **attendances** — every attendance attempt. Stores the same kind of info as verifications, plus the paths to the saved card and face photos.

### Can I move or back up the data?

Yes. Since it's just a folder with a database file and some images, you can:
- Copy the `.idverify` folder to back it up.
- Move it to another computer to transfer your data.
- Delete it to start fresh (everything will be erased).

If you use a different folder with `--home`, that folder has the same structure.

### How to clear today's attendance

If you want to start the attendance list fresh (for example, at the start of a new day):

```
python main.py attend --clear
```

This only deletes today's attendance entries — it does not delete enrolled people or past history.

---

## Offline vs online — how to make it work both ways

### Offline (the default — works right now, no internet needed)

The app is built to work **completely offline**. Here's what that means:

- **No internet needed.** After you install the packages once, you never need to go online again to use the app.
- **No servers.** There's no server running anywhere. Everything happens on your computer.
- **No API keys.** You don't need to sign up for anything or get any keys.
- **No model downloads.** The OCR and face models are already included. They come with the installed packages.

The data is stored in a local database file on your computer (`idverify.db`). The camera, the text reading, the face matching — all of it happens on your machine.

This is useful when:
- You're in a room with no Wi-Fi (exam hall, field site, basement office).
- You're on a closed network (security-conscious environment).
- You just want a simple tool that works without setting up servers.

### Online (if you want to use it across multiple devices or store data centrally)

Right now, the app is offline-only by default. If you want to make it work online — for example, so multiple devices share the same database, or so the data is stored in the cloud — here's how that would work.

**Important:** The online part is **not built in yet**. The app itself doesn't have a "go online" button. But the design makes it possible to add. Here are the ways you could do it:

#### Option 1: Use a central database (best for multiple devices on a network)

The app's storage is all in one place — a file called `store.py`. That file is responsible for saving and loading data. To go online, you would:

1. Set up a networked database (like PostgreSQL or MySQL) on a server that all your devices can reach.
2. Rewrite `store.py` to talk to that database instead of the local SQLite file.
3. Keep the same structure (the same tables: persons, verifications, attendances) so nothing else breaks.

The rest of the app — the camera, the OCR, the face matching, the buttons, the commands — doesn't change at all. Only the part that saves and loads data changes.

#### Option 2: Keep it local, but sync to the cloud when online

If you want each device to keep working offline (when there's no internet) and then sync up when there is internet:

1. Each device keeps its own local `idverify.db` (so it works offline).
2. You add a "sync" step that sends new records to a central place when internet is available.
3. You also pull down new enrollments from the central place, so a device can recognize people who were enrolled on a different device.

The local database is the main copy while offline. Syncing is a background thing that happens when there's a connection.

#### Option 3: Build a sync server

If you want a central server that all devices talk to, the simplest version would have a few endpoints (URLs that the app calls):

- `POST /enroll` — send a new person's details to the server.
- `GET /persons` — get the list of everyone enrolled (so the local device knows who to check against).
- `POST /attendance` — send an attendance record to the server.
- `GET /attendance?date=2026-11-03` — get the attendance list for a specific day.

Each device keeps a local copy of the enrolled people (so the face check still works offline), and updates that copy when it syncs.

### What does NOT need to change when going online

No matter which online option you choose, these parts stay exactly the same:
- The camera capture (taking photos from the webcam).
- The OCR (reading text from the card).
- The face matching (comparing faces).
- The thresholds and quality checks.
- The visual app (the window with buttons).
- The command-line commands.

Only the storage part changes. That's by design — it makes it easy to add online later without rebuilding the whole app.

### Bottom line

- **Right now:** the app is 100% offline. It works on any computer with a camera, no internet, no setup beyond the initial install.
- **Later, if you want online:** you can add it by changing one file (`store.py`) or adding a sync step. The rest of the app doesn't need to change.

---

## Things to keep in mind

### Camera needed for live features

The attendance mode and the `--camera` option need a working camera. If there's no camera (for example, on a server or some cloud environments), those parts simply skip the camera and work with photos from files instead. The rest of the app still works fine.

### Photo quality matters

- **Glare on the card** can make the text hard to read.
- **Blurry photos** can make the face hard to recognize.
- **Bad angle** (face turned away) can lower the similarity score.
- **Small card photos** might not have enough detail.

Try to take clear, well-lit photos for best results.

### The thresholds are a starting point

The numbers used to decide "match" vs "no match" (0.45 and 0.30) were chosen based on testing with the developer's camera and lighting. If your setup is different, you might want to adjust them. You can find them in `idverify/config.py`.

### Not a complete security system (yet)

This app checks faces, but it doesn't do everything a high-security system would do. For example:
- It has a basic check against someone holding up a photo of a photo (the two-frame liveness check), but it's not as strong as dedicated anti-spoofing.
- It doesn't stop someone from using a video on a phone screen (though the basic check helps against simple cases).

If you're using this for something important (like opening a secure door), you should add stronger checks first.

### Multiple faces on a card

If the card photo has more than one face (for example, a card with a group photo), the app picks the clearest one and warns you. Check the preview to make sure it picked the right face.

### Unusual card layouts

The app reads cards by looking for labels like "NAME:", "ID NUMBER:", "DATE OF BIRTH:". If your card uses different labels or a different layout, you might need to add the new labels to the app's label list (in `idverify/idcard.py`).

---

## Quick cheat sheet

### First-time setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Mac/Linux
.venv\Scripts\pip install -r requirements.txt     # Windows
.venv/bin/pip install -r requirements.txt         # Mac/Linux
```

(Mac/Linux GUI only: also run `sudo apt install python3-tk`)

### Open the visual app

```
python main.py
```

### Common commands

```
# Read a card's text
python main.py scan --card card.jpg

# Add a new person
python main.py enroll --card card.jpg --face person.jpg

# Verify a person (from files)
python main.py verify --card card.jpg --face person.jpg

# Verify a person (card file + live camera)
python main.py verify --card card.jpg --camera 0

# See everyone enrolled
python main.py list

# See check history
python main.py history

# Remove a person
python main.py forget --id 1234-5678-9012

# Attendance: capture card + face, record result
python main.py attend

# Attendance + add new people automatically
python main.py attend --enroll

# Attendance: use a different camera
python main.py attend --camera 1

# Attendance: see today's list
python main.py attend --list

# Attendance: save today's list as CSV
python main.py attend --export today.csv

# Attendance: clear today's list
python main.py attend --clear

# Use a different data folder
python main.py attend --home C:\my-data
```

---

## How the pieces fit together (for curious readers)

This section is for people who want to know how the app is built. You don't need to read this to use the app.

```
main.py              — the starting point. Opens the visual app or runs commands.
idverify/
  config.py          — settings: the score cutoffs, minimum face sizes, file locations.
  face.py            — finds faces (YuNet) and recognizes them (SFace), compares two faces.
  idcard.py          — reads the card text (OCR), figures out which text is which field, finds the portrait.
  store.py           — saves and loads data from the database (people, checks, attendance).
  engine.py          — the main logic: enrolls people, verifies them, decides the result.
  camera.py          — talks to the webcam to take photos.
  attend.py          — the attendance mode: opens the camera and walks you through it.
  gui.py             — the visual window with tabs and buttons.
  cli.py             — the command-line interface (the typed commands).
  util.py            — helper functions: loading/saving images, timestamps, folders.
models/              — the two face model files (already included).
tests/               — automatic tests that check the app works correctly.
```

### How a verification flows through the app

1. You give it a card photo and a face photo (or use the camera).
2. `engine.py` sends the card photo to `idcard.py` to read the text and find the portrait.
3. `engine.py` sends the face photo to `face.py` to find and describe the face.
4. `engine.py` compares the two faces using `face.py`.
5. `engine.py` checks the database (via `store.py`) to see if the ID number is already enrolled.
6. `engine.py` decides the result (VERIFIED / REJECTED / NEEDS REVIEW).
7. `engine.py` saves the result to the database via `store.py`.
8. The result is shown to you (in the window or printed in the terminal).

---

## Running the tests (for developers)

If you're working on the app and want to make sure nothing is broken, run the automatic tests:

```
.venv/bin/python -m pytest tests/ -q     # Mac/Linux
.venv\Scripts\python -m pytest tests/ -q  # Windows
```

These tests create a fake ID card, then check that the app reads it correctly, matches faces correctly, saves data correctly, and so on. There are 46 tests in total.

---

## License

This project is open source. See the project's license file for details.
