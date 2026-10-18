# PA_DRIVER

<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/0d905b99-ec1d-4bc0-9765-94a125777d88" />

PA_DRIVER is a local behavior-cloning experiment for driving a kart game from screen pixels and keyboard input. It records human gameplay, trains a lightweight CNN policy, and runs the trained model back in the game through keyboard control.

This project is designed for private/local testing rooms. It does not use reinforcement learning, game memory reading, packet interception, or game file modification.

<img width="995" height="558" alt="image" src="https://github.com/user-attachments/assets/c6b25733-36d5-4ce5-8fae-1eb3b2169d8d" />

How this model perform? See https://www.bilibili.com/video/BV1jt3M6jEvz/

## What It Does

- Collects gameplay frames from the screen and records keyboard labels.
- Trains a supervised imitation-learning model from `.h5` driving data.
- Supports correction collection: let the model drive, hold `Left Ctrl` to take over, and save only human correction frames.
- Runs a full app loop with image-template UI automation from `img/*.png`.
- Hot-reloads runtime settings from `config.yml`.
- Provides review, trimming, batch-cleaning, and H5 combining tools.
- 去他妈的快速刷卡丁等级，恶心之极！

## Current Controls

The real-game driving labels are:

| Label | Key | Meaning |
| --- | --- | --- |
| `w_forward` | `W` | forward |
| `a_left` | `A` | turn left |
| `d_right` | `D` | turn right |
| `s_reverse` | `S` | reverse / recover |
| `space_drift` | `Space` | drift / jump |
| `j_item` | `J` | item, collected but usually ignored by training |

Global hotkeys:

- `F8`: start / pause recording, model control, or app workflow.
- `F9`: save and quit.
- `Left Ctrl`: in correction collection, hold to take manual control from the model.

## Requirements

- Windows
- Python 3.10 or 3.11
- CPU is enough for the small models used here
- A running game window
- Templates in `img/` for UI automation

Install Python dependencies:

```bat
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Main dependencies include PyTorch, OpenCV, HDF5, MSS screen capture, pynput, PyYAML, and watchdog.

## Project Layout

```text
PA_DRIVER/
  img/                         UI template images for OpenCV matching
  pa_driver/                   model and preprocessing code
  scripts/                     data collection, training, review, trim tools
  src/                         app runtime modules
  data/                        local driving datasets, not recommended for git
  runs/                        trained model checkpoints, not recommended for git
  config.yml                   runtime configuration
  run.py                       full app entrypoint
  COMMANDS.md                  detailed local command reference
```

## Quick Start

1. Install dependencies.
2. Collect normal driving data.
3. Train a model.
4. Test the model manually.
5. Collect correction data.
6. Train a corrected model.
7. Run the full app workflow.

The most useful command reference is in [COMMANDS.md](COMMANDS.md).

## Collect Gameplay

Use the 30 FPS collector for normal human driving:

```bat
collect_gameplay_30fps.bat
```

No-preview version:

```bat
collect_gameplay_30fps_no_preview.bat
```

Workflow:

1. Start the bat file.
2. Put the game window in the expected position/resolution.
3. Press `F8` to start recording.
4. Drive normally.
5. Press `F8` to pause if needed.
6. Press `F9` to save and quit.

Output goes to `data/*.h5`. Existing files are not overwritten; suffixes like `_001` are added automatically.

## Review And Clean Data

Open the H5 tools shell:

```bat
h5_tools.bat
```

Useful commands inside it:

```bat
stats data\party_kart*.h5
review data\party_kart_01.h5
trim data\party_kart_01.h5
batch_trim
combineh5 data\cleaned\*.h5 --out data\combined_cleaned.h5
```

Review hotkeys:

- `Space`: play / pause
- `A` / `D`: previous / next frame
- `W` / `S`: jump forward / backward
- `B`: mark bad range start
- `E`: mark bad range end
- `U`: undo last range
- `Q`: save ranges and quit

## Train

Train from original gameplay data:

```bat
train_party_kart.bat
```

Train from cleaned data:

```bat
train_party_kart_cleaned.bat
```

Train with corrections:

```bat
train_party_kart_with_corrections.bat
```

Recommended path:

```bat
batch_trim_h5.bat
train_party_kart_cleaned.bat
train_party_kart_cleaned_with_corrections.bat
```

The trainer uses multi-label supervised learning over keyboard states. It is not reinforcement learning.

## Run A Model

Base runtime:

```bat
run_party_kart_policy.bat
```

Corrected model runtime:

```bat
run_party_kart_policy_corrected.bat
```

Dry-run preview:

```bat
preview_party_kart_policy.bat
```

## Collect Corrections

Use this when the model can drive somewhat, but fails on walls, edges, recovery, or special cases:

```bat
collect_party_kart_corrections.bat
```

Workflow:

1. Start the bat file.
2. Press `F8` to enable model control.
3. When the model makes a mistake, hold `Left Ctrl`.
4. While holding `Left Ctrl`, drive manually with `W/A/S/D/Space`.
5. Release `Left Ctrl` to return control to the model.
6. Press `F9` to save and quit.

Only manual takeover frames are saved.

## Full App Loop

The app runtime combines UI template matching and model driving:

```bat
run_app.bat
```

`run.py` checks the screen:

- If `diy` is found, it creates and configures a room.
- If `start` is found, it starts the race loop.
- Otherwise, it starts the model directly.

During the race loop:

- Clicks `start`.
- Handles `continue` and `space` prompts.
- Starts model driving.
- Stops after `finish_all_race`.
- Handles optional reward popups like `prize_open_all` and `prize_continue`.
- Waits for `start` and loops again.

## Configuration

Runtime settings live in `config.yml` and are watched while the app is running.

Common fields:

```yaml
image_dir: img
model_path: runs/party_kart_bc_corrected.pt
capture_fps: 30.0
model_fps: 30.0
match_threshold: 0.82
step_delay: 1
```

Screen crop used by the model:

```yaml
model_crop:
  left: 0
  top: 170
  width: 1920
  height: 800
```

Model thresholds:

```yaml
model_no_space: false
model_space_threshold: 0.55
model_turn_threshold: 0.65
model_w_threshold: 0.50
model_w_release_threshold: 0.40
model_s_threshold: 0.45
```

Animated UI buttons can be stabilized before clicking:

```yaml
animated_templates:
  - start
  - create
  - lock
animated_stable_frames: 6
animated_stable_tolerance_px: 3
animated_settle_delay: 0.3
```

If a sliding button is clicked too early, increase `animated_stable_frames` or `animated_settle_delay`.

## UI Templates

The app automation reads `.png` templates from `img/` and matches them against the screen with OpenCV.

Current expected templates include:

```text
diy, create, lab, cart, select_non, formula, create_confirm,
lock, race_setting, default_circle, single_circle, apply,
start, continue, space, finish_all_race,
prize_open_all, prize_continue,
diy_room_home, return
```

If your resolution, UI scale, language, or game theme changes, recapture the affected templates.

## Data Notes

Behavior cloning is sensitive to bad labels. Clean data usually matters more than just adding more data.

Recommended mix:

- 80-90% normal stable driving
- 10-20% corrections

Avoid training on polluted clips where the car is stuck, driving into a wall, spinning, or where keys were recorded incorrectly.

## Sharing The Collector

Build a collector package:

```bat
package_collector.bat
```

Send:

```text
dist_collector\PA_DRIVER_COLLECTOR.zip
```

Other people can unzip it, install Python, run setup, collect `.h5` files, and send them back.

## GitHub Notes

Do not commit large or private local files unless you intentionally want to publish them:

```text
data/
runs/
logs/
build/
dist/
dist_collector/
.venv/
```

The repository can contain code, bat files, config examples, and template images. Datasets and trained models are usually better distributed separately.

## Disclaimer

This project is for local automation and imitation-learning experiments. Use it only in environments where automation is allowed. The code does not bypass anti-cheat systems and should not be used in public or competitive games without permission.

## Contributor

SmerryBeta: Training data collection & Model trainer & Designer of PA_DRIVER.

ChengXiaoXing(澄小星): Training data collection (but due to its high resolution, it was not collected, but we also appreciate his support)
