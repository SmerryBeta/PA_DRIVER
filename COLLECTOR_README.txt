PA_DRIVER collector package

Purpose:
Collect 30 FPS screenshot + keyboard data for kart driving.

First setup:
1. Install Python 3.10 or 3.11 from python.org.
2. Double-click setup_collector_env.bat.

Collect data:
1. Open the game at 1920x1080.
2. Enter the target track and make sure the game window is visible.
3. Double-click collect_gameplay_30fps.bat.
4. Press F8 to start/pause recording.
5. Drive normally:
   W forward, A left, D right, S reverse, Space drift, J item.
6. Press F9 to save and quit.

Output:
Data is saved under the data folder as party_kart_30fps_01.h5.
If that file already exists, the collector automatically writes party_kart_30fps_01_001.h5, then _002, etc.

Recommended recording:
- Prefer stable, clean laps.
- Avoid chat windows, menus, overlays, or alt-tab during recording.
- If you crash badly, pause with F8 or send the file separately and mark it as possibly dirty.

Send back:
Send the .h5 files from the data folder.
