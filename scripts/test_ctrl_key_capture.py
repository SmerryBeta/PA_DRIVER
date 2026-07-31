from __future__ import annotations

import time
import sys
from pathlib import Path

from pynput import keyboard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_gameplay_corrections import CorrectionKeyState, compact_keys


def main() -> None:
    state = CorrectionKeyState()
    listener = keyboard.Listener(on_press=state.on_press, on_release=state.on_release)
    listener.start()

    print("Hold Left Ctrl, then press W/A/S/D/Space.")
    print("This should show human_control=True and the pressed driving keys.")
    print("Press F9 to quit.")

    try:
        while not state.stop:
            print(
                f"human_control={state.human_control} keys={compact_keys(state.labels())}",
                end="\r",
            )
            time.sleep(0.1)
    finally:
        listener.stop()
        print()


if __name__ == "__main__":
    main()
