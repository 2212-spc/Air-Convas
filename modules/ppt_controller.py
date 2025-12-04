import logging

try:
    import pyautogui
except Exception:  # pragma: no cover - optional on headless envs
    pyautogui = None
    logging.warning("pyautogui not available; PPT control disabled.")


class PPTController:
    def __init__(self) -> None:
        self.enabled = pyautogui is not None

    def _press(self, key: str) -> None:
        if not self.enabled:
            return
        pyautogui.press(key)

    def next_slide(self) -> None:
        self._press("right")

    def prev_slide(self) -> None:
        self._press("left")

    def first_slide(self) -> None:
        self._press("home")

    def last_slide(self) -> None:
        self._press("end")
