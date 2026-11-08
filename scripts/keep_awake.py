"""Streamlit Community Cloud 앱을 주기적으로 방문해 휴면을 막는다.

단순 HTTP 요청은 방문으로 집계되지 않아 실제 브라우저(Playwright)로 접속한다.
이미 잠들어 있으면 깨우기 버튼을 눌러 앱이 뜰 때까지 기다린다.
앱이 끝내 뜨지 않으면 종료 코드 1 → GitHub Actions가 실패 알림을 보낸다.
"""

import os
import re
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://stovelens-ai.streamlit.app/")
WAKE_TIMEOUT_SEC = int(os.environ.get("WAKE_TIMEOUT_SEC", "300"))
POLL_SEC = 5
SCREENSHOT_PATH = os.environ.get("SCREENSHOT_PATH", "keep_awake_last.png")

APP_READY_SELECTOR = '[data-testid="stAppViewContainer"]'
WAKE_BUTTON_PATTERN = r"get this app back up|wake"


def app_is_ready(page) -> bool:
    for frame in page.frames:
        try:
            if frame.locator(APP_READY_SELECTOR).count() > 0:
                return True
        except Exception:
            continue
    return False


def click_wake_button_if_present(page) -> bool:
    for frame in page.frames:
        try:
            button = frame.get_by_role("button", name=re.compile(WAKE_BUTTON_PATTERN, re.I))
            if button.count() > 0 and button.first.is_visible():
                button.first.click()
                return True
        except Exception:
            continue
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"[keep-awake] visiting {APP_URL}")
        try:
            page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        except PlaywrightTimeout:
            print("[keep-awake] initial load timed out, continuing to poll")

        deadline = time.time() + WAKE_TIMEOUT_SEC
        clicked = False
        while time.time() < deadline:
            if app_is_ready(page):
                print(f"[keep-awake] app is up ({page.url})")
                browser.close()
                return 0
            if not clicked and click_wake_button_if_present(page):
                clicked = True
                print("[keep-awake] app was asleep — clicked wake button, waiting for boot")
            time.sleep(POLL_SEC)

        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        print(f"[keep-awake] FAILED: app not ready after {WAKE_TIMEOUT_SEC}s (url={page.url}), screenshot={SCREENSHOT_PATH}")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
