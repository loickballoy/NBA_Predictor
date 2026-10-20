"""
Captures d'écran de la démo pour le README : `python -m demo.screenshots [--site docs]`

Sert le dossier exporté par demo.export en local et le photographie avec Playwright
(pip install playwright, puis `playwright install chromium` si aucun Chromium n'est présent).
Écrit docs/screenshots/{home,player,player-mobile,dark,model}.png.
"""
import argparse
import functools
import http.server
import os
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(directory: Path) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def open_player(page, url: str) -> None:
    page.goto(url)
    page.locator("#featured .chip").first.click()
    page.locator("#explain p").first.wait_for()
    page.locator("#result").scroll_into_view_if_needed()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="docs", help="dossier produit par demo.export")
    parser.add_argument("--out", default="docs/screenshots")
    args = parser.parse_args()

    site, out = Path(args.site), Path(args.out)
    if not (site / "data.json").exists():
        raise SystemExit(f"{site}/data.json absent : lance d'abord python -m demo.export --out {site}")
    out.mkdir(parents=True, exist_ok=True)

    server = serve(site)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    with sync_playwright() as pw:
        # CHROMIUM_PATH : utiliser un Chromium déjà installé plutôt que celui de Playwright.
        browser = pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))

        desktop = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=2)
        desktop.goto(url)
        desktop.locator("#featured .chip").first.wait_for()
        desktop.screenshot(path=out / "home.png")
        open_player(desktop, url)
        desktop.locator("#result").screenshot(path=out / "player.png")

        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                  is_mobile=True, has_touch=True)
        open_player(mobile, url)
        mobile.screenshot(path=out / "player-mobile.png")

        dark = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=2,
                                color_scheme="dark")
        open_player(dark, url)
        dark.screenshot(path=out / "dark.png")

        model = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=2)
        model.goto(url)
        model.locator("#modele:not([hidden])").wait_for()
        model.locator("#modele").screenshot(path=out / "model.png")

        browser.close()
    server.shutdown()
    print(f"Captures écrites dans {out}/")


if __name__ == "__main__":
    main()
