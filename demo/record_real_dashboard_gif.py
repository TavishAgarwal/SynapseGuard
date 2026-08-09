"""
Record pixel-perfect animated GIF of the actual running web app (http://localhost:5174/)
using Playwright browser automation.
"""

import time
import os
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

def record_dashboard_gif(url: str = "http://localhost:5174/", out_gif: str = "demo/showcase_clip.gif"):
    print(f"Launching Playwright to capture live browser simulation from {url}...")
    
    frames = []
    tmp_dir = Path("/tmp/synapse_frames")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=2)
        page = context.new_page()
        
        page.goto(url)
        page.wait_for_load_state("networkidle")
        time.sleep(1)
        
        # Scroll to #demo section
        demo_el = page.locator("#demo")
        demo_el.scroll_into_view_if_needed()
        time.sleep(0.5)
        
        # Capture initial static frame
        img_bytes = demo_el.screenshot()
        tmp_frame_path = tmp_dir / "frame_00.png"
        with open(tmp_frame_path, "wb") as f:
            f.write(img_bytes)
        frames.append(Image.open(tmp_frame_path).convert("RGB"))
        
        # Click #inline-btn-play to start real-time token stream
        play_btn = page.locator("#inline-btn-play")
        play_btn.click()
        
        # Capture live frames as tokens stream
        for idx in range(1, 15):
            time.sleep(0.35)
            tmp_frame_path = tmp_dir / f"frame_{idx:02d}.png"
            img_bytes = demo_el.screenshot()
            with open(tmp_frame_path, "wb") as f:
                f.write(img_bytes)
            img = Image.open(tmp_frame_path).convert("RGB")
            frames.append(img)
            
            # Check if status banner reached HALT
            banner_text = page.locator("#inline-status-banner").inner_text()
            if "HALT" in banner_text:
                print(f"Captured HALT interception state at frame {idx}!")
                # Add extra pause frames for HALT
                for _ in range(4):
                    frames.append(img)
                break
                
        browser.close()
        
    if len(frames) > 0:
        out_path = Path(out_gif)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
        print(f"Successfully recorded real browser GIF to {out_path} ({len(frames)} frames)")

if __name__ == "__main__":
    record_dashboard_gif()
