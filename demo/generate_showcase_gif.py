"""
Generate high-resolution animated showcase GIF (demo/showcase_clip.gif) replaying 
demo_02_hallucination_prone.json token-by-token with real-time PSC gauge animation.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def render_frame(prompt_text, tokens_so_far, current_token, psc_score, status_text, is_halt, width=800, height=450):
    """Renders a single frame of the SynapseGuard live interception console."""
    # Dark base background (#05070B)
    img = Image.new("RGB", (width, height), color=(5, 7, 11))
    draw = ImageDraw.Draw(img)

    try:
        font_mono = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 14)
        font_mono_bold = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 16)
        font_title = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 11)
    except Exception:
        font_mono = font_mono_bold = font_title = font_small = ImageFont.load_default()

    # Outer Hardware Bezel (#151C2C)
    draw.rectangle([20, 20, width - 20, height - 20], outline=(30, 41, 59), fill=(10, 14, 23), width=2)
    draw.rectangle([25, 25, width - 25, height - 25], outline=(6, 182, 212), fill=(7, 10, 16), width=1)

    # Title Bar
    draw.text((40, 40), "SYNAPSEGUARD DIAGNOSTIC INTERCEPTION CONSOLE", font=font_title, fill=(6, 182, 212))
    draw.text((40, 65), "Real-Time SAE Activation & PSC Coherence Monitor", font=font_small, fill=(100, 116, 139))

    # Prompt Box
    draw.rectangle([40, 95, width - 40, 135], outline=(30, 41, 59), fill=(15, 23, 42))
    draw.text((50, 107), "PROMPT:", font=font_mono_bold, fill=(245, 158, 11))
    draw.text((125, 107), prompt_text, font=font_mono, fill=(226, 232, 240))

    # Streamed Token Output Box (Left 440px)
    draw.rectangle([40, 155, 480, 340], outline=(30, 41, 59), fill=(5, 7, 11))
    draw.text((50, 165), "STREAMED TOKEN OUTPUT", font=font_small, fill=(148, 163, 184))

    # Draw tokens so far
    x_offset, y_offset = 50, 195
    for t in tokens_so_far:
        text = repr(t.get("token_text", t.get("token", "")))[1:-1]
        tok_psc = t.get("psc_score", t.get("psc", 0.0))
        if tok_psc >= 0.85:
            bg_col = (153, 27, 27) # Red
            txt_col = (254, 202, 202)
        elif tok_psc >= 0.65:
            bg_col = (146, 64, 14) # Amber
            txt_col = (254, 243, 199)
        else:
            bg_col = (30, 41, 59) # Slate
            txt_col = (110, 231, 183)

        bbox = font_mono.getbbox(text)
        w_tok = max(16, bbox[2] - bbox[0] + 12)
        h_tok = 24
        if x_offset + w_tok > 460:
            x_offset = 50
            y_offset += 32

        draw.rectangle([x_offset, y_offset, x_offset + w_tok, y_offset + h_tok], fill=bg_col, outline=(51, 65, 85))
        draw.text((x_offset + 6, y_offset + 4), text, font=font_mono, fill=txt_col)
        x_offset += w_tok + 8

    # Right Column: PSC Gauge & Interception Banner (Right 240px)
    draw.rectangle([500, 155, width - 40, 340], outline=(30, 41, 59), fill=(5, 7, 11))
    draw.text((515, 165), "PSC COHERENCE SCORE", font=font_small, fill=(148, 163, 184))
    
    # Score value
    score_str = f"{psc_score:.2f}"
    score_col = (239, 68, 68) if is_halt else ((245, 158, 11) if psc_score >= 0.65 else (16, 185, 129))
    draw.text((680, 162), score_str, font=font_mono_bold, fill=score_col)

    # Track
    track_left, track_right = 515, width - 55
    track_y = 210
    draw.rectangle([track_left, track_y, track_right, track_y + 14], fill=(15, 23, 42), outline=(30, 41, 59))
    
    # Markers
    warn_x = track_left + int(0.65 * (track_right - track_left))
    halt_x = track_left + int(0.85 * (track_right - track_left))
    draw.line([warn_x, track_y, warn_x, track_y + 14], fill=(245, 158, 11), width=2)
    draw.line([halt_x, track_y, halt_x, track_y + 14], fill=(239, 68, 68), width=2)

    # Fill
    fill_w = int(min(1.0, max(0.0, psc_score)) * (track_right - track_left))
    if fill_w > 0:
        draw.rectangle([track_left, track_y + 1, track_left + fill_w, track_y + 13], fill=score_col)

    draw.text((track_left, track_y + 20), "0.00", font=font_small, fill=(100, 116, 139))
    draw.text((warn_x - 15, track_y + 20), "0.65 Warn", font=font_small, fill=(245, 158, 11))
    draw.text((halt_x - 15, track_y + 20), "0.85 Halt", font=font_small, fill=(239, 68, 68))

    # Banner Box
    banner_bg = (127, 29, 29) if is_halt else ((120, 53, 15) if psc_score >= 0.65 else (6, 78, 59))
    banner_border = (239, 68, 68) if is_halt else ((245, 158, 11) if psc_score >= 0.65 else (16, 185, 129))
    draw.rectangle([515, 275, width - 55, 325], fill=banner_bg, outline=banner_border, width=2)
    
    banner_msg = "⚠️ HALT: GENERATION INTERCEPTED" if is_halt else ("WARNING: UNCERTAINTY DRIFT" if psc_score >= 0.65 else "STATUS: SAFE")
    draw.text((525, 295), banner_msg, font=font_mono_bold, fill=(255, 255, 255))

    # Footer
    draw.text((40, 395), "SynapseGuard Real-Time Early-Warning Layer", font=font_small, fill=(100, 116, 139))
    draw.text((width - 240, 395), "Replaying demo_02 trace", font=font_small, fill=(6, 182, 212))

    return img

def main():
    trace_path = Path("data/results/demo_traces/demo_02_hallucination_prone.json")
    if not trace_path.exists():
        print(f"Error: {trace_path} not found.")
        return

    with open(trace_path, "r") as f:
        data = json.load(f)

    prompt = data.get("prompt", "Who was the first president of Mars in 1984?")
    tokens = data.get("tokens", [])

    frames = []
    # Initial empty frame
    frames.append(render_frame(prompt, [], None, 0.0, "SAFE", False))
    
    # Step-by-step frames
    for i in range(len(tokens)):
        toks_so_far = tokens[:i+1]
        curr = tokens[i]
        psc = curr.get("psc_score", curr.get("psc", 0.0))
        is_halt = (psc >= 0.85) or (curr.get("status") == "HALT")
        
        frame = render_frame(prompt, toks_so_far, curr, psc, curr.get("status", "SAFE"), is_halt)
        frames.append(frame)
        if is_halt:
            # Add extra pause frames for halt
            for _ in range(4):
                frames.append(frame)
            break

    # Save animated GIF
    out_dir = Path("demo")
    out_dir.mkdir(exist_ok=True)
    out_gif = out_dir / "showcase_clip.gif"
    
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=600,
        loop=0
    )
    print(f"Successfully generated showcase GIF: {out_gif}")

if __name__ == "__main__":
    main()
