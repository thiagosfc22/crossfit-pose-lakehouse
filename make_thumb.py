"""Build the LinkedIn cover frame for the crossfit pipeline video.

Composites a single frame — skeleton drawn fresh from the cached keypoints so
the cover isn't carrying the video's HUD — under a title and the Databricks
mark.

    python make_thumb.py CLIP.mp4 POSE.npz FRAME OUT.png --logo LOGO.png
"""

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_pipeline import BONES, HEAD, CYAN, RED, KP_CONF, FONT_DISPLAY, FONT_MONO

INK = np.array([20, 14, 11], np.float32)   # BGR #0B0E14
PAPER = (232, 230, 227)
STEEL = (150, 162, 178)


def vertical_ramp(h, w, y0, y1, peak):
    """Smoothstep alpha from y0 to y1, held at peak past y1."""
    ys = np.arange(h, dtype=np.float32)
    t = np.clip((ys - y0) / (y1 - y0), 0, 1)
    return (t * t * (3 - 2 * t) * peak)[:, None, None] * np.ones((1, w, 1), np.float32)


def tracked(draw, xy, text, font, fill, extra=4.0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + extra
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("pose")
    ap.add_argument("frame", type=int)
    ap.add_argument("dst")
    ap.add_argument("--logo", required=True)
    args = ap.parse_args()

    data = np.load(args.pose)
    kps, confs = data["xy"][args.frame], data["conf"][args.frame]
    valid = confs > KP_CONF

    cap = cv2.VideoCapture(args.clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame}")
    h, w = frame.shape[:2]

    # --- skeleton, same treatment as the video ------------------------------
    cyan_layer = np.zeros_like(frame)
    red_layer = np.zeros_like(frame)
    for a, b in BONES:
        if valid[a] and valid[b]:
            cv2.line(cyan_layer, tuple(np.int32(kps[a])), tuple(np.int32(kps[b])),
                     CYAN, 3, cv2.LINE_AA)
    for a, b in HEAD:
        if valid[a] and valid[b]:
            cv2.line(cyan_layer, tuple(np.int32(kps[a])), tuple(np.int32(kps[b])),
                     CYAN, 2, cv2.LINE_AA)
    for i in range(17):
        if valid[i]:
            p = (int(kps[i][0]), int(kps[i][1]))
            cv2.circle(red_layer, p, 9, RED, -1, cv2.LINE_AA)
            cv2.circle(red_layer, p, 4, (255, 255, 255), -1, cv2.LINE_AA)

    halo = cv2.addWeighted(cv2.GaussianBlur(cyan_layer, (31, 31), 0), 0.9,
                           cv2.GaussianBlur(red_layer, (17, 17), 0), 0.6, 0)
    out = cv2.addWeighted(frame, 1.0, halo, 1.0, 0)
    out = cv2.addWeighted(out, 1.0, cv2.add(cyan_layer, red_layer), 0.9, 0)

    # --- ground the type: darken top and base -------------------------------
    out = out.astype(np.float32)
    top_ramp = np.clip((470 - np.arange(h, dtype=np.float32)) / 470, 0, 1)
    top_ramp = (top_ramp ** 1.25 * 0.86)[:, None, None] * np.ones((1, w, 1), np.float32)
    base_ramp = vertical_ramp(h, w, 1150, 1540, 0.94)
    out = out * (1 - top_ramp) + INK * top_ramp
    out = out * (1 - base_ramp) + INK * base_ramp
    img = Image.fromarray(out.astype(np.uint8)[:, :, ::-1])

    # --- type ---------------------------------------------------------------
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(FONT_DISPLAY, 132)
    eyebrow = ImageFont.truetype(FONT_DISPLAY, 38)
    mono = ImageFont.truetype(FONT_MONO, 26)

    tracked(d, (86, 150), "COMPUTER VISION", eyebrow, PAPER, 5.0)
    tracked(d, (86, 1512), "FROM BARBELL", title, PAPER, 1.0)
    tracked(d, (86, 1636), "TO LAKEHOUSE", title, PAPER, 1.0)

    d.line([(88, 1806), (994, 1806)], fill=(58, 66, 78), width=2)
    d.text((88, 1832), "POSE ESTIMATION   17 KEYPOINTS / FRAME   13.804 ROWS",
           font=mono, fill=STEEL)

    # --- mark ---------------------------------------------------------------
    logo = Image.open(args.logo).convert("RGBA")
    lw = 250
    logo = logo.resize((lw, round(logo.height * lw / logo.width)), Image.LANCZOS)
    img.paste(logo, (w - lw - 80, 112), logo)

    img.save(args.dst)
    img.convert("RGB").save(args.dst.rsplit(".", 1)[0] + ".jpg", quality=93)
    print(f"wrote {args.dst} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
