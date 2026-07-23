"""Pass 3 — lay the HUD over the crossfaded cut.

Kept separate from render_pipeline.py so the type, the logo and the record
counter survive the transition: cross-fading two clips that each carry their
own HUD double-exposes the digits for the length of the fade.

    python apply_hud.py CUT.mp4 OUT.mp4 --segment 0:IMG_0658 --segment 537:IMG_0678 \
        --records-total 13804
"""

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_pipeline import (FONT_MONO, LINE_X0, LOGO_CX, LINE_Y, ICON_CY_RATIO,
                             LOGO_W, alpha_paste, build_hud_text, load_logo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--logo", required=True)
    ap.add_argument("--segment", action="append", required=True,
                    help="START_FRAME:LABEL — the source shown from that frame on")
    ap.add_argument("--records-total", type=int, required=True)
    args = ap.parse_args()

    segments = []
    for s in args.segment:
        start, label = s.split(":", 1)
        segments.append((int(start), label))
    segments.sort()

    cap = cv2.VideoCapture(args.src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # One prebuilt type layer per source label.
    layers = {}
    for _, label in segments:
        rgba = build_hud_text(w, h, label, f"{w}x{h}  {round(fps)} FPS")
        layers[label] = (rgba[:, :, [2, 1, 0]].astype(np.float32),
                         rgba[:, :, 3:4].astype(np.float32) / 255.0)

    logo_bgr, logo_a = load_logo(args.logo, LOGO_W)
    lh, lw = logo_a.shape[:2]
    logo_x = LOGO_CX - lw // 2
    logo_y = LINE_Y - round(lh * ICON_CY_RATIO)

    mono_num = ImageFont.truetype(FONT_MONO, 34)
    writer = cv2.VideoWriter(args.dst, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        label = segments[0][1]
        for start, lab in segments:
            if idx >= start:
                label = lab
        hud_bgr, hud_a = layers[label]

        out = (frame.astype(np.float32) * (1 - hud_a)
               + hud_bgr * hud_a).astype(np.uint8)
        alpha_paste(out, logo_bgr, logo_a, logo_x, logo_y)

        # Records accrue evenly across the cut — 17 keypoints per frame read.
        records = round(args.records_total * (idx + 1) / total)
        plate = Image.new("RGBA", (420, 54), (0, 0, 0, 0))
        ImageDraw.Draw(plate).text(
            (0, 0), f"{records:,}".replace(",", "."), font=mono_num,
            fill=(232, 230, 227, 255))
        pa = np.array(plate)
        alpha_paste(out, pa[:, :, [2, 1, 0]].astype(np.float32),
                    pa[:, :, 3:4].astype(np.float32) / 255.0, LINE_X0 + 132, 1738)

        writer.write(out)
        idx += 1
        if idx % 200 == 0 or idx == total:
            print(f"  {idx}/{total}", flush=True)

    cap.release()
    writer.release()
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
