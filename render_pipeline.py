"""Pass 2 — draw the neon skeleton and the ingestion pipeline overlay.

Reads the cached keypoints from extract_pose.py and composites, per frame:
  - a thin cyan skeleton over the footage
  - Lakehouse-red joints, the accent of the piece
  - one particle per emission, falling from a joint onto a conveyor line and
    running right into the Databricks mark, which pulses as it absorbs them

    python render_pipeline.py IN.mp4 POSE.npz OUT.mp4 \
        --source-label "IMG_0658" --records-start 0
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- palette ---
RED = (33, 54, 255)        # BGR — #FF3621 Databricks Lakehouse red
CYAN = (255, 229, 0)       # BGR — #00E5FF
INK = (20, 14, 11)         # BGR — #0B0E14
PAPER = (227, 230, 232)    # BGR — #E8E6E3
STEEL = (138, 122, 110)    # BGR — #6E7A8A

def _font(env_var, candidates, label):
    """First font that exists, overridable by env var so this runs off-macOS."""
    override = os.environ.get(env_var)
    if override:
        return override
    for path in candidates:
        if Path(path).exists():
            return path
    raise SystemExit(
        f"no {label} font found. Set {env_var} to a .ttf, or install one of:\n  "
        + "\n  ".join(candidates))


FONT_DISPLAY = _font("POSE_FONT_DISPLAY", [
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
], "condensed display")

FONT_MONO = _font("POSE_FONT_MONO", [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
], "monospace")

# COCO-17 skeleton, head pairs kept separate so we can draw them lighter.
BONES = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
HEAD = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]
KP_CONF = 0.5

# --------------------------------------------------------------- geometry ---
LINE_Y = 1714              # the conveyor
LINE_X0 = 90
LOGO_W = 220
LOGO_CX = 900
ICON_CY_RATIO = 0.30       # icon sits ~30% down the stacked logo art


class Particle:
    """A keypoint travelling from the body to the lakehouse."""

    __slots__ = ("x0", "y0", "xr", "age", "life_a", "life_b", "trail", "done")

    def __init__(self, x0, y0, xr, life_a, life_b):
        self.x0, self.y0, self.xr = x0, y0, xr
        self.age = 0.0
        self.life_a, self.life_b = life_a, life_b
        self.trail = []
        self.done = False

    def step(self, dt, x_dest):
        self.age += dt
        if self.age < self.life_a:
            # Quadratic Bezier: drop vertically, then bend onto the line.
            t = self.age / self.life_a
            u = 1 - t
            px = u * u * self.x0 + 2 * u * t * self.x0 + t * t * self.xr
            py = u * u * self.y0 + 2 * u * t * LINE_Y + t * t * LINE_Y
        else:
            t = (self.age - self.life_a) / self.life_b
            if t >= 1.0:
                self.done = True
                return None
            px = self.xr + (x_dest - self.xr) * t
            py = LINE_Y
        pos = (px, py)
        self.trail.append(pos)
        if len(self.trail) > 4:
            self.trail.pop(0)
        return pos


def load_logo(path, width):
    """Load the Databricks mark as (bgr, alpha) at the requested width."""
    art = Image.open(path).convert("RGBA")
    h = round(art.height * width / art.width)
    art = art.resize((width, h), Image.LANCZOS)
    rgba = np.array(art)
    bgr = rgba[:, :, [2, 1, 0]].astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    return bgr, alpha


def alpha_paste(canvas, bgr, alpha, x, y):
    """Composite an RGBA-style layer onto canvas at (x, y), clipped to bounds."""
    h, w = alpha.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sub_a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    sub_c = bgr[y0 - y:y1 - y, x0 - x:x1 - x]
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (region * (1 - sub_a) + sub_c * sub_a).astype(np.uint8)


def base_gradient(h, w):
    """Alpha ramp that sinks the footage under the HUD band.

    It has to reach near-solid *above* the first line of type, or the gym
    floor shows through and the labels stop being readable.
    """
    ramp = np.zeros((h, w, 1), np.float32)
    fade_top, fade_end, peak = 1440, 1596, 0.93
    ys = np.arange(h, dtype=np.float32)
    t = np.clip((ys - fade_top) / (fade_end - fade_top), 0, 1)
    ramp[:, :, 0] = (t * t * (3 - 2 * t) * peak)[:, None]  # smoothstep
    return ramp


def build_hud_text(w, h, source_label, dims_label):
    """Static type for the HUD, rendered once with PIL and reused per frame."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    display = ImageFont.truetype(FONT_DISPLAY, 40)
    mono = ImageFont.truetype(FONT_MONO, 21)
    small = ImageFont.truetype(FONT_DISPLAY, 27)

    # Tracking isn't a PIL feature; space the display line by hand.
    def tracked(draw, xy, text, font, fill, extra=3.0):
        x, y = xy
        for ch in text:
            draw.text((x, y), ch, font=font, fill=fill)
            x += draw.textlength(ch, font=font) + extra

    tracked(d, (LINE_X0, 1612), "POSE ESTIMATION", display, (232, 230, 227, 255))
    tracked(d, (LINE_X0 + 322, 1612), "· 17 KEYPOINTS / FRAME", display,
            (168, 179, 194, 255))
    d.text((LINE_X0 + 2, 1666), f"SOURCE  {source_label}   {dims_label}",
           font=mono, fill=(140, 152, 168, 255))
    tracked(d, (LINE_X0 + 2, 1745), "INGESTED", small, (140, 152, 168, 255), 2.0)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("pose")
    ap.add_argument("dst")
    ap.add_argument("--logo", required=True,
                    help="PNG with alpha for the mark shown at the end of the "
                         "pipeline (see README)")
    ap.add_argument("--source-label", default="CLIP")
    ap.add_argument("--records-start", type=int, default=0)
    ap.add_argument("--emit-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-hud", action="store_true",
                    help="skip type, logo and counter so apply_hud.py can lay "
                         "them over the crossfaded cut instead")
    args = ap.parse_args()

    data = np.load(args.pose)
    XY, CONF = data["xy"], data["conf"]

    cap = cv2.VideoCapture(args.src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps

    logo_bgr, logo_a = load_logo(args.logo, LOGO_W)
    lh, lw = logo_a.shape[:2]
    icon_cy = round(lh * ICON_CY_RATIO)
    logo_x = LOGO_CX - lw // 2
    logo_y = LINE_Y - icon_cy
    x_dest = float(LOGO_CX)
    line_x1 = logo_x - 26

    ramp = base_gradient(h, w)
    ink_plate = np.full((h, w, 3), INK, np.float32)
    hud_rgba = build_hud_text(w, h, args.source_label, f"{w}x{h}  {round(fps)} FPS")
    hud_bgr = hud_rgba[:, :, [2, 1, 0]].astype(np.float32)
    hud_a = hud_rgba[:, :, 3:4].astype(np.float32) / 255.0

    mono_num = ImageFont.truetype(FONT_MONO, 34)

    rng = random.Random(args.seed)
    particles = []
    pulse = 0.0
    records = args.records_start

    writer = cv2.VideoWriter(args.dst, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= len(XY):
            break
        kps, confs = XY[idx], CONF[idx]
        valid = confs > KP_CONF
        idx += 1

        cyan_layer = np.zeros_like(frame)
        red_layer = np.zeros_like(frame)

        # --- skeleton: thin bones, red joints -------------------------------
        for a, b in BONES:
            if valid[a] and valid[b]:
                cv2.line(cyan_layer, tuple(np.int32(kps[a])),
                         tuple(np.int32(kps[b])), CYAN, 2, cv2.LINE_AA)
        for a, b in HEAD:
            if valid[a] and valid[b]:
                cv2.line(cyan_layer, tuple(np.int32(kps[a])),
                         tuple(np.int32(kps[b])), CYAN, 1, cv2.LINE_AA)
        for i in range(17):
            if not valid[i]:
                continue
            p = (int(kps[i][0]), int(kps[i][1]))
            cv2.circle(red_layer, p, 7, RED, -1, cv2.LINE_AA)
            cv2.circle(red_layer, p, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # --- emit one record from a joint ------------------------------------
        if valid.any():
            records += 17
            if idx % args.emit_every == 0:
                pool = np.flatnonzero(valid)
                j = int(rng.choice(pool.tolist()))
                # Slower fall keeps the per-frame step short, so the comet
                # tail stays a spark instead of streaking across the floor.
                particles.append(Particle(
                    float(kps[j][0]), float(kps[j][1]),
                    float(kps[j][0]) + rng.uniform(20, 90),
                    rng.uniform(0.85, 1.15), rng.uniform(0.55, 0.78)))

        # --- advance particles ------------------------------------------------
        # These live on their own layer: they travel down into the HUD band,
        # so they have to be composited *after* it or the band swallows them.
        part_layer = np.zeros_like(frame)
        alive = []
        for p in particles:
            pos = p.step(dt, x_dest)
            if p.done:
                pulse = 1.0
                continue
            alive.append(p)
            for k in range(1, len(p.trail)):
                a0 = (k / len(p.trail)) ** 2 * 0.55
                x1p, y1p = p.trail[k - 1]
                x2p, y2p = p.trail[k]
                col = tuple(int(c * a0) for c in RED)
                cv2.line(part_layer, (int(x1p), int(y1p)), (int(x2p), int(y2p)),
                         col, 1, cv2.LINE_AA)
            cv2.circle(part_layer, (int(pos[0]), int(pos[1])), 4, RED, -1,
                       cv2.LINE_AA)
            cv2.circle(part_layer, (int(pos[0]), int(pos[1])), 1,
                       (200, 220, 255), -1, cv2.LINE_AA)
        particles = alive

        # --- composite neon ---------------------------------------------------
        # Cyan gets a wide soft halo; red is kept tighter so the accent reads
        # as sharp points rather than a red haze over the whole frame.
        halo = cv2.addWeighted(cv2.GaussianBlur(cyan_layer, (25, 25), 0), 0.85,
                               cv2.GaussianBlur(red_layer, (13, 13), 0), 0.55, 0)
        glow = cv2.add(cyan_layer, red_layer)
        out = cv2.addWeighted(frame, 1.0, halo, 1.0, 0)
        out = cv2.addWeighted(out, 1.0, glow, 0.9, 0)

        # --- HUD band ---------------------------------------------------------
        out = (out.astype(np.float32) * (1 - ramp) + ink_plate * ramp).astype(np.uint8)

        # Conveyor, then the records running along it — both on top of the band.
        cv2.line(out, (LINE_X0, LINE_Y), (line_x1, LINE_Y), (78, 66, 58), 2,
                 cv2.LINE_AA)
        p_halo = cv2.GaussianBlur(part_layer, (17, 17), 0)
        out = cv2.addWeighted(out, 1.0, p_halo, 0.8, 0)
        out = cv2.addWeighted(out, 1.0, part_layer, 0.95, 0)

        if pulse > 0.01:
            halo_r = np.zeros_like(out)
            cv2.circle(halo_r, (LOGO_CX, LINE_Y), 34,
                       tuple(int(c * pulse) for c in RED), -1, cv2.LINE_AA)
            halo_r = cv2.GaussianBlur(halo_r, (45, 45), 0)
            out = cv2.addWeighted(out, 1.0, halo_r, 0.5, 0)
            pulse *= 0.84

        if not args.no_hud:
            out = (out.astype(np.float32) * (1 - hud_a)
                   + hud_bgr * hud_a).astype(np.uint8)
            alpha_paste(out, logo_bgr, logo_a, logo_x, logo_y)

            # Live record counter, tabular so digits don't jitter.
            plate = Image.new("RGBA", (420, 54), (0, 0, 0, 0))
            ImageDraw.Draw(plate).text(
                (0, 0), f"{records:,}".replace(",", "."), font=mono_num,
                fill=(232, 230, 227, 255))
            pa = np.array(plate)
            alpha_paste(out, pa[:, :, [2, 1, 0]].astype(np.float32),
                        pa[:, :, 3:4].astype(np.float32) / 255.0,
                        LINE_X0 + 132, 1738)

        writer.write(out)
        if idx % 100 == 0 or idx == total:
            print(f"  {idx}/{total}", flush=True)

    cap.release()
    writer.release()
    print(f"wrote {args.dst}  (records now {records})")


if __name__ == "__main__":
    main()
