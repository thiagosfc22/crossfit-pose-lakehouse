"""Pass 1 — run YOLO pose once per video and cache the keypoints.

Rendering iterates on the visual treatment; inference doesn't need to re-run
every time, so it lives here and writes an .npz the renderer reads.

    python extract_pose.py IN.mp4 OUT.npz [--weights yolov8x-pose.pt]
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--weights", default="yolov8x-pose.pt")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.src}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(args.weights)
    all_xy, all_conf = [], []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        res = model.track(frame, persist=True, verbose=False,
                          device=args.device, classes=[0], conf=0.35)[0]

        xy = np.zeros((17, 2), np.float32)
        conf = np.zeros(17, np.float32)

        boxes, kp = res.boxes, res.keypoints
        if boxes is not None and len(boxes) and kp is not None and kp.xy is not None:
            # The athlete is the largest person in frame.
            b = boxes.xyxy.cpu().numpy()
            pick = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            xy = kp.xy.cpu().numpy()[pick].astype(np.float32)
            if kp.conf is not None:
                conf = kp.conf.cpu().numpy()[pick].astype(np.float32)
            else:
                conf = np.ones(17, np.float32)

        all_xy.append(xy)
        all_conf.append(conf)
        if idx % 60 == 0 or idx == total:
            print(f"  {idx}/{total}", flush=True)

    cap.release()
    np.savez_compressed(args.dst, xy=np.stack(all_xy), conf=np.stack(all_conf),
                        fps=fps, width=w, height=h)
    got = int((np.stack(all_conf).max(axis=1) > 0).sum())
    print(f"wrote {args.dst}  ({got}/{idx} frames with a pose)")


if __name__ == "__main__":
    main()
