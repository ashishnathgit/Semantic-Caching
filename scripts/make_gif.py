"""Turn the frames from scripts/screenshots.mjs into an animated GIF.

    python scripts/make_gif.py docs/screenshots/frames docs/screenshots/six-stages.gif

Frames are scaled to 960 px wide; each is shown for --ms milliseconds (about the time
between captures). Needs Pillow (installed with matplotlib).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--ms", type=int, default=300)
    args = parser.parse_args()
    files = sorted(args.frames.glob("*.jpg")) + sorted(args.frames.glob("*.png"))
    if not files:
        raise SystemExit(f"no frames in {args.frames}")
    frames = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
        frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    durations = [args.ms] * len(frames)
    durations[-1] = 2500                      # hold the last frame
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{args.out}: {len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
