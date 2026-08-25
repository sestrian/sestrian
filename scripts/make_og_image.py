#!/usr/bin/env python3
"""Generate the Open Graph / Twitter card image for sestrian.com.

Committed as a PNG rather than generated at deploy time on purpose: link
unfurlers (Slack, X, iMessage, Discord) fetch it constantly and cache it hard,
so it has to be a stable URL serving stable bytes. Regenerating it on every
deploy would churn that cache for no reason.

1200x630 is the size every unfurler crops to. Anything important stays well
inside the middle, because several of them centre-crop to 1.91:1 or square.

    uv run --with pillow python scripts/make_og_image.py
"""

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BLACK = (14, 13, 11)          # --black
BONE = (236, 235, 228)        # --bone
DIM = (180, 177, 166)         # --dim
SIGNAL = (255, 77, 0)         # --signal

FONT_DIR = "/System/Library/Fonts/Supplemental"
BLACK_FACE = f"{FONT_DIR}/Arial Black.ttf"
BOLD_FACE = f"{FONT_DIR}/Arial Bold.ttf"
REG_FACE = f"{FONT_DIR}/Arial.ttf"
OUT = "site/og.png"


def main() -> None:
    img = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(img)

    # A single orange rule down the left edge: the site's one accent, used once.
    d.rectangle([0, 0, 10, H], fill=SIGNAL)

    wordmark = ImageFont.truetype(BLACK_FACE, 132)
    tagline = ImageFont.truetype(BOLD_FACE, 44)
    small = ImageFont.truetype(REG_FACE, 25)

    x = 78
    d.text((x, 132), "SESTRIAN", font=wordmark, fill=BONE)

    # The claim, not the mechanism — an unfurl is read in half a second.
    d.text((x, 300), "The AI everyone builds", font=tagline, fill=BONE)
    d.text((x, 364), "and no one owns.", font=tagline, fill=SIGNAL)

    d.text((x, 462),
           "A blockchain whose state is the weights of one public model.",
           font=small, fill=DIM)
    d.text((x, 500),
           "Trained on the world's spare GPUs. It pays the people who train it.",
           font=small, fill=DIM)

    d.line([(x, 556), (x + 190, 556)], fill=SIGNAL, width=3)
    d.text((x, 572), "sestrian.com", font=small, fill=BONE)

    img.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
