"""Uniform boxes, actual Track IDs and compact two-line information cards."""
import math
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
from .geometry import crop_rectangle, intersection

BLUE = "#087DD1"


@lru_cache(maxsize=128)
def font(size, path=""):
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)


def fit_font(text, width, preferred, path=""):
    for size in range(preferred, 9, -1):
        f = font(size, path)
        if f.getlength(text) <= width:
            return f
    raise ValueError("Text cannot fit safely; increase panel width or shorten the displayed field")


def association_panel(frame, width=1600, font_path=""):
    if not isinstance(width, int) or width < 800:
        raise ValueError("Association panel width must be at least 800 pixels")
    k = width / 1600
    photo_h, pad, gap, card_h = (round(v * k) for v in (820, 18, 10, 108))
    ships = frame["ships"]
    if not ships:
        raise ValueError("Cannot render an association panel without accepted records")
    with Image.open(frame["image_path"]) as raw:
        if list(raw.size) != frame["size"]: raise ValueError("Source image dimensions changed")
        rect = crop_rectangle(raw.size, frame["display_crop"])
        photo = raw.convert("RGB").crop(rect)
    ratio = min(width / photo.width, photo_h / photo.height)
    pw, ph = round(photo.width * ratio), round(photo.height * ratio)
    ox, oy = (width - pw) // 2, (photo_h - ph) // 2
    cols = 2 if len(ships) == 4 else min(3, len(ships))
    rows = math.ceil(len(ships) / cols)
    height = photo_h + round(8 * k) + rows * card_h + (rows - 1) * gap + round(12 * k)
    out = Image.new("RGB", (width, height), "white")
    out.paste(photo.resize((pw, ph), Image.Resampling.LANCZOS), (ox, oy))
    draw = ImageDraw.Draw(out)
    boxes, badges, cards = [], [], []
    for ship in ships:
        x1, y1, x2, y2 = ship["bbox"]
        boxes.append([(x1 - rect[0]) * ratio + ox, (y1 - rect[1]) * ratio + oy,
                      (x2 - rect[0]) * ratio + ox, (y2 - rect[1]) * ratio + oy])
    for box in boxes: draw.rectangle(box, outline=BLUE, width=max(1, round(4 * k)))
    cw = (width - 2 * pad - gap * (cols - 1)) / cols
    for i, (ship, box) in enumerate(zip(ships, boxes)):
        label = str(ship["track_id"])
        label_font = font(max(10, round(31 * k)), font_path)
        bw = max(round(46 * k), math.ceil(label_font.getlength(label)) + round(20 * k))
        bh = round(46 * k)
        positions = [(box[0], box[1] - bh - 3), (box[2] - bw, box[1] - bh - 3),
                     (box[0] - bw - 3, box[1]), (box[2] + 3, box[1]), (box[0], box[3] + 3)]
        choices = [[x, y, x + bw, y + bh] for x, y in positions
                   if ox <= x <= ox + pw - bw and oy <= y <= oy + ph - bh]
        if not choices:
            x, y = min(max(ox, box[0]), ox + pw - bw), min(max(oy, box[1]), oy + ph - bh)
            choices = [[x, y, x + bw, y + bh]]
        badge = min(choices, key=lambda q: sum(intersection(q, r) for r in boxes + badges))
        badges.append(badge)
        draw.rectangle(badge, fill=BLUE)
        draw.text(((badge[0] + badge[2]) / 2, (badge[1] + badge[3]) / 2), label,
                  font=label_font, fill="white", anchor="mm")
        x, y = pad + (i % cols) * (cw + gap), photo_h + round(8 * k) + (i // cols) * (card_h + gap)
        draw.rectangle((x, y, x + cw, y + card_h), outline="#111111", width=max(1, round(2 * k)))
        fields = ["MMSI: " + ship["mmsi_masked"], "Track ID: " + label,
                  "Speed: " + ship["speed_text"], "Name: " + ship["name_masked"]]
        usable = cw - 36 * k
        card = []
        for j, text in enumerate(fields):
            row, col = divmod(j, 2)
            left = .60 * usable
            cell_w = left if col == 0 else usable - left
            cell_x = x + 12 * k + (left + 12 * k if col else 0)
            cy = y + (29 + 46 * row) * k
            selected_font = fit_font(text, cell_w - 4 * k, round(32 * k), font_path)
            draw.text((cell_x, cy), text, font=selected_font, fill="#111111", anchor="lm")
            card.append(dict(text=text, x=cell_x, y=cy, width=cell_w, font_size=selected_font.size))
        cards.append(card)
    return out, dict(size=[width, height], boxes=boxes, badges=badges, cards=cards)


def assemble_rows(rows, cell_width=1120, gap=8):
    """Letterbox complete panels; never stretch or crop evidence to fill a slot."""
    if not rows or any(not row for row in rows) or cell_width < 1 or gap < 0:
        raise ValueError("Nonempty rows, positive width and nonnegative gap are required")
    converted = [[im.convert("RGB") for im in row] for row in rows]
    heights = [max(round(im.height * cell_width / im.width) for im in row) for row in converted]
    result = Image.new("RGB", (max(map(len, rows)) * (cell_width + gap) - gap,
                               sum(heights) + gap * (len(rows) - 1)), "white")
    y = 0
    for row, height in zip(converted, heights):
        for j, im in enumerate(row):
            h = round(im.height * cell_width / im.width)
            result.paste(im.resize((cell_width, h), Image.Resampling.LANCZOS), (j * (cell_width + gap), y))
        y += height + gap
    return result
