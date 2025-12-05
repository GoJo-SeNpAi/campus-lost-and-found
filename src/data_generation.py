"""
Create a small synthetic dataset for Lost & Found.
Generates simple colored object images and text descriptions.
"""
import os
import csv
from PIL import Image, ImageDraw, ImageFont
import random

OUT_DIR = "data"
IMG_DIR = os.path.join(OUT_DIR, "images")
CSV_PATH = os.path.join(OUT_DIR, "dataset.csv")
os.makedirs(IMG_DIR, exist_ok=True)

COLORS = {
    "red": (220, 20, 60),
    "blue": (30, 144, 255),
    "green": (34,139,34),
    "black": (10,10,10),
    "white": (245,245,245),
    "yellow": (255,215,0),
    "gray": (120,120,120)
}

OBJECTS = ["water bottle", "wallet", "keys", "school bag", "book", "earphones", "umbrella", "mug"]

def draw_object_image(path, color_rgb, label_text=None, size=(224,224), dent=False):
    img = Image.new("RGB", size, (240,240,240))
    draw = ImageDraw.Draw(img)
    # draw colored rounded rectangle as object
    margin = 30
    bbox = [margin, margin, size[0]-margin, size[1]-margin]
    draw.rectangle(bbox, fill=color_rgb)
    if dent:
        # draw small black dent circle
        cx = random.randint(margin+20, size[0]-margin-20)
        cy = random.randint(margin+20, size[1]-margin-20)
        r = random.randint(6,14)
        draw.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(0,0,0))
    if label_text:
        try:
            f = ImageFont.truetype("arial.ttf", 12)
        except:
            f = None
        draw.text((10, size[1]-20), label_text, fill=(0,0,0), font=f)
    img.save(path)

def random_description(color, obj, extras):
    base = f"{color} {obj}"
    if extras:
        base += " " + ", ".join(extras)
    return base

def main():
    rows = []
    id_ctr = 1
    # create 30 entries
    for i in range(30):
        kind = random.choice(["lost", "found"])
        color = random.choice(list(COLORS.keys()))
        obj = random.choice(OBJECTS)
        dent = random.random() < 0.25
        extras = []
        if random.random() < 0.3:
            extras.append(random.choice(["with a dent", "has sticker", "with initials", "damaged", "chain attached"]))
        if random.random() < 0.25:
            extras.append(random.choice(["small scratch", "zip missing", "waterproof"]))
        desc = random_description(color, obj, extras)
        fname = f"{kind}_{id_ctr:03d}.jpg"
        path = os.path.join(IMG_DIR, fname)
        draw_object_image(path, COLORS[color], label_text=fname, dent=dent)
        rows.append({
            "id": id_ctr,
            "kind": kind,
            "description": desc,
            "image_path": path,
            "object": obj,
            "color": color
        })
        id_ctr += 1

    # write CSV
    with open(CSV_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id","kind","description","image_path","object","color"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Dataset created: {CSV_PATH} with images in {IMG_DIR}")

if __name__ == "__main__":
    main()
