from pathlib import Path
from PIL import Image, ImageDraw

preview_dir = Path(__file__).parent / "previews_no_two_band"
paths = sorted(preview_dir.glob("*.png"))
thumb_w, thumb_h = 520, 300
cols = 3
rows = (len(paths) + cols - 1) // cols
canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 30)), "white")
draw = ImageDraw.Draw(canvas)

for index, path in enumerate(paths):
    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb_w - 12, thumb_h - 12))
    x = (index % cols) * thumb_w + (thumb_w - image.width) // 2
    y = (index // cols) * (thumb_h + 30) + 24
    canvas.paste(image, (x, y))
    draw.text(((index % cols) * thumb_w + 8, (index // cols) * (thumb_h + 30) + 5), path.stem, fill="#111827")

canvas.save(Path(__file__).parent / "previews_no_two_band_contact_sheet.png", quality=92)
