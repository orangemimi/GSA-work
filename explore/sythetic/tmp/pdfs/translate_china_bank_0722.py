from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color, white, black
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path("/Users/mimi/Documents/Code/Github/GSA-work/explore/sythetic")
BACKGROUND = ROOT / "tmp/pdfs/china_bank_0722_300-1.png"
OUTPUT = ROOT / "output/pdf/Bank_of_China_Transaction_Details_English_2026-07-22.pdf"

W, H = 842.0, 595.0
HEADER_GRAY = Color(195 / 255, 195 / 255, 195 / 255)
SEAL_RED = Color(0.67, 0.04, 0.05)


def rect_from_top(c, x0, top, x1, bottom, color=white):
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.rect(x0, H - bottom, x1 - x0, bottom - top, fill=1, stroke=0)


def draw_center(c, text, x0, top, x1, bottom, font="Times-Roman", size=8, color=black):
    c.setFillColor(color)
    c.setFont(font, size)
    baseline = H - ((top + bottom) / 2) - size * 0.34
    c.drawCentredString((x0 + x1) / 2, baseline, text)


def draw_lines(c, lines, x0, top, x1, bottom, font="Times-Roman", size=6, leading=None, color=black):
    if leading is None:
        leading = size * 1.05
    total = leading * len(lines)
    first_baseline_from_top = ((bottom - top) - total) / 2 + size * 0.84
    c.setFillColor(color)
    c.setFont(font, size)
    for i, line in enumerate(lines):
        y_top = top + first_baseline_from_top + i * leading
        c.drawCentredString((x0 + x1) / 2, H - y_top, line)


def fit_text(c, text, x, top, max_width, max_size=9, min_size=5.5, font="Times-Roman", color=black):
    size = max_size
    while size > min_size and stringWidth(text, font, size) > max_width:
        size -= 0.1
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, H - top - size * 0.82, text)


buf = BytesIO()
c = canvas.Canvas(buf, pagesize=(W, H))
c.drawImage(ImageReader(str(BACKGROUND)), 0, 0, width=W, height=H, mask="auto")

# Document title.
rect_from_top(c, 275, 29, 532, 60)
draw_center(c, "BANK OF CHINA TRANSACTION DETAILS", 275, 29, 532, 60, "Times-Bold", 17)

# Summary fields. Each complete field is redrawn so the values remain aligned.
top_fields = [
    (30, 76.5, 199, 90.5, "Transaction Period: 2025-07-23 to 2026-07-22"),
    (202, 76.5, 300, 90.5, "Customer Name: Zhiyi Zhu"),
    (571, 76.5, 640, 90.5, "Page: 1 / 1"),
    (30, 92.5, 194, 106.5, "Debit Card No.: 6217856100119975453"),
    (202, 92.5, 330, 106.5, "Total Debits: 928.00"),
    (364, 92.5, 500, 106.5, "Total Credits: 4,928.00"),
    (571, 92.5, 630, 106.5, "Rows: 8"),
    (30, 109.5, 190, 123.5, "Account No.: 466379196574"),
    (202, 109.5, 320, 123.5, "Transaction Type: All"),
    (364, 109.5, 470, 123.5, "Currency: EUR"),
    (571, 109.5, 720, 123.5, "Printed at: 2026/07/22 22:56:20"),
]
for x0, top, x1, bottom, text in top_fields:
    rect_from_top(c, x0, top, x1, bottom)
    fit_text(c, text, x0 + 2, top + 1.5, x1 - x0 - 4, max_size=8.5, min_size=6.4,
             font="Times-Bold" if ":" in text else "Times-Roman")

# Table headers. The source gray is retained exactly and the black borders are untouched.
cols = [30, 98, 155, 210, 273, 338, 393, 445, 517, 592, 665, 737, 820]
headers = [
    (["Posting Date"], 6.8),
    (["Posting Time"], 6.8),
    (["Currency"], 6.8),
    (["Amount"], 6.8),
    (["Balance"], 6.8),
    (["Transaction"], 6.4),
    (["Channel"], 6.6),
    (["Outlet Name"], 6.3),
    (["Remarks"], 6.6),
    (["Counterparty", "Name"], 5.8),
    (["Counterparty", "Card/Account", "No."], 4.8),
    (["Counterparty", "Bank"], 5.8),
]
for i, (lines, size) in enumerate(headers):
    x0, x1 = cols[i] + 0.8, cols[i + 1] - 0.8
    rect_from_top(c, x0, 130.2, x1, 145.2, HEADER_GRAY)
    draw_lines(c, lines, x0, 130.2, x1, 145.2, "Times-Bold", size, leading=size * 0.96)

# Data-cell translations. The unchanged dates, times, amounts, balances, account numbers,
# dashes, and pre-existing English remittance details remain from the source page.
row_tops = [146, 164, 182, 200, 218, 236, 254, 272]
row_bottoms = [163.4, 181.4, 199.4, 217.4, 235.4, 253.4, 271.4, 289.4]
transactions = [
    ["International", "Remittance"],
    ["International", "Remittance"],
    ["International", "Remittance"],
    ["Transfer Out"],
    ["Foreign Exchange", "Purchase"],
    ["Foreign Exchange", "Purchase"],
    ["Cash", "Withdrawal"],
    ["Foreign Exchange", "Purchase"],
]
channels = [["Other"], ["Other"], ["Other"], ["Mobile", "Banking"], ["Mobile", "Banking"],
            ["Mobile", "Banking"], ["Counter"], ["Mobile", "Banking"]]

for i, (top, bottom) in enumerate(zip(row_tops, row_bottoms)):
    # Currency
    rect_from_top(c, 155.8, top + 0.7, 209.2, bottom - 0.3)
    draw_center(c, "EUR", 155.8, top, 209.2, bottom, size=6.8)

    # Transaction type
    rect_from_top(c, 338.8, top + 0.7, 392.2, bottom - 0.3)
    draw_lines(c, transactions[i], 338.8, top, 392.2, bottom, size=6.0, leading=6.2)

    # Channel
    rect_from_top(c, 393.8, top + 0.7, 444.2, bottom - 0.3)
    draw_lines(c, channels[i], 393.8, top, 444.2, bottom, size=6.0, leading=6.2)

# Counterparty names on rows that contain the Chinese account-holder name.
for i in [3, 4, 5, 7]:
    top, bottom = row_tops[i], row_bottoms[i]
    rect_from_top(c, 592.8, top + 0.7, 664.2, bottom - 0.3)
    draw_center(c, "Zhiyi Zhu", 592.8, top, 664.2, bottom, size=6.7)

# Counterparty bank on domestic-transfer / exchange rows.
branch_lines = ["Bank of China Nanjing", "Chengdong Sub-branch", "Business Department"]
for i in [3, 4, 5, 7]:
    top, bottom = row_tops[i], row_bottoms[i]
    rect_from_top(c, 737.8, top + 0.7, 819.2, bottom - 0.3)
    draw_lines(c, branch_lines, 737.8, top, 819.2, bottom, size=4.75, leading=4.8)

# Outlet name on the cash-withdrawal row.
i = 6
top, bottom = row_tops[i], row_bottoms[i]
rect_from_top(c, 445.8, top + 0.7, 516.2, bottom - 0.3)
draw_lines(c, branch_lines, 445.8, top, 516.2, bottom, size=4.55, leading=4.7)

# Footer note and page number.
rect_from_top(c, 18, 550.5, 590, 566.5)
fit_text(
    c,
    "Note: 1. The posting date/time is the date/time when the system processes the transaction and may differ from the actual transaction submission time.",
    20,
    553,
    560,
    max_size=8.0,
    min_size=6.0,
)
rect_from_top(c, 352, 574.0, 445, 590.0)
draw_center(c, "Page 1 of 1", 352, 574.0, 445, 590.0, size=8.5)

# Preserve the original official seal image and provide its formal English translation.
# This avoids redrawing or falsifying the seal while translating its contents for review.
rect_from_top(c, 706, 104.5, 838, 127.5)
draw_lines(
    c,
    ["[Seal: BANK OF CHINA LIMITED]", "Mobile Banking Channel", "Business-Specific Seal"],
    706,
    104.5,
    838,
    127.5,
    font="Helvetica-Bold",
    size=4.8,
    leading=5.5,
    color=SEAL_RED,
)

c.showPage()
c.save()
buf.seek(0)

reader = PdfReader(buf)
writer = PdfWriter()
writer.add_page(reader.pages[0])
writer.add_metadata({
    "/Title": "Bank of China Transaction Details - English Translation",
    "/Subject": "English translation of Bank of China transaction details dated 2026-07-22",
})
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("wb") as f:
    writer.write(f)

print(OUTPUT)
