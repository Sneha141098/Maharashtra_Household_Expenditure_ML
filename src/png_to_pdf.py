from pathlib import Path
from reportlab.platypus import (
    SimpleDocTemplate, Image, PageBreak
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch
from PIL import Image as PILImage

# Project folders
input_folder = Path("outputs/graphs/advanced")
output_folder = Path("outputs")

output_folder.mkdir(exist_ok=True)

# Find all PNG images
png_files = sorted(input_folder.glob("*.png"))

if not png_files:
    print("No PNG images found in:", input_folder)
    exit()

# Create PDF
pdf_path = output_folder / "Advanced_Visualizations_Report.pdf"

pdf = SimpleDocTemplate(
    str(pdf_path),
    pagesize=landscape(A4),
    rightMargin=30,
    leftMargin=30,
    topMargin=30,
    bottomMargin=30
)

page_width, page_height = landscape(A4)

available_width = page_width - 60
available_height = page_height - 60

elements = []

for index, png in enumerate(png_files):

    # Read image dimensions
    with PILImage.open(png) as img:
        width, height = img.size

    # Calculate scaling factor with extra safety margin
    scale = min(
        available_width / width,
        available_height / height
    ) * 0.93

    # Resize image proportionally
    graph = Image(
        str(png),
        width=width * scale,
        height=height * scale
    )

    elements.append(graph)

    # Add page break after each image
    if index < len(png_files) - 1:
        elements.append(PageBreak())

# Generate PDF
pdf.build(elements)

print("\nPDF created successfully!")
print("Location:", pdf_path)
print("Total graphs:", len(png_files))