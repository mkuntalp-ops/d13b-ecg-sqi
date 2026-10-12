import os
import re
import subprocess
import shutil

REPO_DIR = "/Volumes/KIOXIA/Projects/d13b-ecg-sqi"
MD_PATH = os.path.join(REPO_DIR, "manuscript/final_manuscript.md")
HTML_PATH = "/tmp/manuscript_formatted.html"
PDF_REPO_PATH = os.path.join(REPO_DIR, "manuscript/Zero-shot-and-task-specific-models-for-ECG-usability-screening-patient-independent-performance-cross-dataset-transfer-and-computational-cost.pdf")
DOWNLOADS_PDF_PATH = "/Users/mehmetkuntalp/Downloads/Zero-shot-and-task-specific-models-for-ECG-usability-screening-patient-independent-performance-cross-dataset-transfer-and-computational-cost.pdf"
BRAIN_PDF_PATH = "/Users/mehmetkuntalp/.gemini/antigravity/brain/fd0d5598-c574-4cf6-a79a-861d3e51441d/Zero-shot-and-task-specific-models-for-ECG-usability-screening-patient-independent-performance-cross-dataset-transfer-and-computational-cost.pdf"

print("Reading markdown:", MD_PATH)
with open(MD_PATH, "r") as f:
    md_text = f.read()

# Replace relative image paths with absolute file:// URLs so Chrome resolves them cleanly
figures_dir = os.path.join(REPO_DIR, "figures")
def replace_img(match):
    alt = match.group(1)
    img_name = os.path.basename(match.group(2))
    abs_path = os.path.join(figures_dir, img_name)
    return f"![{alt}]({abs_path})"

md_text = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_img, md_text)

# Write temp markdown
temp_md = "/tmp/temp_manuscript_processed.md"
with open(temp_md, "w") as f:
    f.write(md_text)

# Run pandoc to convert to HTML
pandoc_bin = "/opt/homebrew/bin/pandoc"
subprocess.run([
    pandoc_bin,
    temp_md,
    "-s",
    "--metadata", "title=Zero-shot and task-specific models for ECG usability screening: patient-independent performance, cross-dataset transfer, and computational cost",
    "-o", HTML_PATH
], check=True)

print("Generated raw HTML:", HTML_PATH)

# Inject custom publication CSS into HTML
with open(HTML_PATH, "r") as f:
    html_content = f.read()

CUSTOM_CSS = """
<style>
@page {
  size: A4 portrait;
  margin: 18mm 14mm 18mm 14mm;
}

html {
  margin: 0 !important;
  padding: 0 !important;
  background-color: #ffffff !important;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 9.2pt;
  line-height: 1.45;
  color: #1e293b;
  margin: 0 !important;
  padding: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
  background-color: #ffffff !important;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

h1 {
  font-size: 15pt;
  font-weight: 700;
  color: #0f172a;
  margin-top: 0;
  margin-bottom: 12px;
  line-height: 1.3;
}

h2 {
  font-size: 11.5pt;
  font-weight: 700;
  color: #0f172a;
  border-bottom: 1.5px solid #cbd5e1;
  padding-bottom: 4px;
  margin-top: 18px;
  margin-bottom: 8px;
  page-break-after: avoid;
}

h3 {
  font-size: 10.2pt;
  font-weight: 600;
  color: #1e293b;
  margin-top: 14px;
  margin-bottom: 6px;
  page-break-after: avoid;
}

h4 {
  font-size: 9.2pt;
  font-weight: 600;
  color: #334155;
  margin-top: 12px;
  margin-bottom: 4px;
  page-break-after: avoid;
}

p {
  margin-top: 0;
  margin-bottom: 6px;
  text-align: justify;
}

hr {
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 16px 0;
}

/* Publication Compact Table Styling - STRICT OVERFLOW PREVENTION & FULL WIDTH */
table {
  width: 100% !important;
  max-width: 100% !important;
  border-collapse: collapse !important;
  margin: 10px 0 14px 0 !important;
  font-size: 7.2pt !important;
  table-layout: fixed !important;
  page-break-inside: avoid !important;
  border: 1px solid #cbd5e1 !important;
  box-sizing: border-box !important;
}

th, td {
  border: 1px solid #cbd5e1 !important;
  padding: 3.5px 4.5px !important;
  text-align: center !important;
  vertical-align: middle !important;
  word-wrap: break-word !important;
  overflow-wrap: break-word !important;
  hyphens: auto !important;
  line-height: 1.25 !important;
  box-sizing: border-box !important;
}

th:first-child, td:first-child {
  text-align: left !important;
}

th {
  background-color: #f1f5f9 !important;
  font-weight: 600 !important;
  color: #0f172a !important;
}

tr:nth-child(even) td {
  background-color: #f8fafc !important;
}

img {
  max-width: 95% !important;
  height: auto !important;
  display: block;
  margin: 12px auto !important;
  page-break-inside: avoid !important;
}

em {
  font-size: 8.5pt;
  color: #475569;
  display: block;
  margin-bottom: 6px;
}
</style>
"""

# Strip out pandoc's default styles and inject custom publication CSS
html_content = re.sub(r'<style>.*?</style>', '', html_content, flags=re.DOTALL)
html_content = html_content.replace("</head>", f"{CUSTOM_CSS}\n</head>")

with open(HTML_PATH, "w") as f:
    f.write(html_content)

print("Injected publication CSS into HTML (stripped Pandoc defaults)")

# Print to PDF using Headless Chrome
chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_bin,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_REPO_PATH}",
    HTML_PATH
]

print("Running Chrome print-to-pdf...")
subprocess.run(cmd, check=True)
print("Generated PDF at:", PDF_REPO_PATH)

# Copy PDF to Downloads and Brain artifact directory
shutil.copy2(PDF_REPO_PATH, DOWNLOADS_PDF_PATH)
print("Copied PDF to Downloads:", DOWNLOADS_PDF_PATH)
shutil.copy2(PDF_REPO_PATH, BRAIN_PDF_PATH)
print("Copied PDF to Brain Artifacts:", BRAIN_PDF_PATH)

print("\nPDF Generation Finished Successfully!")
