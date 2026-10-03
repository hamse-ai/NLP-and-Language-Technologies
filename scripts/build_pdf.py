"""Builds reports/Formative2_Report.pdf from reports/report.md. Run: python scripts/build_pdf.py"""
import base64
import re
from pathlib import Path

import markdown
import matplotlib
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"


def strip_h1(md_text: str) -> str:
    """Drop the first H1 heading from an included file (report.md supplies its own)."""
    lines = md_text.splitlines()
    out, dropped = [], False
    for line in lines:
        if not dropped and line.startswith("# "):
            dropped = True
            continue
        out.append(line)
    return "\n".join(out)


def replace_placeholder(text: str, marker: str, replacement: str) -> str:
    """Replace the '*(... `marker` ...)*' placeholder paragraph with real content."""
    pattern = re.compile(r"\*\(.*?\[`" + re.escape(marker) + r"`\].*?\)\*", re.DOTALL)
    new_text, n = pattern.subn(replacement, text, count=1)
    if n == 0:
        raise RuntimeError(f"Placeholder for {marker} not found in report.md")
    return new_text


def inline_images(html: str) -> str:
    """Replace every local <img src="...png"> with a base64 data URI."""
    def repl(match):
        src = match.group(1)
        if src.startswith("http") or src.startswith("data:"):
            return match.group(0)
        img_path = Path(src)
        if not img_path.exists():
            print("WARNING: image not found, skipping inline:", src)
            return match.group(0)
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return match.group(0).replace(src, f"data:image/png;base64,{data}")

    return re.sub(r'src="([^"]+\.png)"', repl, html)


def main():
    report = (REPORTS / "report.md").read_text(encoding="utf-8")
    related_work = (REPORTS / "related_work.md").read_text(encoding="utf-8")
    eda_summary = (REPORTS / "eda_summary.md").read_text(encoding="utf-8")

    title, subtitle = "", ""
    fm_match = re.match(r"^---\n(.*?)\n---\n", report, flags=re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("subtitle:"):
                subtitle = line.split(":", 1)[1].strip().strip('"')
        report = report[fm_match.end():]

    report = replace_placeholder(report, "related_work.md", strip_h1(related_work))
    report = replace_placeholder(report, "eda_summary.md", strip_h1(eda_summary))

    report = report.replace("](../figures/", f"]({FIGURES.as_posix()}/")
    report = report.replace("](figures/", f"]({FIGURES.as_posix()}/")
    report = re.sub(r"\]\(([^)]+\.md[^)]*)\)", "](#)", report)  # dead .md cross-links -> anchors

    html_body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    html_body = inline_images(html_body)

    mpl_font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular_b64 = base64.b64encode((mpl_font_dir / "DejaVuSans.ttf").read_bytes()).decode("ascii")
    bold_b64 = base64.b64encode((mpl_font_dir / "DejaVuSans-Bold.ttf").read_bytes()).decode("ascii")

    title_block = ""
    if title:
        title_block = f'<h1 class="doctitle">{title}</h1>'
        if subtitle:
            title_block += f'<p class="docsubtitle">{subtitle}</p>'

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @font-face {{
    font-family: "BodyFont";
    src: url("data:font/ttf;base64,{regular_b64}");
  }}
  @font-face {{
    font-family: "BodyFont";
    src: url("data:font/ttf;base64,{bold_b64}");
    font-weight: bold;
  }}
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: "BodyFont"; font-size: 10.5pt; line-height: 1.4; }}
  h1 {{ font-size: 18pt; margin-top: 22px; border-bottom: 1.5pt solid #333; padding-bottom: 4px; }}
  h1.doctitle {{ font-size: 20pt; border-bottom: none; margin-top: 0; text-align: center; }}
  p.docsubtitle {{ text-align: center; font-size: 12pt; color: #444; margin-top: -6px; margin-bottom: 20px; }}
  h2 {{ font-size: 14pt; margin-top: 16px; color: #1a1a1a; }}
  h3 {{ font-size: 12pt; margin-top: 12px; }}
  table {{ width: 100%; margin: 10px 0; font-size: 9pt; }}
  th, td {{ border: 0.75pt solid #999; padding: 4px 6px; text-align: left; }}
  th {{ background-color: #eee; }}
  img {{ max-width: 480px; display: block; margin: 8px auto; }}
  code {{ background: #f4f4f4; padding: 1px 3px; }}
</style>
</head>
<body>
{title_block}
{html_body}
</body>
</html>
"""

    out_pdf = REPORTS / "Formative2_Report.pdf"
    with open(out_pdf, "wb") as f:
        result = pisa.CreatePDF(html_doc, dest=f)

    if result.err:
        raise RuntimeError(f"PDF build failed with {result.err} error(s)")
    print("Wrote:", out_pdf)


if __name__ == "__main__":
    main()
