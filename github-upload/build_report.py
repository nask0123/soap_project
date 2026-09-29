from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

out = Path('output/pdf/report.pdf')
out.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodySmall',fontName='Helvetica',fontSize=9.7,leading=12.4,spaceAfter=6))
styles.add(ParagraphStyle(name='SubSmall',fontName='Helvetica-Bold',fontSize=10,leading=12,spaceBefore=5,spaceAfter=4,textColor=colors.HexColor('#184c59')))
styles.add(ParagraphStyle(name='TitleSmall',fontName='Helvetica-Bold',fontSize=17,leading=20,spaceAfter=9,textColor=colors.HexColor('#14343d')))
story=[]
for block in Path('REPORT.md').read_text(encoding='utf-8').replace('<!-- PAGE -->', '\n\n<!-- PAGE -->\n\n').split('\n\n'):
    block=block.strip()
    if not block.strip(): continue
    if block.strip()=='<!-- PAGE -->': story.append(PageBreak()); continue
    style='BodySmall'
    if block.startswith('## '): block=block[3:]; style='SubSmall'
    elif block.startswith('# '): block=block[2:]; style='TitleSmall'
    story.append(Paragraph(escape(block),styles[style]))
def footer(c,doc):
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#647078'))
    c.drawString(34,23,'Soup take-home | Colab evidence, 29 September 2026 UTC')
    c.drawRightString(A4[0]-34,23,str(doc.page))
SimpleDocTemplate(str(out),pagesize=A4,leftMargin=34,rightMargin=34,topMargin=30,bottomMargin=35).build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(out)
print('PDF pages:',len(reader.pages))
assert len(reader.pages)==2, 'Report exceeds two-page limit'
