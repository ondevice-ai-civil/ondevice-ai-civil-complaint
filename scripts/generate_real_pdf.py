from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

def create_large_pdf(filename, title, content_list):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    
    # 제목 작성
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)
    
    # 본문 작성 (수백 줄 분량)
    c.setFont("Helvetica", 10)
    y = height - 80
    for line in content_list:
        if y < 50: # 페이지 넘김
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50
        c.drawString(50, y, line)
        y -= 15
    
    c.save()

# 사하구 도로점용 매뉴얼 가상 데이터 (5000자 이상 시뮬레이션)
manual_lines = ["[Saha-gu Road Occupation Manual]"] + [f"Section {i}: Technical standards and administrative procedures for road occupation permit in Saha-gu. Clause {i}.1: Application requirements..." for i in range(1, 200)]

# 사하구 주차장 조례 가상 데이터
law_lines = ["[Saha-gu Parking Ordinance]"] + [f"Article {i}: Enforcement rules for parking management in Saha-gu district. Paragraph {i}.a: Fee structure and discount policies..." for i in range(1, 100)]

os.makedirs("data/raw/manuals", exist_ok=True)
os.makedirs("data/raw/laws", exist_ok=True)

create_large_pdf("data/raw/manuals/saha_road_manual_real.pdf", "Saha-gu Road Occupation Manual 2026", manual_lines)
create_large_pdf("data/raw/laws/saha_parking_law_real.pdf", "Saha-gu Parking Ordinance Full Text", law_lines)

print("✅ Real PDF files generated with large content.")
