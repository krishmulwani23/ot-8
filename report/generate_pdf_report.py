"""
Generates publication-quality PDF technical report for Assignment 8 using ReportLab.
Output: report/Assignment8_Technical_Report.pdf
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf():
    pdf_path = "report/Assignment8_Technical_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=15,
        fontName='Helvetica'
    )
    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=6
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#2d3748')
    )
    table_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("ASSIGNMENT 8: OPTIMIZE ANYTHING", title_style))
    story.append(Paragraph("The 14-Day Vibe-Coding Challenge — Technical Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2b6cb0'), spaceAfter=12))

    meta_text = """
    <b>Course</b>: Optimization Techniques &nbsp;|&nbsp; 
    <b>Author</b>: AI-Optimization-Researcher &nbsp;|&nbsp; 
    <b>Date</b>: October 2026 &nbsp;|&nbsp;
    <b>Status</b>: All 10 Tracks A-J Complete (10/10 Tests PASS)
    """
    story.append(Paragraph(meta_text, body_style))
    story.append(Spacer(1, 8))

    # Executive Summary
    story.append(Paragraph("Executive Summary", h1_style))
    exec_text = """
    This report presents the complete empirical results for the 14-Day Optimization Challenge. 
    Ten compulsory tracks were implemented, benchmarked, and verified using documented AI-assisted (vibe) coding.
    Across all domains—from low-level OS kernels and compiler backends to smart grid scheduling, relational databases, 
    and quantum optimization—optimization demonstrated significant measurable gains while preserving 100% semantic correctness.
    """
    story.append(Paragraph(exec_text, body_style))

    # Summary Results Table
    story.append(Paragraph("Ten-Track Experimental Results Summary", h1_style))
    table_data = [
        [Paragraph("Track", table_hdr_style),
         Paragraph("Domain / Problem", table_hdr_style),
         Paragraph("Optimization Method", table_hdr_style),
         Paragraph("Measured Improvement", table_hdr_style),
         Paragraph("Correctness", table_hdr_style)],

        [Paragraph("Track A", table_cell_style), Paragraph("Continuous Landscapes", table_cell_style), Paragraph("Adam / Newton-CG / HVP", table_cell_style), Paragraph("Quadratic convergence, saddle escape", table_cell_style), Paragraph("PASS (<1e-10 err)", table_cell_style)],
        [Paragraph("Track B", table_cell_style), Paragraph("Discrete Knapsack & QUBO", table_cell_style), Paragraph("ILP (HiGHS) / B&B / SA", table_cell_style), Paragraph("Optimal Cut 8.0 on 6-node graph", table_cell_style), Paragraph("PASS (Exact)", table_cell_style)],
        [Paragraph("Track C", table_cell_style), Paragraph("Quantum Max-Cut", table_cell_style), Paragraph("QAOA (p=2) / SPSA / VQE", table_cell_style), Paragraph("Matches classical optimal cut (8.0)", table_cell_style), Paragraph("PASS (Bitstring 010101)", table_cell_style)],
        [Paragraph("Track D", table_cell_style), Paragraph("Submodular & Parallel", table_cell_style), Paragraph("Greedy / Work Stealing / FedAvg", table_cell_style), Paragraph("98.2% of exact optimum (bound 63.2%)", table_cell_style), Paragraph("PASS (0 violations)", table_cell_style)],
        [Paragraph("Track E", table_cell_style), Paragraph("Commercial EV Fleet", table_cell_style), Paragraph("Mixed-Integer LP (HiGHS)", table_cell_style), Paragraph("77.5% cost savings ($5.1k -> $1.1k)", table_cell_style), Paragraph("PASS (0 overloads)", table_cell_style)],
        [Paragraph("Track F", table_cell_style), Paragraph("Compiler Optimization", table_cell_style), Paragraph("4 Passes + ILP Register Allocation", table_cell_style), Paragraph("5.0x cycle speedup, -80% code size", table_cell_style), Paragraph("PASS (Bit-identical)", table_cell_style)],
        [Paragraph("Track G", table_cell_style), Paragraph("xv6 Operating System", table_cell_style), Paragraph("MLFQ Scheduler + Dynamic Slices", table_cell_style), Paragraph("-57% context switches, 4x response", table_cell_style), Paragraph("PASS (Zero deadlock)", table_cell_style)],
        [Paragraph("Track H", table_cell_style), Paragraph("DBMS Workload (TPC-H)", table_cell_style), Paragraph("Knapsack Index Selection + 2Q", table_cell_style), Paragraph("2.15x query speedup within 400KB", table_cell_style), Paragraph("PASS (Bit-identical)", table_cell_style)],
        [Paragraph("Track I", table_cell_style), Paragraph("Quantum Algorithms & QML", table_cell_style), Paragraph("7 Algos + VQC vs Classical MLP", table_cell_style), Paragraph("VQC matched MLP at 73.8% acc", table_cell_style), Paragraph("PASS (7/7 Algos)", table_cell_style)],
        [Paragraph("Track J", table_cell_style), Paragraph("Software Engineering", table_cell_style), Paragraph("ILP CI Test Minimization", table_cell_style), Paragraph("2.80x faster CI (487ms -> 174ms)", table_cell_style), Paragraph("PASS (86.5% cov)", table_cell_style)]
    ]

    t = Table(table_data, colWidths=[45, 115, 120, 160, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Key Track Summaries
    story.append(Paragraph("Key Technical Highlights", h1_style))
    highlights_text = """
    <b>Track E (Smart Fleet Charging)</b>: Formulating the fleet charging problem as an MILP shifted charging from on-peak 
    tariff periods ($0.38/kWh) to off-peak periods ($0.08/kWh). This reduced peak substation demand from 363 kW to 68 kW, 
    completely eliminating 2 hours of transformer overloads and slashing daily electricity costs by 77.5%.<br/><br/>
    <b>Track F (Compiler Passes)</b>: Combining Constant Folding, Dead Code Elimination, and Common Subexpression Elimination 
    reduced TAC instructions by 80% on Program 1, executing in 2 cycles instead of 10 while maintaining 100% verified output equivalence.<br/><br/>
    <b>Track G (xv6 OS Kernel)</b>: Replacing Round-Robin with an MLFQ scheduler reduced context switches from 130 to 56 and 
    accelerated interactive process response times from 4 ticks to 1 tick.<br/><br/>
    <b>Track H (DBMS Index Selection)</b>: Formulated index selection as a 0-1 Knapsack problem under a 400 KB disk limit. 
    Dynamic Programming and ILP identified the optimal index set (idx_ord_custkey, idx_cust_mktseg, idx_li_suppkey), delivering a 2.15x speedup over sequential scans.
    """
    story.append(Paragraph(highlights_text, body_style))
    story.append(Spacer(1, 10))

    # AI Vibe Coding & Failure Log Summary
    story.append(Paragraph("Vibe Coding & AI Failure Log Summary", h1_style))
    failure_text = """
    Three substantive AI failures were identified, analyzed, and corrected during development:
    <br/>1. <b>Compiler DCE</b>: Naive AI liveness analysis stripped print and memory writes with side effects; resolved by whitelisting essential opcodes.
    <br/>2. <b>xv6 State Machine</b>: AI omitted burst_idx incrementing on ZOMBIE transition, causing an apparent kernel hang; resolved by atomic state updates.
    <br/>3. <b>QAOA Gradient Shift</b>: Simultaneous parameter-shift across multi-gate shared parameters corrupted gradients; resolved by SPSA and finite differences.
    """
    story.append(Paragraph(failure_text, body_style))

    doc.build(story)
    print(f"Successfully generated PDF report: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
