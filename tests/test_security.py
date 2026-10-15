"""
Unit tests for RBAC and Row-Level Security context management
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import pytest
from src.security.rbac import SecurityContext, ClinicalRole

def test_session_sql_command_generation():
    ctx = SecurityContext(user_id="doc-42", role=ClinicalRole.PHYSICIAN, department="Cardiology")
    commands = ctx.get_session_sql_commands()
    assert any("SET LOCAL ROLE attending_physician;" in c for c in commands)
    assert any("SET LOCAL app.clinician_dept = 'Cardiology';" in c for c in commands)

def test_rls_filtering_radiologist():
    ctx = SecurityContext(user_id="rad-01", role=ClinicalRole.RADIOLOGIST)
    docs = [
        {"title": "Chest X-Ray Note", "document_type": "radiology_report"},
        {"title": "Cardiology Progress Note", "document_type": "progress_note"},
        {"title": "General Discharge Summary", "document_type": "discharge_summary"}
    ]
    filtered = ctx.filter_documents_offline(docs)
    assert len(filtered) == 1
    assert filtered[0]["document_type"] == "radiology_report"

def test_rls_filtering_department_physician():
    ctx = SecurityContext(user_id="phys-01", role=ClinicalRole.PHYSICIAN, department="Cardiology")
    docs = [
        {"title": "Cardio Note", "department": "Cardiology"},
        {"title": "Neuro Note", "department": "Neurology"}
    ]
    filtered = ctx.filter_documents_offline(docs)
    assert len(filtered) == 1
    assert filtered[0]["department"] == "Cardiology"
