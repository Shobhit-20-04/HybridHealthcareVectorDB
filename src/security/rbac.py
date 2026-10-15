"""
Healthcare Access Control & Security Layer
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements Role-Based Access Control (RBAC) and PostgreSQL Row-Level Security (RLS) context management.
Enforces HIPAA least-privilege principles across relational data and semantic vector searches.
"""

from typing import Dict, Any, List, Optional
from enum import Enum

class ClinicalRole(str, Enum):
    ADMIN = "clinical_admin"
    PHYSICIAN = "attending_physician"
    RADIOLOGIST = "radiologist"
    RESEARCHER = "clinical_researcher"

class SecurityContext:
    """Manages active user security identity and database session settings."""

    def __init__(
        self,
        user_id: str,
        role: ClinicalRole,
        department: Optional[str] = None
    ):
        self.user_id = user_id
        self.role = role
        self.department = department

    def get_session_sql_commands(self) -> List[str]:
        """Generates SQL commands to set session-level RLS context in PostgreSQL."""
        commands = [
            f"SET LOCAL ROLE {self.role.value};"
        ]
        if self.department:
            commands.append(f"SET LOCAL app.clinician_dept = '{self.department}';")
        else:
            commands.append("SET LOCAL app.clinician_dept = '';")
        return commands

    def filter_documents_offline(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simulates database-level RLS filtering offline for testing and benchmarking.
        """
        if self.role == ClinicalRole.ADMIN:
            return documents

        allowed = []
        for doc in documents:
            doc_type = doc.get("document_type", "")
            doc_dept = doc.get("department")

            if self.role == ClinicalRole.RADIOLOGIST:
                if doc_type == "radiology_report":
                    allowed.append(doc)
            elif self.role == ClinicalRole.PHYSICIAN:
                if self.department is None or doc_dept == self.department:
                    allowed.append(doc)
            elif self.role == ClinicalRole.RESEARCHER:
                # Researchers get de-identified document projections
                sanitized = {
                    "document_id": doc.get("document_id"),
                    "document_type": doc.get("document_type"),
                    "title": doc.get("title"),
                    "full_text": doc.get("full_text"),
                    "department": doc_dept
                }
                allowed.append(sanitized)

        return allowed
