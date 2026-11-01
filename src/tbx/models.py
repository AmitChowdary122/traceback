"""Answer and check-result models (CONTRACTS.md v1.1 sections 6 and 7)."""
from __future__ import annotations

from pydantic import BaseModel, field_validator


class Citation(BaseModel):
    artifact_id: str
    field: str
    value: str  # always a string, as copied from the tool row

    @field_validator("value", mode="before")
    @classmethod
    def _numbers_as_strings(cls, v):
        # Models sometimes copy numeric cells as JSON numbers; keep the contract type (str).
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, (int, float)):
            return repr(v) if isinstance(v, float) else str(v)
        return v


class Claim(BaseModel):
    text: str
    time_start: str | None = None
    time_end: str | None = None
    citations: list[Citation] = []


class Answer(BaseModel):
    claims: list[Claim]
    note: str | None = None


class Reason(BaseModel):
    code: str
    detail: str
    citation_index: int | None = None


class RejectedClaim(BaseModel):
    claim: Claim
    reasons: list[Reason]


class CheckResult(BaseModel):
    verified: list[Claim] = []
    rejected: list[RejectedClaim] = []
