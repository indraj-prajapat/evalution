"""Extractors package — auto-discovers all entity extractors."""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseExtractor

# Import all concrete extractors so they register themselves
from .company import CompanyExtractor
from .financial_year import FinancialYearExtractor
from .project import ProjectExtractor
from .partner import PartnerExtractor
from .certificate import CertificateExtractor
from .registration import RegistrationExtractor
from .experience import ExperienceExtractor
from .equipment import EquipmentExtractor
from .personnel import PersonnelExtractor
from .tax_record import TaxRecordExtractor
from .compliance import ComplianceRecordExtractor
from .declaration import DeclarationExtractor
from .other import OtherExtractor

# Registry mapping entity_type -> extractor class
EXTRACTOR_REGISTRY: Dict[str, Type[BaseExtractor]] = {
    cls.entity_type: cls
    for cls in [
        CompanyExtractor,
        FinancialYearExtractor,
        ProjectExtractor,
        PartnerExtractor,
        CertificateExtractor,
        RegistrationExtractor,
        ExperienceExtractor,
        EquipmentExtractor,
        PersonnelExtractor,
        TaxRecordExtractor,
        ComplianceRecordExtractor,
        DeclarationExtractor,
        OtherExtractor,
    ]
}


def get_extractor(entity_type: str, tracker) -> BaseExtractor:
    """Return an extractor instance for the given entity type.

    Falls back to OtherExtractor if the type is not recognised.
    """
    cls = EXTRACTOR_REGISTRY.get(entity_type, OtherExtractor)
    return cls(tracker)