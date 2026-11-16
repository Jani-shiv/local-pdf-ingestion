"""Domain-specific extractors for unique day-to-day problems."""
from .contract import Contract, ContractExtractor
from .invoice import Invoice, InvoiceExtractor
from .medical import MedicalExtractor, MedicalReport
from .recipe import Recipe, RecipeExtractor
from .study import StudyExtractor, StudyGuide

__all__ = [
    "Contract",
    "ContractExtractor",
    "Invoice",
    "InvoiceExtractor",
    "MedicalExtractor",
    "MedicalReport",
    "Recipe",
    "RecipeExtractor",
    "StudyExtractor",
    "StudyGuide",
]