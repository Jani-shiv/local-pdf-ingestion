"""Domain-specific extractors for unique day-to-day problems."""
from .recipe import RecipeExtractor, Recipe
from .invoice import InvoiceExtractor, Invoice
from .medical import MedicalExtractor, MedicalReport
from .contract import ContractExtractor, Contract
from .study import StudyExtractor, StudyGuide

__all__ = [
    "RecipeExtractor",
    "Recipe",
    "InvoiceExtractor",
    "Invoice",
    "MedicalExtractor",
    "MedicalReport",
    "ContractExtractor",
    "Contract",
    "StudyExtractor",
    "StudyGuide",
]