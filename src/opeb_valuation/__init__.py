"""
OPEB Full Valuation Engine

Production-ready GASB 75 OPEB valuation engine implementing Entry Age Normal
actuarial cost method with comprehensive per-member liability calculations.

Version: 2.1.0 (West Florida Planning Corrections - 2025-12-28)

Compliance:
- GASB Statement No. 75 and Implementation Guide No. 2017-3
- ASOP 4: Measuring Pension Obligations
- ASOP 6: Measuring Retiree Group Benefits Obligations
- ASOP 25: Credibility Procedures
- ASOP 35: Selection of Demographic Assumptions

Author: Actuarial Pipeline Project
License: MIT
"""

__version__ = "2.1.0"
__author__ = "Actuarial Pipeline Project"

from .engine import (
    ValuationEngine,
    CensusRecord,
    MemberResult,
    RetireeValuator,
    ActiveValuator,
    create_engine
)

from .mortality import (
    MortalityCalculator,
    create_mortality_calculator,
    MortalityTable
)

from .decrements import (
    DecrementTensor,
    DecrementType,
    MultipleDecrementCalculator,
    TerminationRates,
    DisabilityRates,
    RetirementEligibility,
    create_decrement_calculator
)

from .financials import (
    FinancialEngine,
    TrendModel,
    MorbidityModel,
    create_financial_engine
)

from .excel_updater import (
    # Main functions
    update_full_valuation_excel,
    verify_valuation_output,
    print_valuation_summary,
    
    # Data classes
    FullValuationInputs,
    FullValuationResults,
    
    # Helper functions
    copy_cell_format,
    adjust_formula_row,
)

__all__ = [
    # Main engine
    "ValuationEngine",
    "create_engine",
    
    # Census and results
    "CensusRecord",
    "MemberResult",
    
    # Sub-valuators
    "RetireeValuator",
    "ActiveValuator",
    
    # Mortality
    "MortalityCalculator",
    "create_mortality_calculator",
    "MortalityTable",
    
    # Decrements
    "DecrementTensor",
    "DecrementType",
    "MultipleDecrementCalculator",
    "TerminationRates",
    "DisabilityRates",
    "RetirementEligibility",
    "create_decrement_calculator",
    
    # Financials
    "FinancialEngine",
    "TrendModel",
    "MorbidityModel",
    "create_financial_engine",
    
    # Excel Updater (NEW)
    "update_full_valuation_excel",
    "verify_valuation_output",
    "print_valuation_summary",
    "FullValuationInputs",
    "FullValuationResults",
    "copy_cell_format",
    "adjust_formula_row",
]
