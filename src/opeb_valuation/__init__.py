"""
OPEB Full Valuation Engine - Shackleford Precision Edition

Production-ready GASB 75 OPEB valuation engine with scientific precision:

MATHEMATICAL ENHANCEMENTS:
1. Competing Risks: Geometric/logarithmic MDT distribution (not simple summation)
2. Mid-Year Timing: v^{t+0.5} discounting and √(1+trend) adjustment
3. Level % of Pay: True EAN attribution with backward salary projection
4. Joint-Life: Conditional probability vectors for spousal benefits

ENTERPRISE FEATURES:
1. ASOP 23 Imputation: Actuarially defensible defaults for missing data
2. SHA-256 Audit Trail: Cryptographic file hashing for reproducibility
3. PII Anonymization: HIPAA-compliant data handling
4. Quality Reporting: Complete data quality documentation

Compliance:
- GASB Statement No. 75 and Implementation Guide No. 2017-3
- ASOP 4: Measuring Pension Obligations
- ASOP 6: Measuring Retiree Group Benefits Obligations
- ASOP 23: Data Quality
- ASOP 25: Credibility Procedures
- ASOP 35: Selection of Demographic Assumptions

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
Version: 3.0.0 (Shackleford Precision)
"""

__version__ = "3.0.0"
__author__ = "Joseph Shackelford"
__precision__ = "Shackleford"

# Core Engine
from .engine import (
    ValuationEngine,
    CensusRecord,
    MemberResult,
    RetireeValuator,
    ActiveValuator,
    create_engine
)

# Mortality
from .mortality import (
    MortalityCalculator,
    create_mortality_calculator,
    MortalityTable
)

# Decrements with Competing Risk MDT
from .decrements import (
    DecrementTensor,
    DecrementType,
    MultipleDecrementCalculator,
    TerminationRates,
    DisabilityRates,
    RetirementEligibility,
    create_decrement_calculator,
    calculate_mdt_probability,
    calculate_mdt_survival_probability
)

# Financial Engine with Mid-Year Precision
from .financials import (
    FinancialEngine,
    TrendModel,
    MorbidityModel,
    SalaryProjector,
    create_financial_engine
)

# Enterprise Data Ingestion
from .ingestion import (
    CensusLoader,
    CensusResult,
    ImputationEngine,
    PIIAnonymizer,
    DataQualityReportGenerator,
    load_census
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__precision__",
    
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
    
    # Decrements (Shackleford MDT)
    "DecrementTensor",
    "DecrementType",
    "MultipleDecrementCalculator",
    "TerminationRates",
    "DisabilityRates",
    "RetirementEligibility",
    "create_decrement_calculator",
    "calculate_mdt_probability",
    "calculate_mdt_survival_probability",
    
    # Financials (Mid-Year Precision)
    "FinancialEngine",
    "TrendModel",
    "MorbidityModel",
    "SalaryProjector",
    "create_financial_engine",
    
    # Enterprise Ingestion
    "CensusLoader",
    "CensusResult",
    "ImputationEngine",
    "PIIAnonymizer",
    "DataQualityReportGenerator",
    "load_census",
]


def info():
    """Print package information."""
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║         OPEB Full Valuation Engine - Shackleford Precision           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Version: {__version__}                                                      ║
║  Author:  {__author__}                                       ║
║  License: MIT                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  MATHEMATICAL ENHANCEMENTS:                                          ║
║  ✓ Competing Risk MDT (geometric/logarithmic distribution)           ║
║  ✓ Mid-Year Discounting (v^{{t+0.5}})                                  ║
║  ✓ Mid-Year Trending (√(1+trend) adjustment)                         ║
║  ✓ Level % of Pay EAN (backward salary projection)                   ║
║  ✓ Joint-Life Spouse Benefits (conditional probability vectors)      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ENTERPRISE FEATURES:                                                ║
║  ✓ ASOP 23 Data Imputation                                           ║
║  ✓ SHA-256 Audit Trail                                               ║
║  ✓ PII Anonymization                                                 ║
║  ✓ Quality Reporting                                                 ║
╚══════════════════════════════════════════════════════════════════════╝
""")
