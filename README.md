# OPEB Full Valuation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GASB 75 Compliant](https://img.shields.io/badge/GASB%2075-Compliant-green.svg)](https://www.gasb.org/)
[![Precision: Shackleford](https://img.shields.io/badge/Precision-Shackleford-gold.svg)](https://github.com/risky-scout/opeb-full-valuation)

**Production-ready GASB 75 OPEB valuation engine with Entry Age Normal methodology and Excel automation.**

Version 2.1.0 - West Florida Planning Corrections (2025-12-28)

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Risky-Scout/opeb-full-valuation.git
cd opeb-full-valuation
pip install -e .

# Run interactive mode
python run_valuation.py --interactive

# Or command line
python run_valuation.py \
    --census census.xlsx \
    --template prior_year.xlsx \
    --output current_year.xlsx \
    --measurement-date 2025-09-30 \
    --prior-date 2024-09-30 \
    --discount-rate 0.0502 \
    --prior-rate 0.0381 \
    --verify
```

---

## What is a Full Valuation?

A full OPEB valuation calculates liabilities from census data using the Entry Age Normal (EAN) actuarial cost method. Unlike roll-forwards, full valuations:

- Use current census data
- Calculate actual experience gains/losses
- Provide per-member liability detail
- Required at least biennially per GASB 75

**Entry Age Normal Method:**
```
Normal_Cost_Rate = PVFB_entry / (Salary_entry × ä_sal)
Service_Cost = Normal_Cost_Rate × Current_Salary
TOL = PVFB - PVFNC
```

---

## Features

### 1. Valuation Engine
- **Entry Age Normal** cost method per GASB 75 ¶162
- **Pub-2010 Mortality** with MP-2021 generational projection
- **Multiple Decrement Tables** (MDT) with competing risks
- **Getzen Healthcare Trend** model
- **Per-member calculations** for detailed analysis

### 2. Excel Template Automation
- Updates existing GASB 75 Excel templates
- Handles all 10 worksheet tabs
- Preserves formula structures
- Copies cell formatting correctly

### 3. Quality Verification
- Automated verification checks
- Validates output formulas evaluate correctly

---

## Critical Fixes (2025-12-28)

This version incorporates all corrections from production debugging:

| # | Fix | Why It Matters |
|---|-----|----------------|
| 1 | Clear OPEB Exp & Def C6:C28 | Prevents leftover data in reports |
| 2 | Net OPEB D22:D25 No Fill | Consistent formatting |
| 3 | RSI I23 = benefit changes | Documents plan modifications |
| 4 | RSI I26 = VALUE | Must be number, not formula |
| 5 | B14/B26 copy FULL style | Formatting matches adjacent cells |
| 6 | B13 handles experience | Formula references Net OPEB correctly |
| 7 | Skip Table7AmortDeferred2 | Not currently used |

---

## Module Structure

```
opeb-full-valuation/
├── run_valuation.py                    # Main entry point
├── src/opeb_valuation/
│   ├── __init__.py                     # Package exports
│   ├── engine.py                       # EAN valuation engine
│   ├── mortality.py                    # Pub-2010 + MP-2021
│   ├── decrements.py                   # MDT calculations
│   ├── financials.py                   # Trend & discounting
│   ├── gasb_disclosure.py              # Table generators
│   └── excel_updater.py                # Excel automation (NEW)
├── tests/
│   └── test_valuation.py               # Validation tests
├── README.md
└── pyproject.toml
```

---

## Usage

### Python API

```python
from datetime import date
from opeb_valuation import (
    create_engine,
    update_full_valuation_excel,
    FullValuationInputs,
    FullValuationResults,
)

# Create valuation engine
engine = create_engine(
    discount_rate=0.0502,
    measurement_date=date(2025, 9, 30),
)

# Run valuation on census
results = engine.run_valuation(census_df)

# Create input/result objects
inputs = FullValuationInputs(
    prior_measurement_date=date(2024, 9, 30),
    new_measurement_date=date(2025, 9, 30),
    prior_discount_rate=0.0381,
    new_discount_rate=0.0502,
    active_count=86,
    retiree_count=31,
    covered_payroll=5_200_000,
    benefit_changes="None",
)

results = FullValuationResults(
    boy_tol_old_rate=6_911_729,
    boy_tol_new_rate=6_200_000,
    eoy_tol=10_201_072,
    sensitivity_disc_plus=8_500_000,
    sensitivity_disc_minus=12_500_000,
    sensitivity_trend_plus=12_000_000,
    sensitivity_trend_minus=8_800_000,
    service_cost=683_256,
    interest=285_000,
    assumption_change=-711_729,
    experience=3_033_616,  # Non-zero for full valuations!
    active_count=86,
    retiree_count=31,
    covered_payroll=5_200_000,
)

# Update Excel
update_full_valuation_excel(
    input_path='prior_year.xlsx',
    output_path='current_year.xlsx',
    inputs=inputs,
    results=results,
)
```

### Command Line

```bash
# Interactive (prompts for all inputs)
python run_valuation.py --interactive

# Full command line
python run_valuation.py \
    --census "census_2025.xlsx" \
    --template "GASB_75_2024.xlsx" \
    --output "GASB_75_2025.xlsx" \
    --measurement-date 2025-09-30 \
    --prior-date 2024-09-30 \
    --discount-rate 0.0502 \
    --prior-rate 0.0381 \
    --benefit-changes "None" \
    --verify
```

---

## Valuation Pipeline

### Stage 1: Census Data Preparation
```python
# Load and validate census
census_df = pd.read_excel('census.xlsx')

# Required fields
required = ['participant_id', 'dob', 'doh', 'gender', 'status', 
            'coverage_tier', 'salary']
```

### Stage 2: Run Valuation Engine
```python
# Create engine with assumptions
engine = create_engine(
    discount_rate=0.0502,
    initial_trend=0.065,
    ultimate_trend=0.045,
    measurement_date=date(2025, 9, 30),
)

# Run for all members
results = engine.run_valuation(census_df)

# Get aggregates
total_tol = results['tol'].sum()
total_sc = results['service_cost'].sum()
```

### Stage 3: Calculate Sensitivities
```python
# Discount rate sensitivities
engine_plus = create_engine(discount_rate=0.0602)
engine_minus = create_engine(discount_rate=0.0402)

tol_disc_plus = engine_plus.run_valuation(census_df)['tol'].sum()
tol_disc_minus = engine_minus.run_valuation(census_df)['tol'].sum()

# Trend sensitivities
engine_trend_plus = create_engine(initial_trend=0.075)
engine_trend_minus = create_engine(initial_trend=0.055)
```

### Stage 4: Update Excel Disclosure
```python
from opeb_valuation import update_full_valuation_excel

update_full_valuation_excel(
    input_path='prior_year.xlsx',
    output_path='current_year.xlsx',
    inputs=inputs,
    results=results,
)
```

### Stage 5: Verify Output
```python
from opeb_valuation import verify_valuation_output

verification = verify_valuation_output('current_year.xlsx')
if verification['passed']:
    print("All checks passed!")
```

---

## Excel Template Structure

| Sheet | Purpose |
|-------|---------|
| Model Inputs | Primary data input (TOL, rates, dates, census) |
| Net OPEB | TOL roll-forward reconciliation |
| RSI | Required Supplementary Information (10-year history) |
| OPEB Exp & Def | OPEB Expense and Deferred items |
| Table7AmortDeferred | Deferred inflows/outflows amortization |
| AmortDeferredOutsIns | ARSL tracking |
| 1%-+ | Sensitivity analysis |
| Assumptions | Actuarial assumptions |

---

## Verification Checklist

After running, open the output in Excel and verify:

| Check | Expected |
|-------|----------|
| Net OPEB D22 | Experience (gain)/loss value |
| Net OPEB D22:D25 | No background fill |
| Table7AmortDeferred AI49 | "GOOD" |
| OPEB Exp & Def H40 | "GOOD" |
| RSI I23 | Benefit changes description |
| RSI I26 | Discount rate (e.g., 5.02%) |
| B14, B26 formatting | Matches adjacent cells |

---

## GASB 75 Compliance

This engine complies with:

- **GASB Statement No. 75**: Accounting and Financial Reporting for Postemployment Benefits Other Than Pensions
- **GASB Implementation Guide No. 2017-3**: Practical implementation guidance
- **ASOP 4**: Measuring Pension Obligations
- **ASOP 6**: Measuring Retiree Group Benefits Obligations
- **ASOP 25**: Credibility Procedures
- **ASOP 35**: Selection of Demographic Assumptions

---

## Mathematical Precision

### Geometric Fractional Interpolation
```
q_{x+f} = q_x^{1-f} × q_{x+1}^f
```

### Competing Risk MDT
```
q_j(mdt) = [ln(1-q_j') / ln(p_total)] × q_total
```

### Mid-Year Discounting
```
Discount: v^{t+0.5}
Trend: τ(t) = CumTrend_{t-1} × √(1+Trend_t)
```

---

## Dependencies

- Python 3.10+
- numpy
- pandas
- openpyxl

```bash
pip install numpy pandas openpyxl
```

---

## License

MIT License - See LICENSE file

---

## Author

**Joseph Shackelford** - Actuarial Pipeline Project

---

## Changelog

### v2.1.0 (2025-12-28)
- Added `excel_updater.py` with production-ready Excel automation
- Incorporated all West Florida Planning corrections
- Added verification functions
- Added interactive mode runner
- Updated documentation

### v2.0.0
- Production release with complete valuation engine
- Entry Age Normal implementation
- Multiple decrement tables
- Sensitivity calculations
