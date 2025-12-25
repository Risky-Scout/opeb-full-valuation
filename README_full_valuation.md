# OPEB Full Valuation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GASB 75 Compliant](https://img.shields.io/badge/GASB%2075-Compliant-green.svg)](https://www.gasb.org/)

Production-ready GASB 75 OPEB valuation engine implementing Entry Age Normal (EAN) actuarial cost method.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/risky-scout/opeb-full-valuation.git
cd opeb-full-valuation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Run Your First Valuation

```python
import pandas as pd
from datetime import date
from opeb_valuation import create_engine

# 1. Configure the valuation
config = {
    'valuation_date': date(2025, 9, 30),
    'discount_rate': 0.0381,          # EOY discount rate (Bond Buyer 20-Bond Index)
    'discount_rate_boy': 0.0409,      # BOY discount rate
    'mortality_load': 1.20,           # 120% of Pub-2010 base rates
    'contribution_rate': 0.45,        # Employee pays 45% of premium
    'dental_premiums': {
        'Employee': 13.24,
        'Employee + Spouse': 26.26,
        'Employee + Child(ren)': 28.93,
        'Employee + Family': 45.98,
    },
    'admin_fee_monthly': 35.44,
    'married_fraction': 0.40,         # 40% assumed married
    'spouse_age_diff': -3,            # Spouse 3 years younger
    'avg_service_life': 5.0,          # For amortization
}

# 2. Load your census data
actives = pd.read_excel('census.xlsx', sheet_name='Actives')
retirees = pd.read_excel('census.xlsx', sheet_name='Retirees')

# 3. Run the valuation
engine = create_engine(config)
results = engine.run_valuation(actives, retirees)

# 4. View results
print(f"Total OPEB Liability: ${results['TOL'].sum():,.0f}")
print(f"Service Cost: ${results['ServiceCost'].sum():,.0f}")
print(f"Active TOL: ${results[results['Status']=='Active']['TOL'].sum():,.0f}")
print(f"Retiree TOL: ${results[results['Status']=='Retiree']['TOL'].sum():,.0f}")

# 5. Export to Excel
results.to_excel('valuation_results.xlsx', index=False)
```

---

## 📊 Census Data Format

Your census Excel file should have these columns:

### Actives Sheet

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `MemberID` | string | Unique identifier | A001 |
| `DOB` | date | Date of birth | 1980-05-15 |
| `DOH` | date | Date of hire | 2010-03-01 |
| `Gender` | string | M or F | M |
| `CoverageLevel` | string | Coverage tier | Employee + Spouse |
| `AnnualSalary` | float | Current salary | 65000 |

### Retirees Sheet

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `MemberID` | string | Unique identifier | R001 |
| `DOB` | date | Date of birth | 1955-08-20 |
| `DOH` | date | Date of hire | 1985-01-15 |
| `Gender` | string | M or F | F |
| `CoverageLevel` | string | Coverage tier | Employee |

---

## 📁 Project Structure

```
opeb-full-valuation/
├── src/
│   └── opeb_valuation/
│       ├── __init__.py          # Package exports
│       ├── engine.py            # Core valuation engine
│       ├── mortality.py         # Pub-2010 + MP-2021 mortality
│       ├── decrements.py        # Termination, disability, retirement
│       ├── financials.py        # Discounting, trending, morbidity
│       └── gasb_disclosure.py   # GASB 75 footnote table generator
├── tests/
│   └── test_valuation.py        # Actuarial validation tests
├── pyproject.toml               # Package configuration
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## 🔧 Configuration Reference

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `valuation_date` | date | Measurement date |
| `discount_rate` | float | EOY discount rate (e.g., 0.0381 for 3.81%) |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discount_rate_boy` | Same as EOY | BOY discount rate |
| `mortality_load` | 1.20 | Mortality table load factor |
| `contribution_rate` | 0.45 | Employee contribution % |
| `married_fraction` | 0.40 | Assumed % married |
| `spouse_age_diff` | -3 | Spouse age difference |
| `max_age` | 110 | Maximum age in projection |
| `avg_service_life` | 5.0 | ARSL for amortization |

---

## 📈 Output Columns

The `results` DataFrame contains:

| Column | Description |
|--------|-------------|
| `MemberID` | Participant identifier |
| `Status` | Active or Retiree |
| `Age` | Current age |
| `Service` | Years of service |
| `PVFB` | Present Value of Future Benefits |
| `TOL` | Total OPEB Liability (attributed) |
| `ServiceCost` | Annual service cost |
| `Medical_PVFB` | Medical component |
| `Dental_PVFB` | Dental component |
| `AttributionRatio` | EAN attribution (0 to 1) |
| `RetirementAge` | Expected retirement age |
| `ProbReachRetirement` | Probability of reaching retirement |

---

## 📋 Generate GASB 75 Disclosure Tables

```python
from opeb_valuation.gasb_disclosure import create_disclosure_generator
from datetime import date

generator = create_disclosure_generator(
    measurement_date=date(2025, 9, 30),
    prior_date=date(2024, 9, 30),
    tol_boy=6911729,
    tol_eoy=7712986,
    service_cost=683256,
    benefit_payments=450000,
    discount_rate_boy=0.0409,
    discount_rate_eoy=0.0381,
    active_count=86,
    retiree_count=31,
    covered_payroll=5200000,
    sensitivity_results={
        'dr_minus1': 9500000,
        'dr_plus1': 6200000,
        'trend_minus1': 6500000,
        'trend_plus1': 9200000,
    },
    avg_service_life=5.0,
    client_name="City of DeRidder"
)

# Generate all tables
tables = generator.generate_all_tables()

# Export to Excel
generator.export_to_excel('GASB75_Disclosures.xlsx')
```

---

## 🧪 Run Validation Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_valuation.py::TestPureAnnuityCheck -v
```

### Validation Tests Included

1. **No-Subsidy Null Hypothesis** - Verifies $0 liability when employee pays 100%
2. **Pure Annuity Check** - Verifies $50k for 5-year $10k annuity at 0% rate
3. **Implicit Slope Sensitivity** - Verifies age-graded > flat morbidity
4. **EAN Service Cost Logic** - Verifies attribution for new hires

---

## 🔬 Run Sensitivity Analysis

```python
from opeb_valuation import create_engine
from datetime import date

# Base configuration
base_config = {
    'valuation_date': date(2025, 9, 30),
    'discount_rate': 0.0381,
}

# Run at different discount rates
for rate in [0.0281, 0.0381, 0.0481]:
    config = {**base_config, 'discount_rate': rate}
    engine = create_engine(config)
    results = engine.run_valuation(actives, retirees)
    print(f"Rate {rate:.2%}: TOL = ${results['TOL'].sum():,.0f}")
```

---

## 📐 Mathematical Framework

### Entry Age Normal Attribution (GASB 75 ¶162)

```
TOL = PVFB × (Past Service / Expected Total Service)
```

### Active Employee PVFB

```
PVFB = Σ [P(retire at r) × v^(r-x) × Annuity(r) × Benefit(r)]
```

### Interest Cost (GASB 75 ¶44)

```
Interest = (BOY_TOL + SC/2 - Benefits/2) × Discount_Rate
```

---

## 📜 Compliance

- **GASB Statement No. 75** - Accounting and Financial Reporting for OPEB
- **GASB Implementation Guide No. 2017-3**
- **ASOP 4** - Measuring Pension Obligations
- **ASOP 6** - Measuring Retiree Group Benefits Obligations
- **ASOP 25** - Credibility Procedures
- **ASOP 35** - Selection of Demographic Assumptions

---

## 🛠️ Troubleshooting

### Import Error
```bash
pip install -e .  # Reinstall in development mode
```

### Missing Dependencies
```bash
pip install pandas numpy openpyxl
```

### Date Parsing Issues
Ensure dates in Excel are formatted as dates, not text.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

## 👤 Author

**Joseph Shackelford** - Actuarial Pipeline Project

---

## ⚠️ Disclaimer

This software is provided for educational and professional use. Actuarial valuations for official financial reporting should be reviewed and signed by a qualified actuary.
