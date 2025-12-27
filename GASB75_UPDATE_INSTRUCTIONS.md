# GASB 75 OPEB Disclosure File Update Instructions

## Overview

This document describes the exact process for updating a GASB 75 OPEB disclosure Excel file from one measurement period to the next. This template works for both **full valuations** and **roll-forward valuations** - the only difference is the source of the values entered into ProVal1.

## File Structure Summary

| Tab | Purpose | Update Method |
|-----|---------|---------------|
| ProVal1 | Source data input | Manual entry of valuation results |
| Assumptions | Dates and assumptions | Manual update of dates and rates |
| Net OPEB | Table 3 - Changes in TOL | Formulas auto-calculate from ProVal1 |
| 1%-+ | Table 4 - Sensitivities | Formulas auto-calculate from ProVal1 |
| OPEB Exp & Def | Table 5 - Expense & Deferrals | Formulas auto-calculate |
| RSI | Table 6 - Required Supplementary Info | Column shift + formulas |
| Table7AmortDeferred | Table 7 - Amortization | Formulas from AmortDeferredOutsIns |
| AmortDeferredOutsIns | Amortization engine | Year update + ARSL shift |

## Critical Formula Dependencies

```
ProVal1 → Net OPEB (via VLOOKUP)
   D12 (BOY TOL) = VLOOKUP("EAN Acctg Liab", ProVal1!A:B, 2, FALSE) = ProVal1!B19
   D14 (Service Cost) = VLOOKUP("EAN Acctg Normal Cost", ProVal1!A:B, 2, FALSE) = ProVal1!B38
   D16 (Interest) = (D12 + 0.5*D14) * Assumptions!C11  ← Uses PRIOR discount rate
   D18 (Assumption Change) = VLOOKUP(...C19) - D12 = ProVal1!C19 - ProVal1!B19
   D22 (Experience) = D29 - D24 - D14 - D16 - D12 - D18 - D20 (RESIDUAL)
   D29 (EOY TOL) = VLOOKUP("EAN Acctg Liab", ProVal1!A:D, 4, FALSE) = ProVal1!D19

Net OPEB → RSI Current Year Column (formulas in current year, hardcoded in historical)
   Row 7 (Experience) = 'Net OPEB'!D22
   Row 8 (Assumptions) = 'Net OPEB'!D18

RSI → AmortDeferredOutsIns (via INDEX lookup by year)
   B13 = 'Net OPEB'!D22 (current year experience - LIVE formula)
   B14 = INDEX(RSI!$A$7:$K$7, 1, 2+A14-$C$9) (looks up RSI column by year)
   
   The INDEX formula maps years to RSI columns:
   Year 2018 → column 2+2018-2018 = 2 (B)
   Year 2019 → column 2+2019-2018 = 3 (C)
   ...
   Year 2024 → column 2+2024-2018 = 8 (H)
   Year 2025 → column 2+2025-2018 = 9 (I)

AmortDeferredOutsIns → Table7AmortDeferred (direct cell references)
   All cells in Table7AmortDeferred point to corresponding cells in AmortDeferredOutsIns
```

## ARSL (Average Remaining Service Life) - Critical Understanding

**ARSL values are tied to YEARS, not ROWS.**

Each year's ARSL is determined by the actuary at the time of that valuation and stays fixed forever. When the year rows shift down (2024 moves from row 13 to row 14), the ARSL must follow its year.

Example:
- 2024's ARSL = 5 (determined when 2024 was valued)
- 2023's ARSL = 6 (determined when 2023 was valued)
- 2022's ARSL = 7 (determined when 2022 was valued)

When updating from 2024 to 2025:
- Row 13 becomes 2025 (new ARSL from formula)
- Row 14 becomes 2024 (needs ARSL = 5)
- Row 15 becomes 2023 (needs ARSL = 6)
- etc.

**Therefore, ALL C column values must shift down by one row.**

## Step-by-Step Update Process

### STEP 1: Hardcode RSI Current Year Column (Preserve Prior Year Data)

**Purpose:** The current year column has formulas pointing to Net OPEB. Before updating, these must be hardcoded so the prior year's data isn't lost when Net OPEB updates to the new year.

**Example:** Updating from 2024 to 2025 - Column H is 2024, Column I will be 2025

Replace formulas with their current VALUES in column H:
```
H3:  2024 (was =YEAR(Assumptions!C4))
H4:  [Service Cost value] (was ='Net OPEB'!D14)
H5:  [Interest value] (was ='Net OPEB'!D16)
H6:  [Benefit Changes value] (was ='Net OPEB'!D20)
H7:  [Experience G/L value] (was ='Net OPEB'!D22)  ← CRITICAL
H8:  [Assumption Changes value] (was ='Net OPEB'!D18)  ← CRITICAL
H9:  [Benefit Payments value] (was ='Net OPEB'!D24)
H10: [Net Change value] (was ='Net OPEB'!D27)
H12: [BOY TOL value] (was ='Net OPEB'!D12)
H14: [EOY TOL value] (was ='Net OPEB'!D29)
H17: [Covered Payroll value]
H20: [TOL as % of Payroll value] (was =H14/H17)
H26: [Discount Rate text] (was =AmortDeferredOutsIns!C6)
```

### STEP 2: Set RSI New Year Column Formulas

**Purpose:** The new current year column needs formulas pointing to Net OPEB.

Set formulas in column I for 2025:
```
I3:  =YEAR(Assumptions!C4)
I4:  ='Net OPEB'!D14
I5:  ='Net OPEB'!D16
I6:  ='Net OPEB'!D20
I7:  ='Net OPEB'!D22
I8:  ='Net OPEB'!D18
I9:  ='Net OPEB'!D24
I10: ='Net OPEB'!D27
I12: ='Net OPEB'!D12
I14: ='Net OPEB'!D29
I17: [Covered payroll - hardcode or use formula]
I20: =I14/I17
I26: =AmortDeferredOutsIns!C6
I27: "Pub-2010/2021" (or current mortality table)
I28: "Getzen model" (or current trend model)
```

### STEP 3: Shift ARSL Values in AmortDeferredOutsIns

**Purpose:** ARSL values must follow their years when rows shift.

**CRITICAL:** Capture current values BEFORE making changes, then shift ALL values down:

```python
# Capture current values
c13_val = C13 value  # Current year ARSL (e.g., 5 for 2024)
c14_val = C14 value  # Prior year ARSL (e.g., 6 for 2023)
c15_val = C15 value  # (e.g., 7 for 2022)
c16_val = C16 value  # (e.g., 8 for 2021)
c17_val = C17 value  # (e.g., 9 for 2020)
c18_val = C18 value  # (e.g., 11 for 2019)
# c19_val falls off (oldest year drops out of 7-year window)

# Shift down - each value moves to the next row
C14 = c13_val  # 2024's ARSL (5) moves to row 14
C15 = c14_val  # 2023's ARSL (6) moves to row 15
C16 = c15_val  # 2022's ARSL (7) moves to row 16
C17 = c16_val  # 2021's ARSL (8) moves to row 17
C18 = c17_val  # 2020's ARSL (9) moves to row 18
C19 = c18_val  # 2019's ARSL (11) moves to row 19
# C13 keeps its formula - will get 2025's ARSL
```

**Also shift the Assumption section C23-C29 if they are hardcoded (check first - they may be formulas pointing to C13-C19).**

### STEP 4: Update AmortDeferredOutsIns A13

Change the current year:
```
A13: 2025 (was 2024)
```

The cascade formulas (A14=A13-1, A15=A14-1, etc.) will automatically update:
- A14 → 2024
- A15 → 2023
- A16 → 2022
- etc.

### STEP 5: Update Assumptions Tab

```
C2:  10/1/2024 (Valuation Date)
C3:  9/30/2024 (Prior Measurement Date)
C4:  9/30/2025 (Measurement Date)  ← CRITICAL - drives year calculations
C11: 0.0381 (Prior Discount Rate - was current, now prior)  ← Used for interest calculation
C12: "5.02% annually which is the Bond Buyer 20-Bond General Obligation Index..."
```

### STEP 6: Update ProVal1

**For Roll-Forward Valuation:**
```
B19: 24010 (Prior EOY TOL becomes BOY at old rate)
C19: 21104.79 (BOY at new rate = B19 + Assumption Change)
D19: 22238.67 (EOY TOL = B19 + SC + Interest + Assumption Change)
E19: 20459.58 (EOY at Discount +1%)
F19: 24017.76 (EOY at Discount -1%)
G19: 22238.67 (EOY Trend baseline = D19)
H19: 23128.22 (EOY at Trend +1%)
I19: 21349.12 (EOY at Trend -1%)
B38: 215 (Service Cost - use prior year's for roll-forward)
D88: 0.0502 (New Discount Rate)
```

**Roll-Forward Calculations:**
```
Interest = (BOY + 0.5 * Service Cost) * Prior Discount Rate
         = (24010 + 0.5 * 215) * 0.0381 = 918.88

Assumption Change ≈ -BOY * (New Rate - Prior Rate) * Duration
                  ≈ -24010 * (0.0502 - 0.0381) * 10 = -2905.21

EOY = BOY + Service Cost + Interest + Assumption Change
    = 24010 + 215 + 918.88 + (-2905.21) = 22238.67

Experience = 0 (for roll-forward; formula calculates as residual)
```

**For Full Valuation:**
Enter actual calculated values from the valuation engine/census. The experience will calculate as the residual (D22 = D29 - D24 - D14 - D16 - D12 - D18 - D20).

### STEP 7: Update Net OPEB Labels

```
A9:  "Table 3: Changes in Net OPEB Liability for the plan's fiscal year ending 9/30/2025"
A12: "Balances at 9/30/2024"
A29: "Balances at 9/30/2025"
```

### STEP 8: Save and Verify in Excel

**IMPORTANT:** LibreOffice cannot properly calculate this file's complex formulas. You MUST open in Excel.

1. Save the file
2. Open in Excel (formulas recalculate)
3. Verify Table7AmortDeferred:
   - Row 13: Year=2025, Experience=$0 (for roll-forward) or actual (for full val), ARSL=new
   - Row 14: Year=2024, Experience=$1,832, ARSL=5
   - Row 15: Year=2023, Experience=$1,816, ARSL=6
   - Row 23: Year=2025, Assumptions=$(2,905)
   - Row 24: Year=2024, Assumptions=$104, ARSL=5
4. Verify RSI:
   - Column H: Hardcoded 2024 values
   - Column I: Calculated 2025 values from formulas
5. Verify Net OPEB D22 shows experience (0 for roll-forward, actual for full val)

## Verification Checklist

- [ ] RSI prior year column (H) is hardcoded (no formulas pointing to Net OPEB)
- [ ] RSI current year column (I) has formulas pointing to Net OPEB
- [ ] AmortDeferredOutsIns A13 = 2025
- [ ] AmortDeferredOutsIns C14 = 5 (2024's ARSL, shifted from C13)
- [ ] AmortDeferredOutsIns C15 = 6 (2023's ARSL, shifted from C14)
- [ ] AmortDeferredOutsIns C16 = 7 (2022's ARSL, shifted from C15)
- [ ] AmortDeferredOutsIns C17 = 8 (2021's ARSL, shifted from C16)
- [ ] AmortDeferredOutsIns C18 = 9 (2020's ARSL, shifted from C17)
- [ ] AmortDeferredOutsIns C19 = 11 (2019's ARSL, shifted from C18)
- [ ] Assumptions C4 = 9/30/2025
- [ ] Assumptions C11 = 0.0381 (prior discount rate)
- [ ] ProVal1 B19 = 24010 (BOY)
- [ ] ProVal1 C19 = 21104.79 (BOY at new rate)
- [ ] ProVal1 D19 = 22238.67 (EOY)
- [ ] ProVal1 D88 = 0.0502 (new discount rate)
- [ ] Net OPEB labels show 9/30/2025
- [ ] Table7AmortDeferred Row 14 shows 2024, $1,832 experience, ARSL=5
- [ ] Table7AmortDeferred Row 24 shows 2024, $104 assumptions, ARSL=5

## Common Errors to Avoid

1. **Forgetting to hardcode RSI current year column** → Prior year data lost when Net OPEB updates
2. **Not shifting ALL ARSL values down** → Wrong ARSL for each historical year, wrong amortization
3. **Using wrong discount rate for interest** → Interest uses Assumptions!C11 (PRIOR rate), not current
4. **Not shifting C14-C19** → Historical years have wrong ARSL values
5. **Testing with LibreOffice instead of Excel** → Complex formulas don't calculate properly
