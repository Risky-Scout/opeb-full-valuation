#!/usr/bin/env python3
"""
GASB 75 OPEB Disclosure Excel File Updater
==========================================

This module contains the correct logic for updating a GASB 75 disclosure file
from one measurement period to the next.

The update process is CRITICAL and must follow this exact sequence:
1. Hardcode RSI current year column (preserve prior year data)
2. Set RSI new year column formulas
3. Shift ARSL values in AmortDeferredOutsIns (ALL values shift down)
4. Update AmortDeferredOutsIns A13 (current year)
5. Update Assumptions tab (dates and rates)
6. Update ProVal1 (valuation results)
7. Update Net OPEB labels

This works for both roll-forward and full valuations - only the ProVal1
values differ.
"""

from openpyxl import load_workbook
from datetime import date
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GASB75UpdateInputs:
    """All inputs needed to update a GASB 75 file."""
    # Dates
    valuation_date: date
    prior_measurement_date: date
    measurement_date: date
    
    # Discount rates
    prior_discount_rate: float  # Rate at prior measurement date (for interest calc)
    new_discount_rate: float    # Rate at new measurement date
    
    # ProVal1 values - TOL
    tol_boy_old_rate: float     # B19 - BOY TOL at prior EOY rate
    tol_boy_new_rate: float     # C19 - BOY TOL revalued at new BOY rate
    tol_eoy_baseline: float     # D19 - EOY TOL at new rate
    tol_eoy_disc_plus_1: float  # E19 - EOY at discount +1%
    tol_eoy_disc_minus_1: float # F19 - EOY at discount -1%
    tol_eoy_trend_baseline: float  # G19 - EOY trend baseline
    tol_eoy_trend_plus_1: float    # H19 - EOY at trend +1%
    tol_eoy_trend_minus_1: float   # I19 - EOY at trend -1%
    
    # ProVal1 values - Other
    service_cost: float         # B38 - Service cost for the year
    covered_payroll: float      # D17 - Covered payroll (optional update)
    active_count: Optional[int] = None   # D6
    retiree_count: Optional[int] = None  # D8


def update_gasb75_file(
    input_file: str,
    output_file: str,
    inputs: GASB75UpdateInputs,
    new_arsl: Optional[float] = None
) -> Dict[str, Any]:
    """
    Update a GASB 75 disclosure file from one measurement period to the next.
    
    Args:
        input_file: Path to the current year's GASB 75 file
        output_file: Path for the updated file
        inputs: GASB75UpdateInputs with all required values
        new_arsl: New ARSL for current year (if None, formula will calculate)
    
    Returns:
        Dict with summary of changes made
    """
    
    # Load workbooks - one for formulas, one for values
    wb = load_workbook(input_file)
    wb_data = load_workbook(input_file, data_only=True)
    
    summary = {
        'prior_year': None,
        'new_year': None,
        'rsi_hardcoded': {},
        'arsl_shifted': {},
        'proval_updated': {},
    }
    
    # ==========================================================================
    # STEP 1: HARDCODE RSI CURRENT YEAR COLUMN
    # ==========================================================================
    rsi = wb['RSI']
    rsi_data = wb_data['RSI']
    
    # Determine current year column (H for 2024, I for 2025, etc.)
    # The current year is in Assumptions C4
    assumptions_data = wb_data['Assumptions']
    current_measurement_date = assumptions_data['C4'].value
    if hasattr(current_measurement_date, 'year'):
        current_year = current_measurement_date.year
    else:
        current_year = int(current_measurement_date) if current_measurement_date else 2024
    
    summary['prior_year'] = current_year
    summary['new_year'] = inputs.measurement_date.year
    
    # Map year to column: 2018=B, 2019=C, 2020=D, 2021=E, 2022=F, 2023=G, 2024=H, 2025=I
    year_to_col = {2018: 'B', 2019: 'C', 2020: 'D', 2021: 'E', 2022: 'F', 2023: 'G', 2024: 'H', 2025: 'I', 2026: 'J', 2027: 'K'}
    current_col = year_to_col.get(current_year, 'H')
    new_col = year_to_col.get(inputs.measurement_date.year, 'I')
    
    # Rows to hardcode in RSI
    rsi_rows = {
        3: 'Year',
        4: 'Service Cost',
        5: 'Interest',
        6: 'Benefit Changes',
        7: 'Experience',
        8: 'Assumptions',
        9: 'Benefit Payments',
        10: 'Net Change',
        12: 'BOY TOL',
        14: 'EOY TOL',
        17: 'Covered Payroll',
        20: 'TOL % of Payroll',
        26: 'Discount Rate',
    }
    
    # Hardcode current column values
    for row, label in rsi_rows.items():
        cell_ref = f'{current_col}{row}'
        current_value = rsi_data[cell_ref].value
        rsi[cell_ref] = current_value
        summary['rsi_hardcoded'][f'{cell_ref} ({label})'] = current_value
    
    # ==========================================================================
    # STEP 2: SET RSI NEW YEAR COLUMN FORMULAS
    # ==========================================================================
    rsi[f'{new_col}3'] = f"=YEAR(Assumptions!C4)"
    rsi[f'{new_col}4'] = f"='Net OPEB'!D14"
    rsi[f'{new_col}5'] = f"='Net OPEB'!D16"
    rsi[f'{new_col}6'] = f"='Net OPEB'!D20"
    rsi[f'{new_col}7'] = f"='Net OPEB'!D22"
    rsi[f'{new_col}8'] = f"='Net OPEB'!D18"
    rsi[f'{new_col}9'] = f"='Net OPEB'!D24"
    rsi[f'{new_col}10'] = f"='Net OPEB'!D27"
    rsi[f'{new_col}12'] = f"='Net OPEB'!D12"
    rsi[f'{new_col}14'] = f"='Net OPEB'!D29"
    rsi[f'{new_col}17'] = inputs.covered_payroll  # Hardcode or use formula
    rsi[f'{new_col}20'] = f"={new_col}14/{new_col}17"
    rsi[f'{new_col}26'] = f"=AmortDeferredOutsIns!C6"
    rsi[f'{new_col}27'] = "Pub-2010/2021"
    rsi[f'{new_col}28'] = "Getzen model"
    
    # ==========================================================================
    # STEP 3: SHIFT ARSL VALUES IN AmortDeferredOutsIns
    # ==========================================================================
    amort = wb['AmortDeferredOutsIns']
    amort_data = wb_data['AmortDeferredOutsIns']
    
    # Capture current ARSL values BEFORE shifting
    c13_val = amort_data['C13'].value  # Current year ARSL
    c14_val = amort_data['C14'].value  # Prior year 1
    c15_val = amort_data['C15'].value  # Prior year 2
    c16_val = amort_data['C16'].value  # Prior year 3
    c17_val = amort_data['C17'].value  # Prior year 4
    c18_val = amort_data['C18'].value  # Prior year 5
    # c19 falls off
    
    # Shift ALL values down by one row
    amort['C14'] = c13_val  # Current year's ARSL moves to row 14
    amort['C15'] = c14_val  # Prior year 1 moves to row 15
    amort['C16'] = c15_val  # Prior year 2 moves to row 16
    amort['C17'] = c16_val  # Prior year 3 moves to row 17
    amort['C18'] = c17_val  # Prior year 4 moves to row 18
    amort['C19'] = c18_val  # Prior year 5 moves to row 19
    # C13 keeps its formula - will calculate new year's ARSL
    
    summary['arsl_shifted'] = {
        'C14 (was C13)': c13_val,
        'C15 (was C14)': c14_val,
        'C16 (was C15)': c15_val,
        'C17 (was C16)': c16_val,
        'C18 (was C17)': c17_val,
        'C19 (was C18)': c18_val,
    }
    
    # ==========================================================================
    # STEP 4: UPDATE AmortDeferredOutsIns A13 (current year)
    # ==========================================================================
    amort['A13'] = inputs.measurement_date.year
    
    # ==========================================================================
    # STEP 5: UPDATE ASSUMPTIONS TAB
    # ==========================================================================
    assumptions = wb['Assumptions']
    assumptions['C2'] = inputs.valuation_date
    assumptions['C3'] = inputs.prior_measurement_date
    assumptions['C4'] = inputs.measurement_date
    assumptions['C11'] = inputs.prior_discount_rate
    assumptions['C12'] = f"{inputs.new_discount_rate*100:.2f}% annually which is the Bond Buyer 20-Bond General Obligation Index on the Measurement Date.  The 20-Bond Index consists of 20 general obligation bonds that mature in 20 years."
    
    # ==========================================================================
    # STEP 6: UPDATE PROVAL1
    # ==========================================================================
    proval = wb['ProVal1']
    
    proval['B19'] = inputs.tol_boy_old_rate
    proval['C19'] = inputs.tol_boy_new_rate
    proval['D19'] = inputs.tol_eoy_baseline
    proval['E19'] = inputs.tol_eoy_disc_plus_1
    proval['F19'] = inputs.tol_eoy_disc_minus_1
    proval['G19'] = inputs.tol_eoy_trend_baseline
    proval['H19'] = inputs.tol_eoy_trend_plus_1
    proval['I19'] = inputs.tol_eoy_trend_minus_1
    proval['B38'] = inputs.service_cost
    proval['D88'] = inputs.new_discount_rate
    
    if inputs.active_count is not None:
        proval['D6'] = inputs.active_count
    if inputs.retiree_count is not None:
        proval['D8'] = inputs.retiree_count
    if inputs.covered_payroll:
        proval['D17'] = inputs.covered_payroll
    
    summary['proval_updated'] = {
        'B19 (BOY old rate)': inputs.tol_boy_old_rate,
        'C19 (BOY new rate)': inputs.tol_boy_new_rate,
        'D19 (EOY)': inputs.tol_eoy_baseline,
        'B38 (Service Cost)': inputs.service_cost,
        'D88 (Discount Rate)': inputs.new_discount_rate,
    }
    
    # ==========================================================================
    # STEP 7: UPDATE NET OPEB LABELS
    # ==========================================================================
    net_opeb = wb['Net OPEB']
    net_opeb['A9'] = f"Table 3: Changes in Net OPEB Liability for the plan's fiscal year ending {inputs.measurement_date.month}/{inputs.measurement_date.day}/{inputs.measurement_date.year}"
    net_opeb['A12'] = f"Balances at {inputs.prior_measurement_date.month}/{inputs.prior_measurement_date.day}/{inputs.prior_measurement_date.year}"
    net_opeb['A29'] = f"Balances at {inputs.measurement_date.month}/{inputs.measurement_date.day}/{inputs.measurement_date.year}"
    
    # ==========================================================================
    # SAVE
    # ==========================================================================
    wb.save(output_file)
    wb.close()
    wb_data.close()
    
    return summary


def run_rollforward_calculation(
    prior_tol_eoy: float,
    prior_service_cost: float,
    prior_discount_rate: float,
    new_discount_rate: float,
    duration: float = 10.0
) -> Dict[str, float]:
    """
    Calculate roll-forward values.
    
    For a roll-forward, we project the liability forward assuming:
    - Service cost same as prior year
    - Interest at prior discount rate
    - Assumption change from discount rate change
    - Experience = 0 (no actual data)
    
    Returns dict with all calculated values.
    """
    tol_boy = prior_tol_eoy
    service_cost = prior_service_cost
    
    # Interest at prior (BOY) rate
    interest = (tol_boy + 0.5 * service_cost) * prior_discount_rate
    
    # Assumption change from rate change
    rate_change = new_discount_rate - prior_discount_rate
    assumption_change = -tol_boy * rate_change * duration
    
    # BOY at new rate (for D18 formula)
    tol_boy_new_rate = tol_boy + assumption_change
    
    # EOY = BOY + SC + Interest + Assumption + Experience(0) - Benefits(0)
    tol_eoy = tol_boy + service_cost + interest + assumption_change
    
    # Sensitivities (rough approximations)
    tol_eoy_disc_plus_1 = tol_eoy * 0.92
    tol_eoy_disc_minus_1 = tol_eoy * 1.08
    tol_eoy_trend_plus_1 = tol_eoy * 1.04
    tol_eoy_trend_minus_1 = tol_eoy * 0.96
    
    return {
        'tol_boy_old_rate': tol_boy,
        'tol_boy_new_rate': tol_boy_new_rate,
        'tol_eoy_baseline': tol_eoy,
        'tol_eoy_disc_plus_1': tol_eoy_disc_plus_1,
        'tol_eoy_disc_minus_1': tol_eoy_disc_minus_1,
        'tol_eoy_trend_baseline': tol_eoy,
        'tol_eoy_trend_plus_1': tol_eoy_trend_plus_1,
        'tol_eoy_trend_minus_1': tol_eoy_trend_minus_1,
        'service_cost': service_cost,
        'interest': interest,
        'assumption_change': assumption_change,
        'experience': 0.0,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Example: Update West Florida Planning from 2024 to 2025
    
    # Prior year values (from 2024 file)
    prior_tol_eoy = 24010
    prior_service_cost = 215
    prior_discount_rate = 0.0381
    new_discount_rate = 0.0502
    
    # Calculate roll-forward values
    calc = run_rollforward_calculation(
        prior_tol_eoy=prior_tol_eoy,
        prior_service_cost=prior_service_cost,
        prior_discount_rate=prior_discount_rate,
        new_discount_rate=new_discount_rate
    )
    
    print("Roll-Forward Calculation:")
    print(f"  BOY TOL (old rate): ${calc['tol_boy_old_rate']:,.0f}")
    print(f"  BOY TOL (new rate): ${calc['tol_boy_new_rate']:,.2f}")
    print(f"  Service Cost: ${calc['service_cost']:,.0f}")
    print(f"  Interest: ${calc['interest']:,.2f}")
    print(f"  Assumption Change: ${calc['assumption_change']:,.2f}")
    print(f"  Experience: ${calc['experience']:,.2f}")
    print(f"  EOY TOL: ${calc['tol_eoy_baseline']:,.2f}")
    
    # Create inputs
    inputs = GASB75UpdateInputs(
        valuation_date=date(2024, 10, 1),
        prior_measurement_date=date(2024, 9, 30),
        measurement_date=date(2025, 9, 30),
        prior_discount_rate=prior_discount_rate,
        new_discount_rate=new_discount_rate,
        tol_boy_old_rate=calc['tol_boy_old_rate'],
        tol_boy_new_rate=calc['tol_boy_new_rate'],
        tol_eoy_baseline=calc['tol_eoy_baseline'],
        tol_eoy_disc_plus_1=calc['tol_eoy_disc_plus_1'],
        tol_eoy_disc_minus_1=calc['tol_eoy_disc_minus_1'],
        tol_eoy_trend_baseline=calc['tol_eoy_trend_baseline'],
        tol_eoy_trend_plus_1=calc['tol_eoy_trend_plus_1'],
        tol_eoy_trend_minus_1=calc['tol_eoy_trend_minus_1'],
        service_cost=calc['service_cost'],
        covered_payroll=1858084,
    )
    
    # Update the file
    # summary = update_gasb75_file(
    #     input_file='GASB75_2024.xlsx',
    #     output_file='GASB75_2025.xlsx',
    #     inputs=inputs
    # )
    # print(f"\nUpdate Summary: {summary}")
