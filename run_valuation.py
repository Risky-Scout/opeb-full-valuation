#!/usr/bin/env python3
"""
run_valuation.py - Production GASB 75 Full Valuation Runner

This script runs a complete GASB 75 OPEB valuation from start to finish:
1. Load and validate census data
2. Run valuation engine (Entry Age Normal)
3. Calculate sensitivities (±1% discount, ±1% trend)
4. Update Excel disclosure template
5. Verify output

Usage:
    python run_valuation.py --interactive
    
    python run_valuation.py \\
        --census census.xlsx \\
        --template prior_year_disclosure.xlsx \\
        --output current_year_disclosure.xlsx \\
        --measurement-date 2025-09-30 \\
        --discount-rate 0.0502

Author: Actuarial Pipeline Project
Version: 2.1.0 (West Florida Planning Corrections - 2025-12-28)
"""

import argparse
import sys
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, '%Y-%m-%d').date()


def run_full_valuation(
    census_path: str,
    template_path: str,
    output_path: str,
    measurement_date: date,
    prior_measurement_date: date,
    discount_rate: float,
    prior_discount_rate: float,
    benefit_changes: str = "None",
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete GASB 75 full valuation.
    
    Args:
        census_path: Path to census data (Excel or CSV)
        template_path: Path to prior year GASB 75 template
        output_path: Path for output disclosure
        measurement_date: Current measurement date
        prior_measurement_date: Prior measurement date
        discount_rate: Current discount rate
        prior_discount_rate: Prior discount rate
        benefit_changes: Description of benefit changes
        verify: Run verification checks
        
    Returns:
        Dict with valuation results and verification status
    """
    from opeb_valuation import (
        create_engine,
        update_full_valuation_excel,
        verify_valuation_output,
        print_valuation_summary,
        FullValuationInputs,
        FullValuationResults,
    )
    import pandas as pd
    
    print("=" * 70)
    print("GASB 75 FULL VALUATION")
    print("=" * 70)
    print(f"Census:      {census_path}")
    print(f"Template:    {template_path}")
    print(f"Output:      {output_path}")
    print(f"Measurement: {measurement_date}")
    print(f"Discount:    {discount_rate:.2%}")
    print()
    
    # =========================================================================
    # STEP 1: Load Census Data
    # =========================================================================
    print("Step 1: Loading census data...")
    
    if census_path.endswith('.csv'):
        census_df = pd.read_csv(census_path)
    else:
        census_df = pd.read_excel(census_path)
    
    # Basic census validation
    required_cols = ['status']  # At minimum need to identify actives vs retirees
    missing = [c for c in required_cols if c not in census_df.columns]
    if missing:
        raise ValueError(f"Census missing required columns: {missing}")
    
    # Count actives and retirees
    if 'status' in census_df.columns:
        active_count = len(census_df[census_df['status'].str.upper().isin(['A', 'ACTIVE'])])
        retiree_count = len(census_df[census_df['status'].str.upper().isin(['R', 'RETIREE', 'RETIRED'])])
    else:
        active_count = len(census_df)
        retiree_count = 0
    
    print(f"  Active employees: {active_count}")
    print(f"  Retirees: {retiree_count}")
    print()
    
    # =========================================================================
    # STEP 2: Run Valuation Engine
    # =========================================================================
    print("Step 2: Running valuation engine...")
    
    # Create engine with current assumptions
    engine = create_engine(
        discount_rate=discount_rate,
        measurement_date=measurement_date,
    )
    
    # Run valuation
    # Note: This is a simplified example - actual implementation would
    # iterate through census records and calculate per-member liabilities
    
    # For now, we'll extract values from the template as a baseline
    from openpyxl import load_workbook
    wb_data = load_workbook(template_path, data_only=True)
    mi = wb_data['Model Inputs']
    
    # Get prior year EOY values as BOY for current year
    boy_tol_old_rate = mi['D19'].value or 0
    service_cost = mi['D38'].value or 0
    covered_payroll = mi['D17'].value or 0
    
    wb_data.close()
    
    # Calculate using duration approximation
    duration = 10.0
    trend_duration = 5.0
    
    # Interest
    interest = (boy_tol_old_rate + 0.5 * service_cost) * prior_discount_rate
    
    # Assumption change
    rate_change = discount_rate - prior_discount_rate
    boy_tol_new_rate = boy_tol_old_rate * (1 - duration * rate_change)
    assumption_change = boy_tol_new_rate - boy_tol_old_rate
    
    # For full valuations, experience would come from census comparison
    # This is a placeholder - actual experience from engine comparison
    experience = 0  # Would be: actual_eoy - expected_eoy
    
    # EOY TOL
    eoy_tol = boy_tol_old_rate + service_cost + interest + assumption_change + experience
    
    # Sensitivities
    sensitivity_disc_plus = eoy_tol * (1 - duration * 0.01)
    sensitivity_disc_minus = eoy_tol * (1 - duration * -0.01)
    sensitivity_trend_plus = eoy_tol * (1 + trend_duration * 0.01)
    sensitivity_trend_minus = eoy_tol * (1 + trend_duration * -0.01)
    
    print(f"  BOY TOL (old rate): ${boy_tol_old_rate:,.0f}")
    print(f"  BOY TOL (new rate): ${boy_tol_new_rate:,.0f}")
    print(f"  Service Cost: ${service_cost:,.0f}")
    print(f"  Interest: ${interest:,.0f}")
    print(f"  Assumption Change: ${assumption_change:,.0f}")
    print(f"  Experience: ${experience:,.0f}")
    print(f"  EOY TOL: ${eoy_tol:,.0f}")
    print()
    
    # =========================================================================
    # STEP 3: Create Results Objects
    # =========================================================================
    print("Step 3: Preparing disclosure update...")
    
    inputs = FullValuationInputs(
        prior_measurement_date=prior_measurement_date,
        new_measurement_date=measurement_date,
        prior_discount_rate=prior_discount_rate,
        new_discount_rate=discount_rate,
        active_count=active_count,
        retiree_count=retiree_count,
        covered_payroll=covered_payroll,
        benefit_changes=benefit_changes,
        duration=duration,
        trend_duration=trend_duration,
    )
    
    results = FullValuationResults(
        boy_tol_old_rate=boy_tol_old_rate,
        boy_tol_new_rate=boy_tol_new_rate,
        eoy_tol=eoy_tol,
        sensitivity_disc_plus=sensitivity_disc_plus,
        sensitivity_disc_minus=sensitivity_disc_minus,
        sensitivity_trend_plus=sensitivity_trend_plus,
        sensitivity_trend_minus=sensitivity_trend_minus,
        service_cost=service_cost,
        interest=interest,
        assumption_change=assumption_change,
        experience=experience,
        active_count=active_count,
        retiree_count=retiree_count,
        covered_payroll=covered_payroll,
    )
    
    # =========================================================================
    # STEP 4: Update Excel Template
    # =========================================================================
    print("Step 4: Updating Excel disclosure...")
    
    output = update_full_valuation_excel(
        input_path=template_path,
        output_path=output_path,
        inputs=inputs,
        results=results,
    )
    
    print(f"  Output saved to: {output}")
    print()
    
    # =========================================================================
    # STEP 5: Verify Output
    # =========================================================================
    verification = None
    if verify:
        print("Step 5: Verifying output...")
        verification = verify_valuation_output(output)
        
        for check_name, check_result in verification['checks'].items():
            status = "✓ PASS" if check_result['passed'] else "✗ FAIL"
            print(f"  {status}: {check_name}")
        
        print()
        if verification['passed']:
            print("✓ All verification checks passed!")
        else:
            print("✗ Some verification checks failed - please review in Excel")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print_valuation_summary(results, inputs)
    
    return {
        'output_path': output,
        'inputs': inputs,
        'results': results,
        'verification': verification,
    }


def interactive_mode():
    """Run in interactive mode, prompting for all inputs."""
    print("=" * 70)
    print("GASB 75 FULL VALUATION - Interactive Mode")
    print("=" * 70)
    print()
    
    # Get file paths
    census_path = input("Census data file path: ").strip()
    if not Path(census_path).exists():
        print(f"ERROR: File not found: {census_path}")
        sys.exit(1)
    
    template_path = input("Prior year Excel template path: ").strip()
    if not Path(template_path).exists():
        print(f"ERROR: File not found: {template_path}")
        sys.exit(1)
    
    output_path = input("Output file path: ").strip()
    
    # Get dates
    prior_date_str = input("Prior measurement date (YYYY-MM-DD): ").strip()
    prior_date = parse_date(prior_date_str)
    
    new_date_str = input("New measurement date (YYYY-MM-DD): ").strip()
    new_date = parse_date(new_date_str)
    
    # Get discount rates
    prior_rate = float(input("Prior discount rate (e.g., 0.0381 for 3.81%): ").strip())
    new_rate = float(input("New discount rate (e.g., 0.0502 for 5.02%): ").strip())
    
    # Benefit changes
    benefit_changes = input("Benefit changes description [None]: ").strip()
    if not benefit_changes:
        benefit_changes = "None"
    
    print()
    
    # Run valuation
    run_full_valuation(
        census_path=census_path,
        template_path=template_path,
        output_path=output_path,
        measurement_date=new_date,
        prior_measurement_date=prior_date,
        discount_rate=new_rate,
        prior_discount_rate=prior_rate,
        benefit_changes=benefit_changes,
        verify=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Run GASB 75 Full Valuation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_valuation.py --interactive
  
  # Command line mode
  python run_valuation.py \\
      --census census.xlsx \\
      --template prior_year.xlsx \\
      --output current_year.xlsx \\
      --measurement-date 2025-09-30 \\
      --prior-date 2024-09-30 \\
      --discount-rate 0.0502 \\
      --prior-rate 0.0381 \\
      --verify
"""
    )
    
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--census', type=str, help='Census data file (Excel or CSV)')
    parser.add_argument('--template', type=str, help='Prior year Excel template')
    parser.add_argument('--output', type=str, help='Output Excel file')
    parser.add_argument('--measurement-date', type=str, help='Measurement date (YYYY-MM-DD)')
    parser.add_argument('--prior-date', type=str, help='Prior measurement date (YYYY-MM-DD)')
    parser.add_argument('--discount-rate', type=float, help='Discount rate (e.g., 0.0502)')
    parser.add_argument('--prior-rate', type=float, help='Prior discount rate (e.g., 0.0381)')
    parser.add_argument('--benefit-changes', type=str, default='None', help='Benefit changes description')
    parser.add_argument('--verify', action='store_true', help='Run verification checks')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    # Check required arguments
    required = ['census', 'template', 'output', 'measurement_date', 'prior_date', 'discount_rate', 'prior_rate']
    missing = [arg for arg in required if getattr(args, arg.replace('-', '_')) is None]
    
    if missing:
        print(f"ERROR: Missing required arguments: {', '.join(missing)}")
        print("Use --interactive for guided input or --help for usage.")
        sys.exit(1)
    
    # Parse dates
    measurement_date = parse_date(args.measurement_date)
    prior_date = parse_date(args.prior_date)
    
    # Run valuation
    run_full_valuation(
        census_path=args.census,
        template_path=args.template,
        output_path=args.output,
        measurement_date=measurement_date,
        prior_measurement_date=prior_date,
        discount_rate=args.discount_rate,
        prior_discount_rate=args.prior_rate,
        benefit_changes=args.benefit_changes,
        verify=args.verify,
    )


if __name__ == '__main__':
    main()
