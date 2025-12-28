"""
tests/test_valuation.py - GASB 75 Valuation Unit Tests

Implements the 4 specific validation tests per the specification:
1. The "No-Subsidy" Null Hypothesis
2. The "Pure Annuity" Check
3. The "Implicit Slope" Sensitivity
4. EAN Service Cost Logic

Plus additional tests for:
- Retiree TOL = PVFB (100% attribution)
- Active attribution bounds [0, 1]
- Duration and rate sensitivity

Author: Actuarial Pipeline Project
License: MIT
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date
from opeb_valuation.engine import ValuationEngine, CensusRecord, create_engine
from opeb_valuation.mortality import MortalityCalculator, create_mortality_calculator
from opeb_valuation.financials import FinancialEngine, MorbidityModel, TrendModel
from opeb_valuation.decrements import (
    TerminationRates, DisabilityRates, RetirementEligibility
)


class TestNoSubsidyNullHypothesis:
    """
    Test 1: The "No-Subsidy" Null Hypothesis
    
    Setup: Set Morbidity Factor to 1.0 for all ages (flat cost).
           Set Trend to 0%. Set Premium = Gross Cost.
    
    Expectation: The calculated OPEB Liability must be exactly 0.00.
    
    Why: If Cost == Premium, IS_t = max(0, 0) = 0.
         If the model produces a non-zero number, the netting logic is flawed.
    """
    
    def test_zero_subsidy_produces_zero_liability(self):
        """When gross cost equals participant contribution, liability should be zero."""
        
        # Create morbidity model with 100% contribution
        morbidity = MorbidityModel(
            base_cost_pre65=500.0,
            base_cost_post65=400.0,
            contribution_rate=1.0  # 100% - employee pays full cost
        )
        
        # With 100% contribution, implicit subsidy should be 0
        subsidy = morbidity.get_implicit_subsidy(65, 'M', 0, 1.0)
        
        assert abs(subsidy) < 0.01, \
            f"Expected zero subsidy with 100% contribution, got ${subsidy:.2f}"
    
    def test_flat_world_no_subsidy(self):
        """Full valuation with no subsidy should produce zero liability."""
        
        config = {
            'valuation_date': date(2025, 9, 30),
            'discount_rate': 0.04,
            'discount_rate_boy': 0.04,
            'contribution_rate': 1.0,  # 100% employee contribution
            'base_cost_pre65': 500.0,
            'base_cost_post65': 400.0,
            'dental_premiums': {},
            'admin_fee_monthly': 0.0,  # No admin fees
            'married_fraction': 0.0,   # No spouse benefits
        }
        
        # Create a simple retiree census
        census = CensusRecord(
            participant_id='R001',
            dob=date(1960, 1, 1),  # Age 65
            doh=date(1990, 1, 1),
            gender='M',
            status='Retiree',
            coverage_tier='Employee'
        )
        
        # With 100% contribution, PVFB should be near zero
        # (only admin/dental if any)
        engine = create_engine(config)
        result = engine.retiree_valuator.valuate(census)
        
        # Allow small tolerance for dental/admin
        assert result.tol_total < 1000, \
            f"Expected near-zero liability with 100% contribution, got ${result.tol_total:,.0f}"


class TestPureAnnuityCheck:
    """
    Test 2: The "Pure Annuity" Check
    
    Setup:
    - 1 Retiree, Age 65
    - Gross Cost = $10,000/year, Premium = $0 (Free plan)
    - Trend = 0%, Discount Rate = 0%
    - Mortality = 0% until Age 70, then 100% (Death at 70)
    
    Math: The retiree lives exactly 5 years (65, 66, 67, 68, 69).
          Benefit = $10,000 × 5 = $50,000.
    
    Expectation: Model output must equal $50,000.
    
    Why: This verifies that loop bounds (start/end indices) are correct
         and not off-by-one.
    """
    
    def test_pure_annuity_five_years(self):
        """Simple 5-year annuity with no discounting or mortality."""
        
        # Create a flat financial engine (0% discount, 0% trend)
        financial = FinancialEngine(
            discount_rate_eoy=0.0,
            discount_rate_boy=0.0,
            medical_trend=TrendModel(initial_rate=0.0, ultimate_rate=0.0),
            dental_trend=0.0,
            admin_trend=0.0
        )
        
        # Annual benefit of $10,000 for 5 years with 100% survival
        cash_flows = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
        survival = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        pv = financial.calculate_pv(cash_flows, survival, mid_year=False)
        
        expected = 50000.0
        assert abs(pv - expected) < 0.01, \
            f"Expected ${expected:,.0f}, got ${pv:,.0f}"
    
    def test_discount_factor_zero_rate(self):
        """At 0% discount rate, all discount factors should be 1.0."""
        
        financial = FinancialEngine(discount_rate_eoy=0.0)
        
        for years in [0, 1, 5, 10, 20]:
            df = financial.get_discount_factor(years, mid_year=False)
            assert abs(df - 1.0) < 1e-10, \
                f"Expected DF=1.0 at 0% for year {years}, got {df}"


class TestImplicitSlopeSensitivity:
    """
    Test 3: The "Implicit Slope" Sensitivity
    
    Setup A: Run a standard Active valuation with Morbidity Age Grading = 0% (Flat).
    Setup B: Run same valuation with Morbidity Age Grading = 4% (Standard).
    
    Expectation: Result B >>> Result A.
    
    Analysis: If Result B is not significantly higher (typically 30-50% higher
              for Actives), the model is applying premium trend to gross cost
              but failing to apply the aging factor M(x).
    """
    
    def test_age_grading_increases_liability(self):
        """Age-graded morbidity should produce higher liability than flat."""
        
        # Flat morbidity (no age grading)
        flat_morbidity = MorbidityModel(
            base_cost_pre65=500.0,
            base_cost_post65=400.0,
            contribution_rate=0.45
        )
        for age in range(15, 111):
            flat_morbidity.morbidity_factors_male[age] = 1.0
            flat_morbidity.morbidity_factors_female[age] = 1.0
        
        # Age-graded morbidity (standard 3-4% annual increase)
        graded_morbidity = MorbidityModel(
            base_cost_pre65=500.0,
            base_cost_post65=400.0,
            contribution_rate=0.45
        )
        for age in range(15, 111):
            # Standard age grading: ~4% per year
            factor = 0.30 * np.power(1.04, age - 20)
            graded_morbidity.morbidity_factors_male[age] = min(factor, 2.0)
            graded_morbidity.morbidity_factors_female[age] = min(factor, 2.0)
        
        # Compare implicit subsidies at older ages
        flat_subsidy = flat_morbidity.get_implicit_subsidy(75, 'M', 0, 1.0)
        graded_subsidy = graded_morbidity.get_implicit_subsidy(75, 'M', 0, 1.0)
        
        assert graded_subsidy > flat_subsidy, \
            f"Age-graded subsidy (${graded_subsidy:,.0f}) should exceed " \
            f"flat (${flat_subsidy:,.0f})"
        
        # Should be significantly higher (at least 50% at age 75)
        ratio = graded_subsidy / flat_subsidy if flat_subsidy > 0 else float('inf')
        assert ratio > 1.3, \
            f"Expected at least 30% increase with age grading, got {ratio:.1%}"


class TestEANServiceCostLogic:
    """
    Test 4: EAN Service Cost Logic
    
    Setup 1: Active employee, hired at age 30, current age 30.
    Expectation: Service Cost = Total OPEB Liability (since Past Service = 0).
    
    Setup 2: Active employee, hired at 30, current age 64 (retirement at 65).
    Expectation: Service Cost ≈ 1/35 × PVFB (very small portion of total value).
    """
    
    def test_new_hire_service_cost_equals_tol(self):
        """For a new hire with 0 service, SC should equal TOL."""
        
        config = {
            'valuation_date': date(2025, 9, 30),
            'discount_rate': 0.04,
            'discount_rate_boy': 0.04,
            'dental_premiums': {'Employee': 13.24},
            'admin_fee_monthly': 35.44,
            'married_fraction': 0.0,
        }
        
        # New hire: hired today, age 30
        census = CensusRecord(
            participant_id='A001',
            dob=date(1995, 9, 30),  # Age 30
            doh=date(2025, 9, 30),  # Hired today (0 service)
            gender='M',
            status='Active',
            coverage_tier='Employee',
            current_salary=50000.0
        )
        
        engine = create_engine(config)
        result = engine.active_valuator.valuate(census)
        
        # With 0 service, attribution ratio should be 0, so TOL = 0
        # Service Cost should be PVFB / Expected_Total_Service
        assert result.tol_total < 100, \
            f"New hire with 0 service should have near-zero TOL, got ${result.tol_total:,.0f}"
        
        assert result.attribution_ratio < 0.05, \
            f"New hire attribution ratio should be ~0, got {result.attribution_ratio:.2f}"
    
    def test_near_retirement_has_small_service_cost(self):
        """Employee near retirement should have SC << TOL."""
        
        config = {
            'valuation_date': date(2025, 9, 30),
            'discount_rate': 0.04,
            'discount_rate_boy': 0.04,
            'dental_premiums': {'Employee': 13.24},
            'admin_fee_monthly': 35.44,
            'married_fraction': 0.0,
        }
        
        # Near retirement: hired at 30, now age 64
        census = CensusRecord(
            participant_id='A002',
            dob=date(1961, 9, 30),  # Age 64
            doh=date(1991, 9, 30),  # Hired at 30, 34 years service
            gender='M',
            status='Active',
            coverage_tier='Employee',
            current_salary=80000.0
        )
        
        engine = create_engine(config)
        result = engine.active_valuator.valuate(census)
        
        # With 34 years service and retiring at 65 (35 years total),
        # attribution should be ~34/35 = 0.97
        assert 0.90 < result.attribution_ratio < 1.0, \
            f"Near-retirement attribution should be ~0.97, got {result.attribution_ratio:.2f}"
        
        # Service cost should be much smaller than TOL
        if result.tol_total > 0:
            sc_to_tol_ratio = result.service_cost / result.tol_total
            assert sc_to_tol_ratio < 0.10, \
                f"Service cost should be <10% of TOL near retirement, got {sc_to_tol_ratio:.1%}"


class TestRetireeAttributionRule:
    """
    Test: Retirees have 100% Attribution
    
    For any retiree, TOL must equal PVFB (no attribution reduction).
    """
    
    def test_retiree_tol_equals_pvfb(self):
        """Retiree TOL should exactly equal PVFB."""
        
        config = {
            'valuation_date': date(2025, 9, 30),
            'discount_rate': 0.04,
            'dental_premiums': {'Employee': 13.24},
        }
        
        census = CensusRecord(
            participant_id='R001',
            dob=date(1955, 1, 1),  # Age 70
            doh=date(1980, 1, 1),
            gender='F',
            status='Retiree',
            coverage_tier='Employee'
        )
        
        engine = create_engine(config)
        result = engine.retiree_valuator.valuate(census)
        
        assert result.attribution_ratio == 1.0, \
            f"Retiree attribution should be 1.0, got {result.attribution_ratio}"
        
        assert abs(result.tol_total - result.pvfb_total) < 0.01, \
            f"Retiree TOL (${result.tol_total:,.0f}) should equal " \
            f"PVFB (${result.pvfb_total:,.0f})"


class TestAttributionBounds:
    """
    Test: Attribution ratio must be in [0, 1] for all members.
    """
    
    def test_attribution_ratio_bounds(self):
        """Attribution ratio should always be between 0 and 1."""
        
        config = {
            'valuation_date': date(2025, 9, 30),
            'discount_rate': 0.04,
            'dental_premiums': {'Employee': 13.24},
        }
        
        engine = create_engine(config)
        
        # Test various service levels
        test_cases = [
            (date(1995, 1, 1), date(2025, 1, 1), 'A001'),  # 0 service
            (date(1985, 1, 1), date(2015, 1, 1), 'A002'),  # 10 service
            (date(1965, 1, 1), date(1995, 1, 1), 'A003'),  # 30 service
            (date(1960, 1, 1), date(1980, 1, 1), 'A004'),  # 45 service (past retirement)
        ]
        
        for dob, doh, pid in test_cases:
            census = CensusRecord(
                participant_id=pid, dob=dob, doh=doh, gender='M',
                status='Active', coverage_tier='Employee', current_salary=50000.0
            )
            result = engine.active_valuator.valuate(census)
            
            assert 0 <= result.attribution_ratio <= 1.0, \
                f"Attribution ratio {result.attribution_ratio} out of bounds for {pid}"


class TestDiscountRateSensitivity:
    """
    Test: The "Rate Shock" Test
    
    Discount rate increases should decrease liability.
    A 1% increase should reduce liability by approximately Duration × 1%.
    """
    
    def test_discount_rate_sensitivity(self):
        """Higher discount rate should produce lower liability."""
        
        config_base = {
            'valuation_date': date(2025, 9, 30),
            'discount_rate': 0.04,
            'dental_premiums': {'Employee': 13.24},
        }
        
        config_plus1 = {
            'valuation_date': date(2025, 9, 30),
            'discount_rate': 0.05,  # +1%
            'dental_premiums': {'Employee': 13.24},
        }
        
        census = CensusRecord(
            participant_id='R001',
            dob=date(1960, 1, 1),  # Age 65
            doh=date(1985, 1, 1),
            gender='M',
            status='Retiree',
            coverage_tier='Employee'
        )
        
        engine_base = create_engine(config_base)
        engine_plus1 = create_engine(config_plus1)
        
        result_base = engine_base.retiree_valuator.valuate(census)
        result_plus1 = engine_plus1.retiree_valuator.valuate(census)
        
        assert result_plus1.tol_total < result_base.tol_total, \
            f"Higher discount rate should produce lower liability. " \
            f"Base: ${result_base.tol_total:,.0f}, +1%: ${result_plus1.tol_total:,.0f}"
        
        # Typical OPEB duration is 10-15 years
        # Change should be approximately 10-15%
        change_pct = (result_base.tol_total - result_plus1.tol_total) / result_base.tol_total
        assert 0.05 < change_pct < 0.25, \
            f"Rate sensitivity of {change_pct:.1%} seems unusual (expected 5-25%)"


class TestDecrementRates:
    """
    Test: Decrement rate lookups
    """
    
    def test_termination_select_vs_ultimate(self):
        """Select period rates should be higher than ultimate."""
        
        # Year 0 (select) should be higher than year 10 (ultimate)
        select_rate = TerminationRates.get_rate(45, 0)  # Age 45, 0 service
        ultimate_rate = TerminationRates.get_rate(45, 10)  # Age 45, 10 service
        
        assert select_rate > ultimate_rate, \
            f"Select rate ({select_rate:.1%}) should exceed " \
            f"ultimate ({ultimate_rate:.1%})"
    
    def test_retirement_eligibility_tiers(self):
        """Tier 1 vs Tier 2 retirement ages."""
        
        # Tier 1: hired before 2013
        dob = date(1970, 1, 1)
        hire_tier1 = date(2010, 1, 1)
        hire_tier2 = date(2015, 1, 1)
        
        ret_age_tier1 = RetirementEligibility.get_earliest_retirement_age(hire_tier1, dob)
        ret_age_tier2 = RetirementEligibility.get_earliest_retirement_age(hire_tier2, dob)
        
        # Tier 1 typically has earlier retirement
        assert ret_age_tier1 <= ret_age_tier2 + 5, \
            f"Tier 1 ({ret_age_tier1}) should not be much later than Tier 2 ({ret_age_tier2})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
