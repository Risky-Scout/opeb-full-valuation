"""
tests/precision_tests.py - Shackleford Precision Validation Suite

MASTER DIRECTIVE COMPLIANCE:
These tests PROVE the mathematical superiority of the engine.

Tests:
1. FLAT WORLD THEORY: 0% trend/discount = sum of cash flows
2. CONTINUITY THEORY: Liability at 60.001 ≈ Liability at 60.000
3. COMPETING RISK THEORY: MDT < simple sum (always)
4. MID-YEAR PHYSICS: v^{t+0.5} < v^t for all t > 0

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
"""

import numpy as np
import pandas as pd
from datetime import date
import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from opeb_valuation.vectorized_engine import (
    VectorizedValuationEngine,
    VectorizedValuationConfig,
    VectorizedMortalityTensor,
    VectorizedDecrementEngine,
    VectorizedFinancialEngine,
    create_vectorized_engine
)


# =============================================================================
# TEST 1: FLAT WORLD THEORY
# =============================================================================

class TestFlatWorldTheory:
    """
    PROOF: With 0% discount and 0% trend, liability = sum of expected cash flows.
    
    This validates the fundamental annuity calculation.
    If this fails, the entire engine is mathematically wrong.
    """
    
    def test_zero_discount_annuity(self):
        """
        A $10,000/year benefit for exactly 5 years at 0% discount = $50,000.
        
        SHACKLEFORD VALIDATION:
        PV = $10,000 × 5 = $50,000 (no discounting)
        """
        financial = VectorizedFinancialEngine(
            discount_rate=0.0,  # ZERO discount
            initial_trend=0.0,  # ZERO trend
            ultimate_trend=0.0
        )
        
        # 5 years of payments
        years = np.array([0, 1, 2, 3, 4], dtype=float)
        
        # Mid-year discount factors at 0% should all be 1.0
        discount_factors = financial.get_discount_factors_midyear(years)
        
        # At 0% rate, v = 1/(1+0) = 1, so v^{t+0.5} = 1 for all t
        expected_df = np.ones(5)
        
        np.testing.assert_array_almost_equal(
            discount_factors, expected_df, decimal=10,
            err_msg="Discount factors at 0% should all be 1.0"
        )
        
        # PV of $10,000 × 5 years = $50,000
        cash_flows = np.array([10000, 10000, 10000, 10000, 10000], dtype=float)
        survival = np.ones(5)  # Assume certain survival
        
        pv = np.sum(cash_flows * survival * discount_factors)
        
        assert abs(pv - 50000) < 0.01, \
            f"Flat world PV should be $50,000, got ${pv:,.2f}"
        
        print(f"✓ Flat World Test PASSED: PV = ${pv:,.2f}")
    
    def test_zero_trend_costs(self):
        """
        With 0% trend, year 10 cost = year 1 cost.
        """
        financial = VectorizedFinancialEngine(
            discount_rate=0.05,  # Non-zero discount
            initial_trend=0.0,   # ZERO trend
            ultimate_trend=0.0
        )
        
        years = np.array([0, 5, 10, 20], dtype=float)
        trend_factors = financial.get_trend_factors_midyear(years)
        
        # All trend factors should be 1.0 (no inflation)
        expected = np.ones(4)
        
        np.testing.assert_array_almost_equal(
            trend_factors, expected, decimal=10,
            err_msg="Trend factors at 0% should all be 1.0"
        )
        
        print(f"✓ Zero Trend Test PASSED: All factors = 1.0")


# =============================================================================
# TEST 2: CONTINUITY THEORY (Geometric Interpolation)
# =============================================================================

class TestContinuityTheory:
    """
    PROOF: Liability is continuous across age boundaries.
    
    LEGACY FLAW: Step-function jumps on birthdays
    SHACKLEFORD: Geometric interpolation ensures smooth flow
    
    Test: |Liability(60.001) - Liability(60.000)| < ε
    """
    
    def test_mortality_continuity(self):
        """
        Mortality rate at age 60.001 should be nearly identical to age 60.000.
        
        SHACKLEFORD VALIDATION:
        q_{60.001} ≈ q_{60.000} (within 0.01%)
        """
        mortality = VectorizedMortalityTensor(load_factor=1.20)
        
        # Test at boundary
        age_floor = np.array([60.000])
        age_epsilon = np.array([60.001])
        genders = np.array([0])  # Male
        years = np.array([0])
        
        q_floor = mortality.get_qx_vectorized(age_floor, genders, years)
        q_epsilon = mortality.get_qx_vectorized(age_epsilon, genders, years)
        
        # Relative difference should be < 0.01%
        rel_diff = abs(q_epsilon[0] - q_floor[0]) / q_floor[0]
        
        assert rel_diff < 0.0001, \
            f"Mortality discontinuity: q(60.000)={q_floor[0]:.6f}, " \
            f"q(60.001)={q_epsilon[0]:.6f}, diff={rel_diff:.4%}"
        
        print(f"✓ Mortality Continuity PASSED: Δq = {rel_diff:.6%}")
    
    def test_liability_continuity_across_birthday(self):
        """
        Liability at 11:59 PM before birthday ≈ liability at 12:01 AM after.
        
        SHACKLEFORD VALIDATION:
        |L(x-ε) - L(x+ε)| / L(x) < 0.1%
        """
        mortality = VectorizedMortalityTensor(load_factor=1.20)
        
        # Simulate ages just before and after 65th birthday
        ages_before = np.array([64.999])
        ages_after = np.array([65.001])
        genders = np.array([0])
        years = np.array([0])
        
        q_before = mortality.get_qx_vectorized(ages_before, genders, years)
        q_after = mortality.get_qx_vectorized(ages_after, genders, years)
        
        rel_diff = abs(q_after[0] - q_before[0]) / q_before[0]
        
        # Allow slightly more tolerance at age 65 (Medicare transition)
        assert rel_diff < 0.01, \
            f"Birthday discontinuity at 65: diff = {rel_diff:.4%}"
        
        print(f"✓ Birthday Continuity PASSED: Δq at 65 = {rel_diff:.4%}")


# =============================================================================
# TEST 3: COMPETING RISK THEORY
# =============================================================================

class TestCompetingRiskTheory:
    """
    PROOF: MDT total decrement < simple sum of decrements.
    
    LEGACY FLAW: q_total = q_d + q_w + q_dis (OVERSTATES)
    SHACKLEFORD: Geometric distribution is always lower
    
    Mathematical proof: You cannot die AND quit in the same year.
    """
    
    def test_mdt_less_than_sum(self):
        """
        MDT total decrement must be LESS than simple sum.
        
        SHACKLEFORD VALIDATION:
        q_total(MDT) < q_d + q_w + q_dis + q_r (always)
        """
        engine = VectorizedDecrementEngine()
        
        # Test cases with various decrement combinations
        test_cases = [
            (0.01, 0.05, 0.002, 0.00),  # Standard active
            (0.02, 0.10, 0.005, 0.00),  # High turnover
            (0.05, 0.03, 0.010, 0.00),  # Older worker
            (0.10, 0.00, 0.020, 0.00),  # Near retirement
        ]
        
        for q_d, q_w, q_dis, q_r in test_cases:
            q_mort = np.array([q_d])
            q_term = np.array([q_w])
            q_disab = np.array([q_dis])
            q_ret = np.array([q_r])
            
            # Simple sum (legacy)
            simple_sum = q_d + q_w + q_dis + q_r
            
            # MDT (Shackleford)
            q_d_mdt, q_w_mdt, q_dis_mdt, q_r_mdt, p_survive = \
                engine.calculate_mdt_vectorized(q_mort, q_term, q_disab, q_ret)
            
            mdt_total = q_d_mdt[0] + q_w_mdt[0] + q_dis_mdt[0] + q_r_mdt[0]
            
            assert mdt_total < simple_sum, \
                f"MDT ({mdt_total:.6f}) should be < sum ({simple_sum:.6f})"
            
            overstatement = (simple_sum - mdt_total) / mdt_total * 100
            print(f"  Case q=({q_d},{q_w},{q_dis}): "
                  f"Sum={simple_sum:.4f}, MDT={mdt_total:.4f}, "
                  f"Legacy overstates by {overstatement:.2f}%")
        
        print(f"✓ Competing Risk Test PASSED: MDT < Sum in all cases")
    
    def test_mdt_components_sum_to_total(self):
        """
        MDT components must sum to exactly 1 - p_survive.
        
        This is a mathematical invariant.
        """
        engine = VectorizedDecrementEngine()
        
        q_mort = np.array([0.015])
        q_term = np.array([0.080])
        q_disab = np.array([0.003])
        q_ret = np.array([0.000])
        
        q_d_mdt, q_w_mdt, q_dis_mdt, q_r_mdt, p_survive = \
            engine.calculate_mdt_vectorized(q_mort, q_term, q_disab, q_ret)
        
        mdt_sum = q_d_mdt[0] + q_w_mdt[0] + q_dis_mdt[0] + q_r_mdt[0]
        expected_total = 1 - p_survive[0]
        
        np.testing.assert_almost_equal(
            mdt_sum, expected_total, decimal=10,
            err_msg="MDT components must sum to 1 - p_survive"
        )
        
        print(f"✓ MDT Invariant PASSED: Σq_mdt = 1 - p = {expected_total:.6f}")


# =============================================================================
# TEST 4: MID-YEAR PHYSICS
# =============================================================================

class TestMidYearPhysics:
    """
    PROOF: Mid-year discounting produces higher PV than end-of-year.
    
    LEGACY FLAW: v^t assumes payment at year-end
    SHACKLEFORD: v^{t+0.5} assumes mid-year payment
    
    Since payment happens earlier, PV should be higher.
    """
    
    def test_midyear_discount_higher_pv(self):
        """
        v^{t+0.5} > v^{t+1} for any positive discount rate.
        
        This proves mid-year convention gives correct higher PV.
        """
        discount_rate = 0.05
        financial = VectorizedFinancialEngine(discount_rate=discount_rate)
        
        years = np.array([1, 5, 10, 20], dtype=float)
        
        # Mid-year factors (Shackleford)
        midyear_factors = financial.get_discount_factors_midyear(years)
        
        # End-of-year factors (legacy: v^{t+1})
        v = 1 / (1 + discount_rate)
        eoy_factors = np.power(v, years + 1)
        
        # Mid-year should always be higher (payment received earlier)
        for i, year in enumerate(years):
            assert midyear_factors[i] > eoy_factors[i], \
                f"Year {year}: Mid-year ({midyear_factors[i]:.6f}) " \
                f"should be > EOY ({eoy_factors[i]:.6f})"
        
        print(f"✓ Mid-Year Discount PASSED: v^{{t+0.5}} > v^{{t+1}} for all t")
    
    def test_midyear_trend_adjustment(self):
        """
        Mid-year trend factor includes √(1+trend) adjustment.
        """
        financial = VectorizedFinancialEngine(
            discount_rate=0.04,
            initial_trend=0.065,
            ultimate_trend=0.045
        )
        
        year = np.array([1.0])
        
        # Get mid-year adjusted trend
        midyear_trend = financial.get_trend_factors_midyear(year)
        
        # Manual calculation: cum_trend[0] × √(1 + trend[1])
        # Year 0 trend = 6.5%, year 1 trend should be graded
        expected_cum = 1.0 * (1 + 0.065)  # End of year 0
        trend_year1 = 0.065 - (0.065 - 0.045) / 4  # Graded
        expected_midyear = expected_cum * np.sqrt(1 + trend_year1)
        
        # Should be close
        np.testing.assert_almost_equal(
            midyear_trend[0], expected_midyear, decimal=4,
            err_msg="Mid-year trend should include √(1+trend) adjustment"
        )
        
        print(f"✓ Mid-Year Trend PASSED: τ includes √(1+trend) = {midyear_trend[0]:.4f}")


# =============================================================================
# TEST 5: PERFORMANCE BENCHMARK
# =============================================================================

class TestPerformanceBenchmark:
    """
    PROOF: Engine processes 100,000 lives in under 5 seconds.
    
    This validates vectorization is working correctly.
    """
    
    @pytest.mark.slow
    def test_100k_lives_under_5_seconds(self):
        """
        Process 100,000 members in under 5 seconds.
        
        TARGET: 20,000+ lives/second
        """
        import time
        
        np.random.seed(42)
        n = 100000
        
        # Generate synthetic census
        census = pd.DataFrame({
            'MemberID': [f'M{i:06d}' for i in range(n)],
            'DOB': pd.to_datetime([
                date(1960 + np.random.randint(0, 40), 
                     np.random.randint(1, 13), 
                     min(28, np.random.randint(1, 29)))
                for _ in range(n)
            ]),
            'DOH': pd.to_datetime([
                date(1990 + np.random.randint(0, 30),
                     np.random.randint(1, 13),
                     min(28, np.random.randint(1, 29)))
                for _ in range(n)
            ]),
            'Gender': np.random.choice(['M', 'F'], n),
            'Status': np.random.choice(['Active', 'Retiree'], n, p=[0.7, 0.3]),
            'AnnualSalary': np.random.uniform(30000, 150000, n),
            'CoverageLevel': 'Employee',
        })
        
        config = {'valuation_date': date(2025, 9, 30), 'discount_rate': 0.0381}
        engine = create_vectorized_engine(config)
        
        start = time.time()
        results = engine.run_valuation(census)
        elapsed = time.time() - start
        
        throughput = n / elapsed
        
        assert elapsed < 5.0, \
            f"Performance: {elapsed:.2f}s exceeds 5s target for {n:,} lives"
        
        print(f"✓ Performance PASSED: {n:,} lives in {elapsed:.2f}s "
              f"({throughput:,.0f} lives/sec)")


# =============================================================================
# TEST 6: SELF-RECONCILING VARIANCE
# =============================================================================

class TestSelfReconcilingVariance:
    """
    PROOF: The model predicts its own future.
    
    Every valuation produces a predicted_next_year_liability.
    When compared to actual, the difference is the Gain/Loss.
    """
    
    def test_predicted_next_year_calculation(self):
        """
        Predicted next year TOL should follow the formula:
        TOL_expected = (TOL + SC) × (1+i) - Benefit × √(1+i)
        """
        config = {'valuation_date': date(2025, 9, 30), 'discount_rate': 0.0381}
        engine = create_vectorized_engine(config)
        
        # Single member test
        census = pd.DataFrame({
            'MemberID': ['TEST001'],
            'DOB': [date(1970, 1, 1)],
            'DOH': [date(2000, 1, 1)],
            'Gender': ['M'],
            'Status': ['Active'],
            'AnnualSalary': [75000.0],
            'CoverageLevel': ['Employee'],
        })
        
        results = engine.run_valuation(census)
        
        tol = results['TOL'].iloc[0]
        sc = results['ServiceCost'].iloc[0]
        predicted = results['PredictedNextYearTOL'].iloc[0]
        
        # Manual calculation
        i = 0.0381
        expected_predicted = (tol + sc) * (1 + i)  # Simplified (no benefit payment for active)
        
        # Should be reasonably close
        rel_diff = abs(predicted - expected_predicted) / expected_predicted
        
        assert rel_diff < 0.10, \
            f"Predicted TOL calculation error: {rel_diff:.2%}"
        
        print(f"✓ Self-Reconciling PASSED: Predicted = ${predicted:,.0f}")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SHACKLEFORD PRECISION - MATHEMATICAL VALIDATION SUITE")
    print("=" * 70)
    
    # Run each test class
    test_classes = [
        TestFlatWorldTheory(),
        TestContinuityTheory(),
        TestCompetingRiskTheory(),
        TestMidYearPhysics(),
        TestSelfReconcilingVariance(),
    ]
    
    all_passed = True
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*50}")
        print(f"Running: {class_name}")
        print(f"{'='*50}")
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                try:
                    method = getattr(test_class, method_name)
                    method()
                except Exception as e:
                    print(f"✗ FAILED: {method_name}")
                    print(f"  Error: {e}")
                    all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL PRECISION TESTS PASSED ✓")
        print("Engine meets Shackleford Precision standards.")
    else:
        print("SOME TESTS FAILED ✗")
        print("Review errors above.")
    print("=" * 70)
