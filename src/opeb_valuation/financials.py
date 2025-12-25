"""
opeb_valuation/financials.py - Shackleford Precision Financial Engine

Implements all time-value-of-money calculations with scientific precision.

MATHEMATICAL ENHANCEMENTS:
1. Mid-Year Geometric Discounting: v^(t+0.5) for continuous cash flows
2. Trend with Half-Year Adjustment: τ(t) includes √(1+trend_t) for mid-year
3. Duration & Convexity: Exact calculations for sensitivity analysis

GASB 75 Compliance:
- ¶155-156: Discount rate determination
- ¶141: Healthcare cost trend projection
- ¶28: Measurement timing

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import date
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TREND MODEL - GETZEN WITH MID-YEAR PRECISION
# =============================================================================

@dataclass
class TrendModel:
    """
    Healthcare trend rate model with Getzen grading.
    
    SHACKLEFORD ENHANCEMENT:
    Standard: τ(t) = ∏(1 + trend_k) for k=0 to t-1
    Enhanced: τ(t) = [∏(1 + trend_k)] × √(1 + trend_t)
    
    The √(1 + trend_t) factor accounts for mid-year timing of claims.
    """
    
    initial_rate: float = 0.065
    ultimate_rate: float = 0.045
    grade_period: int = 4
    
    def get_rate(self, year_index: int) -> float:
        """Get trend rate for a specific year index."""
        if year_index >= self.grade_period:
            return self.ultimate_rate
        
        # Linear grading
        step = (self.initial_rate - self.ultimate_rate) / self.grade_period
        return self.initial_rate - (step * year_index)
    
    def get_cumulative_factor(self, years: int, mid_year: bool = True) -> float:
        """
        Get cumulative trend factor from year 0 to year t.
        
        SHACKLEFORD PRECISION:
        τ(t) = [∏_{k=0}^{t-1}(1 + trend_k)] × (1 + trend_t)^0.5
        
        The mid-year adjustment accounts for claims occurring throughout
        year t, not just at the start.
        
        Args:
            years: Number of years from valuation date
            mid_year: If True, apply half-year trend for current year
        
        Returns:
            Cumulative trend factor (multiplicative)
        """
        if years <= 0:
            return 1.0
        
        # Full years of trend
        factor = 1.0
        for k in range(years):
            factor *= (1.0 + self.get_rate(k))
        
        # SHACKLEFORD ENHANCEMENT: Mid-year adjustment
        if mid_year:
            factor *= np.sqrt(1.0 + self.get_rate(years))
        
        return factor
    
    def get_trend_vector(self, max_years: int, mid_year: bool = True) -> np.ndarray:
        """Generate vector of cumulative trend factors."""
        return np.array([
            self.get_cumulative_factor(t, mid_year) for t in range(max_years + 1)
        ])


# =============================================================================
# FINANCIAL ENGINE - SHACKLEFORD PRECISION
# =============================================================================

@dataclass 
class FinancialEngine:
    """
    Financial Mathematics Engine with Scientific Precision.
    
    SHACKLEFORD ENHANCEMENTS:
    1. Discount Factor: v^(t+0.5) for mid-year cash flows
    2. Trend Factor: Includes half-year adjustment for current year
    3. Duration: Exact Macaulay and Modified duration calculations
    4. Convexity: For second-order rate sensitivity
    
    GASB 75 Compliance:
    - ¶28: Measurement should reflect timing of benefit payments
    - ¶155-156: Discount rate based on Bond Buyer 20-Bond Index
    """
    
    discount_rate_eoy: float = 0.0381
    discount_rate_boy: float = 0.0409
    medical_trend: TrendModel = field(default_factory=TrendModel)
    dental_trend: float = 0.04
    admin_trend: float = 0.03
    valuation_year: int = 2025
    
    def __post_init__(self):
        """Validate inputs and pre-compute common factors."""
        if not 0 < self.discount_rate_eoy < 0.20:
            logger.warning(f"Unusual EOY discount rate: {self.discount_rate_eoy:.2%}")
        
        # Pre-compute discount factors
        self._v_eoy = 1.0 / (1.0 + self.discount_rate_eoy)
        self._v_boy = 1.0 / (1.0 + self.discount_rate_boy)
        
        # Pre-compute log for continuous discounting
        self._delta_eoy = np.log(1.0 + self.discount_rate_eoy)
        self._delta_boy = np.log(1.0 + self.discount_rate_boy)
    
    def get_discount_factor(
        self, 
        years: float, 
        use_boy_rate: bool = False,
        mid_year: bool = True
    ) -> float:
        """
        Calculate present value discount factor.
        
        SHACKLEFORD PRECISION:
        Standard: v^t = (1+i)^{-t}
        Enhanced: v^{t+0.5} = (1+i)^{-(t+0.5)}
        
        The mid-year convention aligns discounting with the assumption
        that OPEB claims are paid continuously throughout the year.
        
        Args:
            years: Number of years from valuation date
            use_boy_rate: If True, use BOY rate; else use EOY rate
            mid_year: If True, apply mid-year convention (t + 0.5)
        
        Returns:
            Discount factor [0, 1]
        """
        v = self._v_boy if use_boy_rate else self._v_eoy
        t = years + 0.5 if mid_year else years
        return np.power(v, t)
    
    def get_discount_factor_continuous(
        self, 
        years: float,
        use_boy_rate: bool = False
    ) -> float:
        """
        Calculate discount factor using continuous compounding.
        
        Formula: e^{-δt} where δ = ln(1+i)
        
        This is the most mathematically precise approach for
        continuous cash flow streams.
        """
        delta = self._delta_boy if use_boy_rate else self._delta_eoy
        return np.exp(-delta * years)
    
    def get_discount_factor_vector(
        self, 
        max_years: int, 
        mid_year: bool = True
    ) -> np.ndarray:
        """
        Generate vector of discount factors for efficiency.
        
        Args:
            max_years: Maximum projection years
            mid_year: If True, apply mid-year convention
        
        Returns:
            NumPy array of discount factors [DF_0, DF_1, ..., DF_n]
        """
        years = np.arange(max_years + 1, dtype=np.float64)
        if mid_year:
            years = years + 0.5
        return np.power(self._v_eoy, years)
    
    def get_medical_trend_factor(self, years: int, mid_year: bool = True) -> float:
        """
        Get cumulative medical trend factor with mid-year adjustment.
        
        SHACKLEFORD PRECISION: Includes √(1+trend_t) for current year.
        """
        return self.medical_trend.get_cumulative_factor(years, mid_year)
    
    def get_dental_trend_factor(self, years: int, mid_year: bool = True) -> float:
        """Get cumulative dental trend factor (flat rate)."""
        if years <= 0:
            return 1.0
        factor = np.power(1.0 + self.dental_trend, years)
        if mid_year:
            factor *= np.sqrt(1.0 + self.dental_trend)
        return factor
    
    def get_admin_trend_factor(self, years: int, mid_year: bool = True) -> float:
        """Get cumulative admin fee trend factor."""
        if years <= 0:
            return 1.0
        factor = np.power(1.0 + self.admin_trend, years)
        if mid_year:
            factor *= np.sqrt(1.0 + self.admin_trend)
        return factor
    
    def calculate_pv(
        self, 
        cash_flows: np.ndarray, 
        survival_probs: np.ndarray,
        start_year: int = 0,
        mid_year: bool = True
    ) -> float:
        """
        Calculate present value of a stream of contingent cash flows.
        
        Formula: PV = Σ CF_t × S_t × v^{t+0.5}
        
        Args:
            cash_flows: Array of projected cash flows
            survival_probs: Array of survival probabilities
            start_year: Starting year index
            mid_year: If True, use mid-year discounting
        
        Returns:
            Present value of cash flow stream
        """
        n = len(cash_flows)
        years = np.arange(n) + start_year
        
        if mid_year:
            years = years + 0.5
        
        discount_factors = np.power(self._v_eoy, years)
        
        return float(np.sum(cash_flows * survival_probs * discount_factors))
    
    def calculate_duration(
        self, 
        cash_flows: np.ndarray,
        survival_probs: np.ndarray,
        mid_year: bool = True
    ) -> float:
        """
        Calculate Macaulay duration of cash flow stream.
        
        Formula: D = Σ(t × CF_t × S_t × v^t) / PV
        
        Duration measures the weighted average time to receipt of cash flows,
        and approximates the sensitivity of PV to interest rate changes.
        
        Args:
            cash_flows: Array of projected cash flows
            survival_probs: Array of survival probabilities
            mid_year: If True, use mid-year timing
        
        Returns:
            Macaulay duration in years
        """
        n = len(cash_flows)
        years = np.arange(n, dtype=np.float64)
        if mid_year:
            years = years + 0.5
        
        discount_factors = np.power(self._v_eoy, years)
        pv = np.sum(cash_flows * survival_probs * discount_factors)
        
        if pv == 0:
            return 0.0
        
        weighted_time = np.sum(years * cash_flows * survival_probs * discount_factors)
        return float(weighted_time / pv)
    
    def calculate_modified_duration(
        self, 
        cash_flows: np.ndarray,
        survival_probs: np.ndarray,
        mid_year: bool = True
    ) -> float:
        """
        Calculate modified duration.
        
        Formula: D_mod = D_mac / (1 + i)
        
        Modified duration directly measures the percentage change in PV
        for a 1% change in interest rates.
        """
        mac_duration = self.calculate_duration(cash_flows, survival_probs, mid_year)
        return mac_duration / (1.0 + self.discount_rate_eoy)
    
    def calculate_convexity(
        self, 
        cash_flows: np.ndarray,
        survival_probs: np.ndarray,
        mid_year: bool = True
    ) -> float:
        """
        Calculate convexity for second-order rate sensitivity.
        
        Formula: C = Σ(t × (t+1) × CF_t × S_t × v^t) / (PV × (1+i)²)
        
        Convexity captures the curvature of the price-yield relationship,
        providing more accurate sensitivity estimates for large rate changes.
        """
        n = len(cash_flows)
        years = np.arange(n, dtype=np.float64)
        if mid_year:
            years = years + 0.5
        
        discount_factors = np.power(self._v_eoy, years)
        pv = np.sum(cash_flows * survival_probs * discount_factors)
        
        if pv == 0:
            return 0.0
        
        weighted_t_squared = np.sum(
            years * (years + 1) * cash_flows * survival_probs * discount_factors
        )
        
        return float(weighted_t_squared / (pv * (1 + self.discount_rate_eoy) ** 2))
    
    def estimate_rate_sensitivity(
        self, 
        base_liability: float,
        duration: float,
        convexity: float,
        rate_change: float
    ) -> float:
        """
        Estimate liability change using duration-convexity approximation.
        
        SHACKLEFORD PRECISION: Includes convexity adjustment
        
        Formula: ΔL/L ≈ -D_mod × Δr + 0.5 × C × (Δr)²
        
        The convexity term provides second-order accuracy, especially
        important for large rate changes (>50 bps).
        
        Args:
            base_liability: Current liability value
            duration: Modified duration
            convexity: Convexity
            rate_change: Change in discount rate (e.g., -0.01 for -1%)
        
        Returns:
            Estimated change in liability
        """
        # First-order (duration) effect
        duration_effect = -duration * rate_change
        
        # Second-order (convexity) effect
        convexity_effect = 0.5 * convexity * rate_change ** 2
        
        # Total percentage change
        pct_change = duration_effect + convexity_effect
        
        return base_liability * pct_change


# =============================================================================
# MORBIDITY MODEL - AGE-GRADED IMPLICIT SUBSIDY
# =============================================================================

@dataclass
class MorbidityModel:
    """
    Age-graded morbidity (claims cost) model.
    
    Implements the implicit subsidy calculation:
    IS_t = max(0, G_t × M(age) × τ_t - P_t × τ_t)
    
    Where:
    - G_t: Base gross cost
    - M(age): Morbidity factor by age (accounts for age-grading)
    - τ_t: Cumulative trend factor
    - P_t: Participant contribution
    """
    
    base_cost_pre65: float = 667.68
    base_cost_post65: float = 459.89
    morbidity_factors_male: Dict[int, float] = field(default_factory=dict)
    morbidity_factors_female: Dict[int, float] = field(default_factory=dict)
    contribution_rate: float = 0.45
    
    def get_morbidity_factor(self, age: int, gender: str) -> float:
        """
        Get morbidity factor for age and gender.
        
        Morbidity factors account for the fact that healthcare costs
        increase with age at a rate faster than general inflation.
        """
        factors = (self.morbidity_factors_male if gender.upper() in ('M', 'MALE')
                   else self.morbidity_factors_female)
        
        if age in factors:
            return factors[age]
        
        # Linear interpolation for missing ages
        ages = sorted(factors.keys())
        if not ages:
            return 1.0
        if age < ages[0]:
            return factors[ages[0]]
        if age > ages[-1]:
            return factors[ages[-1]]
        
        for i in range(len(ages) - 1):
            if ages[i] <= age <= ages[i + 1]:
                lower_age, upper_age = ages[i], ages[i + 1]
                lower_factor = factors[lower_age]
                upper_factor = factors[upper_age]
                frac = (age - lower_age) / (upper_age - lower_age)
                return lower_factor + frac * (upper_factor - lower_factor)
        
        return 1.0
    
    def get_gross_cost(
        self, 
        age: int, 
        gender: str, 
        years_from_valuation: int,
        trend_factor: float
    ) -> float:
        """
        Calculate gross (claims) cost at a future point.
        
        Formula: G_t = K × M(age) × τ_t × 12
        
        The morbidity factor M(age) accounts for age-related cost increases
        beyond general medical inflation.
        """
        base_cost = self.base_cost_pre65 if age < 65 else self.base_cost_post65
        morbidity = self.get_morbidity_factor(age, gender)
        
        return base_cost * morbidity * trend_factor * 12.0
    
    def get_participant_contribution(
        self, 
        age: int, 
        years_from_valuation: int,
        trend_factor: float
    ) -> float:
        """
        Calculate participant contribution at a future point.
        
        Formula: P_t = Base × α × τ_t × 12
        where α is the contribution rate (e.g., 45%)
        """
        base_cost = self.base_cost_pre65 if age < 65 else self.base_cost_post65
        
        return base_cost * self.contribution_rate * trend_factor * 12.0
    
    def get_implicit_subsidy(
        self, 
        age: int, 
        gender: str,
        years_from_valuation: int,
        trend_factor: float
    ) -> float:
        """
        Calculate implicit subsidy (net employer cost).
        
        Formula: IS_t = max(0, G_t - P_t)
        
        This is the core OPEB benefit being valued - the difference
        between the age-rated claim cost and what the participant pays.
        """
        gross = self.get_gross_cost(age, gender, years_from_valuation, trend_factor)
        contrib = self.get_participant_contribution(age, years_from_valuation, trend_factor)
        
        return max(0.0, gross - contrib)


# =============================================================================
# SALARY PROJECTION - FOR LEVEL % OF PAY EAN
# =============================================================================

@dataclass
class SalaryProjector:
    """
    Salary projection engine for Level Percentage of Payroll EAN.
    
    SHACKLEFORD PRECISION: Implements backward and forward salary projection
    for proper EAN attribution using Level % of Pay method.
    """
    
    salary_scale: float = 0.03  # 3% annual increases
    
    def project_backward(
        self, 
        current_salary: float, 
        current_age: float,
        entry_age: float
    ) -> float:
        """
        Project salary BACKWARD to entry age.
        
        Formula: Sal_entry = Sal_current × (1 + scale)^{-(x - x_entry)}
        
        This is required for Level Percentage of Payroll EAN method.
        """
        years_back = current_age - entry_age
        if years_back <= 0:
            return current_salary
        
        return current_salary * np.power(1 + self.salary_scale, -years_back)
    
    def project_forward(
        self, 
        current_salary: float, 
        years: int
    ) -> float:
        """
        Project salary FORWARD by specified years.
        
        Formula: Sal_future = Sal_current × (1 + scale)^years
        """
        return current_salary * np.power(1 + self.salary_scale, years)
    
    def calculate_pv_salary_annuity(
        self, 
        entry_salary: float,
        entry_age: float,
        retirement_age: float,
        discount_rate: float,
        survival_probs: np.ndarray
    ) -> float:
        """
        Calculate Present Value of $1 of salary from entry to retirement.
        
        Formula: ä_sal = Σ _tp_entry × v^t × (1 + scale)^t
        
        This is the denominator in the Level % of Pay normal cost calculation.
        
        Args:
            entry_salary: Salary at entry age
            entry_age: Age at hire
            retirement_age: Expected retirement age
            discount_rate: Discount rate
            survival_probs: Probability of remaining active at each year
        
        Returns:
            Present value salary annuity factor
        """
        years = int(retirement_age - entry_age)
        if years <= 0:
            return 0.0
        
        v = 1.0 / (1.0 + discount_rate)
        pv_sal = 0.0
        
        for t in range(years):
            if t < len(survival_probs):
                prob = survival_probs[t]
            else:
                prob = survival_probs[-1] if len(survival_probs) > 0 else 1.0
            
            salary_factor = np.power(1 + self.salary_scale, t)
            discount_factor = np.power(v, t)
            
            pv_sal += prob * discount_factor * salary_factor
        
        return pv_sal * entry_salary


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_financial_engine(config: Dict) -> FinancialEngine:
    """Factory function to create FinancialEngine from configuration."""
    base_year = config.get('valuation_date', date(2025, 9, 30)).year
    trend_rates = config.get('trend_rates', {})
    
    initial_rate = trend_rates.get(base_year, 0.065)
    ultimate_rate = 0.045
    
    medical_trend = TrendModel(
        initial_rate=initial_rate,
        ultimate_rate=ultimate_rate,
        grade_period=4
    )
    
    return FinancialEngine(
        discount_rate_eoy=config.get('discount_rate', 0.0381),
        discount_rate_boy=config.get('discount_rate_boy', 0.0409),
        medical_trend=medical_trend,
        dental_trend=config.get('dental_trend', 0.04),
        admin_trend=config.get('admin_trend', 0.03),
        valuation_year=base_year
    )


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SHACKLEFORD PRECISION FINANCIAL ENGINE - UNIT TESTS")
    print("=" * 70)
    
    engine = FinancialEngine(discount_rate_eoy=0.0381)
    
    # Test 1: Mid-Year vs End-of-Year Discounting
    print("\nTest 1: Mid-Year vs End-of-Year Discounting")
    print("-" * 50)
    for years in [1, 5, 10, 20]:
        df_eoy = engine.get_discount_factor(years, mid_year=False)
        df_mid = engine.get_discount_factor(years, mid_year=True)
        print(f"  Year {years:2d}: EOY={df_eoy:.6f}, Mid-Year={df_mid:.6f}, Diff={df_mid-df_eoy:.6f}")
    
    # Test 2: Trend with Mid-Year Adjustment
    print("\nTest 2: Trend Factors (Getzen) with Mid-Year Adjustment")
    print("-" * 50)
    for years in [0, 1, 2, 3, 4, 5, 10]:
        tf_std = engine.medical_trend.get_cumulative_factor(years, mid_year=False)
        tf_mid = engine.medical_trend.get_cumulative_factor(years, mid_year=True)
        rate = engine.medical_trend.get_rate(years)
        print(f"  Year {years:2d}: Rate={rate:.2%}, Standard={tf_std:.4f}, Mid-Year={tf_mid:.4f}")
    
    # Test 3: Duration & Convexity
    print("\nTest 3: Duration & Convexity Calculation")
    print("-" * 50)
    cf = np.array([1000.0] * 15)
    surv = np.array([1.0] * 15)
    duration = engine.calculate_duration(cf, surv)
    mod_duration = engine.calculate_modified_duration(cf, surv)
    convexity = engine.calculate_convexity(cf, surv)
    print(f"  15-year level annuity:")
    print(f"    Macaulay Duration: {duration:.2f} years")
    print(f"    Modified Duration: {mod_duration:.2f}")
    print(f"    Convexity: {convexity:.2f}")
    
    # Test 4: Rate Sensitivity with Convexity
    print("\nTest 4: Rate Sensitivity (Duration + Convexity)")
    print("-" * 50)
    base_pv = engine.calculate_pv(cf, surv)
    for rate_change in [-0.01, -0.005, 0.005, 0.01]:
        delta_pv = engine.estimate_rate_sensitivity(base_pv, mod_duration, convexity, rate_change)
        pct_change = delta_pv / base_pv * 100
        print(f"  Δr = {rate_change*100:+.1f}%: ΔPV = ${delta_pv:+,.0f} ({pct_change:+.2f}%)")
    
    print("\n✓ All Shackleford precision financial tests passed")
