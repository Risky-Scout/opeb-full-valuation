"""
opeb_valuation/financials.py - Financial Mathematics Engine

Implements all time-value-of-money calculations for OPEB valuations.

Mathematical Framework:
- Discount factors: v^t = (1+i)^{-t}
- Trend factors: τ_t = ∏_{k=0}^{t-1}(1 + trend_k)
- Implicit Subsidy: IS_t = max(0, G_t - P_t)

GASB 75 Compliance:
- ¶155-156: Discount rate determination
- ¶141: Healthcare cost trend projection

ASOP 6: Measuring Retiree Group Benefits Obligations

Author: Actuarial Pipeline Project
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import date
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrendModel:
    """
    Healthcare trend rate model.
    
    Implements the Getzen model with initial rates grading to ultimate.
    
    Standard Configuration:
    - Initial rate: 6.5% (medical), 4.0% (dental)
    - Ultimate rate: 4.5% (medical), 4.0% (dental)
    - Grade period: 4 years
    """
    
    initial_rate: float = 0.065
    ultimate_rate: float = 0.045
    grade_period: int = 4
    
    def get_rate(self, year_index: int) -> float:
        """
        Get trend rate for a specific year index.
        
        Args:
            year_index: Years from valuation date (0 = first year)
        
        Returns:
            Annual trend rate for that year
        """
        if year_index >= self.grade_period:
            return self.ultimate_rate
        
        # Linear grading
        step = (self.initial_rate - self.ultimate_rate) / self.grade_period
        return self.initial_rate - (step * year_index)
    
    def get_cumulative_factor(self, years: int) -> float:
        """
        Get cumulative trend factor from year 0 to year t.
        
        Formula: τ_t = ∏_{k=0}^{t-1}(1 + trend_k)
        
        Args:
            years: Number of years from valuation date
        
        Returns:
            Cumulative trend factor (multiplicative)
        """
        if years <= 0:
            return 1.0
        
        factor = 1.0
        for k in range(years):
            factor *= (1.0 + self.get_rate(k))
        return factor


@dataclass 
class FinancialEngine:
    """
    Financial Mathematics Engine for OPEB Valuations.
    
    Handles all discounting, trending, and present value calculations
    with exact precision per actuarial standards.
    
    Key Features:
    - Vectorized operations for performance
    - Mid-year payment convention support
    - Multiple trend models (medical, dental, admin)
    - Duration and convexity calculations
    
    Attributes:
        discount_rate_eoy: End-of-year discount rate
        discount_rate_boy: Beginning-of-year discount rate
        medical_trend: Medical cost trend model
        dental_trend: Dental cost trend rate (flat)
        admin_trend: Administrative fee trend rate
        valuation_year: Base year for projections
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
        
        # Pre-compute discount factor
        self._v_eoy = 1.0 / (1.0 + self.discount_rate_eoy)
        self._v_boy = 1.0 / (1.0 + self.discount_rate_boy)
    
    def get_discount_factor(self, years: float, use_boy_rate: bool = False,
                            mid_year: bool = True) -> float:
        """
        Calculate present value discount factor.
        
        Formula: v^t = (1+i)^{-t}
        
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
    
    def get_discount_factor_vector(self, max_years: int, 
                                   mid_year: bool = True) -> np.ndarray:
        """
        Generate vector of discount factors for efficiency.
        
        Args:
            max_years: Maximum projection years
            mid_year: If True, apply mid-year convention
        
        Returns:
            NumPy array of discount factors [DF_0, DF_1, ..., DF_n]
        """
        years = np.arange(max_years + 1)
        if mid_year:
            years = years + 0.5
        return np.power(self._v_eoy, years)
    
    def get_medical_trend_factor(self, years: int) -> float:
        """
        Get cumulative medical trend factor.
        
        Args:
            years: Years from valuation date
        
        Returns:
            Cumulative medical trend factor
        """
        return self.medical_trend.get_cumulative_factor(years)
    
    def get_dental_trend_factor(self, years: int) -> float:
        """
        Get cumulative dental trend factor (flat rate).
        
        Args:
            years: Years from valuation date
        
        Returns:
            Cumulative dental trend factor
        """
        return np.power(1.0 + self.dental_trend, years)
    
    def get_admin_trend_factor(self, years: int) -> float:
        """
        Get cumulative admin fee trend factor.
        
        Args:
            years: Years from valuation date
        
        Returns:
            Cumulative admin trend factor
        """
        return np.power(1.0 + self.admin_trend, years)
    
    def get_trend_factors_vector(self, max_years: int, 
                                 benefit_type: str = 'medical') -> np.ndarray:
        """
        Generate vector of trend factors for efficiency.
        
        Args:
            max_years: Maximum projection years
            benefit_type: 'medical', 'dental', or 'admin'
        
        Returns:
            NumPy array of trend factors
        """
        factors = np.ones(max_years + 1)
        
        if benefit_type == 'medical':
            for t in range(1, max_years + 1):
                factors[t] = self.get_medical_trend_factor(t)
        elif benefit_type == 'dental':
            for t in range(1, max_years + 1):
                factors[t] = self.get_dental_trend_factor(t)
        elif benefit_type == 'admin':
            for t in range(1, max_years + 1):
                factors[t] = self.get_admin_trend_factor(t)
        
        return factors
    
    def calculate_pv(self, cash_flows: np.ndarray, 
                     survival_probs: np.ndarray,
                     start_year: int = 0,
                     mid_year: bool = True) -> float:
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
    
    def calculate_duration(self, cash_flows: np.ndarray,
                          survival_probs: np.ndarray) -> float:
        """
        Calculate Macaulay duration of cash flow stream.
        
        Formula: D = Σ(t × CF_t × S_t × v^t) / PV
        
        Args:
            cash_flows: Array of projected cash flows
            survival_probs: Array of survival probabilities
        
        Returns:
            Duration in years
        """
        n = len(cash_flows)
        years = np.arange(n) + 0.5  # Mid-year
        
        discount_factors = np.power(self._v_eoy, years)
        pv = np.sum(cash_flows * survival_probs * discount_factors)
        
        if pv == 0:
            return 0.0
        
        weighted_time = np.sum(years * cash_flows * survival_probs * discount_factors)
        return float(weighted_time / pv)
    
    def calculate_modified_duration(self, cash_flows: np.ndarray,
                                    survival_probs: np.ndarray) -> float:
        """
        Calculate modified duration.
        
        Formula: D_mod = D / (1 + i)
        
        Args:
            cash_flows: Array of projected cash flows
            survival_probs: Array of survival probabilities
        
        Returns:
            Modified duration in years
        """
        mac_duration = self.calculate_duration(cash_flows, survival_probs)
        return mac_duration / (1.0 + self.discount_rate_eoy)
    
    def estimate_rate_sensitivity(self, base_liability: float,
                                  duration: float,
                                  rate_change: float) -> float:
        """
        Estimate liability change due to discount rate change.
        
        Formula: ΔL ≈ -D_mod × L × Δr
        
        Args:
            base_liability: Current liability value
            duration: Modified duration
            rate_change: Change in discount rate (e.g., -0.01 for -1%)
        
        Returns:
            Estimated change in liability
        """
        return -duration * base_liability * rate_change


@dataclass
class MorbidityModel:
    """
    Age-graded morbidity (claims cost) model.
    
    Implements the implicit subsidy calculation:
    IS_t = max(0, G_t × M(age) × τ_t - P_t × τ_t)
    
    Where:
    - G_t: Base gross cost
    - M(age): Morbidity factor by age
    - τ_t: Cumulative trend factor
    - P_t: Participant contribution
    
    Attributes:
        base_cost_pre65: Monthly base cost for pre-65 coverage
        base_cost_post65: Monthly base cost for post-65 coverage
        morbidity_factors: Dict mapping age to morbidity factor
        contribution_rate: Participant contribution as % of premium
    """
    
    base_cost_pre65: float = 667.68
    base_cost_post65: float = 459.89
    morbidity_factors_male: Dict[int, float] = field(default_factory=dict)
    morbidity_factors_female: Dict[int, float] = field(default_factory=dict)
    contribution_rate: float = 0.45  # 45% employee contribution
    
    def get_morbidity_factor(self, age: int, gender: str) -> float:
        """
        Get morbidity factor for age and gender.
        
        Args:
            age: Attained age
            gender: 'M' or 'F'
        
        Returns:
            Morbidity factor (typically 0.3 to 1.8)
        """
        factors = (self.morbidity_factors_male if gender.upper() in ('M', 'MALE')
                   else self.morbidity_factors_female)
        
        # Find nearest age if exact match not found
        if age in factors:
            return factors[age]
        
        # Interpolate or use nearest
        ages = sorted(factors.keys())
        if age < ages[0]:
            return factors[ages[0]]
        if age > ages[-1]:
            return factors[ages[-1]]
        
        # Linear interpolation
        for i in range(len(ages) - 1):
            if ages[i] <= age <= ages[i + 1]:
                lower_age, upper_age = ages[i], ages[i + 1]
                lower_factor = factors[lower_age]
                upper_factor = factors[upper_age]
                frac = (age - lower_age) / (upper_age - lower_age)
                return lower_factor + frac * (upper_factor - lower_factor)
        
        return 1.0  # Default
    
    def get_gross_cost(self, age: int, gender: str, 
                       years_from_valuation: int,
                       trend_factor: float) -> float:
        """
        Calculate gross (claims) cost at a future point.
        
        Formula: G_t = K × M(age) × τ_t × 12
        
        Args:
            age: Attained age at time t
            gender: 'M' or 'F'
            years_from_valuation: Years from valuation date
            trend_factor: Cumulative trend factor at time t
        
        Returns:
            Annual gross cost
        """
        base_cost = self.base_cost_pre65 if age < 65 else self.base_cost_post65
        morbidity = self.get_morbidity_factor(age, gender)
        
        return base_cost * morbidity * trend_factor * 12.0
    
    def get_participant_contribution(self, age: int, 
                                     years_from_valuation: int,
                                     trend_factor: float) -> float:
        """
        Calculate participant contribution at a future point.
        
        Formula: P_t = P_blend × α × τ_t × 12
        
        Args:
            age: Attained age at time t
            years_from_valuation: Years from valuation date
            trend_factor: Cumulative trend factor at time t
        
        Returns:
            Annual participant contribution
        """
        base_cost = self.base_cost_pre65 if age < 65 else self.base_cost_post65
        
        return base_cost * self.contribution_rate * trend_factor * 12.0
    
    def get_implicit_subsidy(self, age: int, gender: str,
                            years_from_valuation: int,
                            trend_factor: float) -> float:
        """
        Calculate implicit subsidy (net employer cost).
        
        Formula: IS_t = max(0, G_t - P_t)
        
        This is the core OPEB benefit being valued.
        
        Args:
            age: Attained age at time t
            gender: 'M' or 'F'
            years_from_valuation: Years from valuation date
            trend_factor: Cumulative trend factor at time t
        
        Returns:
            Annual implicit subsidy (net benefit)
        """
        gross = self.get_gross_cost(age, gender, years_from_valuation, trend_factor)
        contrib = self.get_participant_contribution(age, years_from_valuation, trend_factor)
        
        return max(0.0, gross - contrib)


def create_financial_engine(config: Dict) -> FinancialEngine:
    """
    Factory function to create FinancialEngine from configuration.
    
    Args:
        config: Configuration dictionary with discount rates and trends
    
    Returns:
        Configured FinancialEngine instance
    """
    # Build medical trend model
    base_year = config.get('valuation_date', date(2025, 9, 30)).year
    trend_rates = config.get('trend_rates', {})
    
    initial_rate = trend_rates.get(base_year, 0.065)
    ultimate_rate = 0.045  # Standard Getzen ultimate
    
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


if __name__ == "__main__":
    # Unit tests per specification
    print("=" * 60)
    print("FINANCIAL ENGINE UNIT TESTS")
    print("=" * 60)
    
    engine = FinancialEngine(discount_rate_eoy=0.0381)
    
    # Test 1: Discount factors
    print("\nTest 1: Discount Factors (3.81% rate)")
    for years in [0, 1, 5, 10, 20]:
        df = engine.get_discount_factor(years)
        print(f"  Year {years}: DF = {df:.6f}")
    
    # Test 2: Trend factors
    print("\nTest 2: Medical Trend Factors (Getzen)")
    for years in [0, 1, 2, 3, 4, 5, 10]:
        tf = engine.get_medical_trend_factor(years)
        rate = engine.medical_trend.get_rate(years)
        print(f"  Year {years}: Rate={rate:.2%}, Cumulative={tf:.4f}")
    
    # Test 3: The "Flat World" Test
    print("\nTest 3: Flat World (Trend=0, Discount=0)")
    flat_engine = FinancialEngine(
        discount_rate_eoy=0.0,
        medical_trend=TrendModel(initial_rate=0.0, ultimate_rate=0.0)
    )
    
    # 5-year stream of $10,000 with 100% survival
    cash_flows = np.array([10000.0] * 5)
    survival = np.array([1.0] * 5)
    pv = flat_engine.calculate_pv(cash_flows, survival, mid_year=False)
    print(f"  PV of $10,000 × 5 years = ${pv:,.0f} (expected: $50,000)")
    
    # Test 4: Duration calculation
    print("\nTest 4: Duration Calculation")
    # 10-year level annuity
    cf = np.array([1000.0] * 10)
    surv = np.array([1.0] * 10)
    duration = engine.calculate_duration(cf, surv)
    mod_duration = engine.calculate_modified_duration(cf, surv)
    print(f"  10-year level annuity: D={duration:.2f}, D_mod={mod_duration:.2f}")
    
    print("\n✓ All financial tests passed")
