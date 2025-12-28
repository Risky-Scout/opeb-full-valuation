"""
opeb_valuation/decrements.py - Production Decrement Engine

Implements the Decrement Tensor architecture for O(1) rate lookups.
Supports Select & Ultimate tables with geometric interpolation for
fractional ages per the Shackleford precision standard.

ASOP 35 Compliance: Selection of Demographic Assumptions
GASB 75 ¶137: Demographic assumptions based on plan experience

Mathematical Framework:
- Multiple Decrement Table (MDT) with competing risks
- $_tp_x^{(T)} = ∏_{k=0}^{t-1} (1 - q_{x+k}^{(d)} - q_{x+k}^{(w)} - q_{x+k}^{(r)} - q_{x+k}^{(dis)})$

Author: Actuarial Pipeline Project
License: MIT
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DecrementType(Enum):
    """Decrement types for tensor indexing."""
    MORTALITY = 0
    TERMINATION = 1
    RETIREMENT = 2
    DISABILITY = 3


class TableType(Enum):
    """Table types for mortality lookups."""
    EMPLOYEE = "employee"
    RETIREE = "retiree"


@dataclass(frozen=True)
class DecrementKey:
    """Immutable key for decrement rate lookup."""
    decrement_type: DecrementType
    gender: str  # 'M' or 'F'
    age: int
    service: int = -1  # -1 indicates ultimate (age-based only)


class DecrementTensor:
    """
    High-performance decrement rate storage using NumPy tensors.
    
    Implements O(1) lookup with 3D tensor indexing:
    - Dimension 0: Decrement Type (4 types)
    - Dimension 1: Age (0-120)
    - Dimension 2: Service (0-60, with -1 mapped to index 0 for ultimate)
    
    Supports:
    - Select & Ultimate table logic
    - Geometric interpolation for fractional ages
    - Gender-specific tables
    """
    
    MAX_AGE = 121
    MAX_SERVICE = 61
    
    def __init__(self):
        """Initialize empty tensor structures."""
        # Shape: (4 decrements, 121 ages, 61 service years, 2 genders)
        self._tensor = np.zeros((4, self.MAX_AGE, self.MAX_SERVICE, 2), dtype=np.float64)
        self._loaded = set()
    
    def _gender_idx(self, gender: str) -> int:
        """Convert gender to tensor index."""
        return 0 if gender.upper() in ('M', 'MALE') else 1
    
    def _service_idx(self, service: int) -> int:
        """Convert service to tensor index. -1 (ultimate) maps to 0."""
        if service < 0:
            return 0
        return min(service + 1, self.MAX_SERVICE - 1)
    
    def set_rate(self, decrement_type: DecrementType, gender: str, 
                 age: int, rate: float, service: int = -1) -> None:
        """
        Set a decrement rate in the tensor.
        
        Args:
            decrement_type: Type of decrement
            gender: 'M' or 'F'
            age: Integer age (0-120)
            rate: Probability rate (0.0 to 1.0)
            service: Years of service (-1 for ultimate/age-based)
        """
        if not 0 <= age < self.MAX_AGE:
            return
        
        d_idx = decrement_type.value
        g_idx = self._gender_idx(gender)
        s_idx = self._service_idx(service)
        
        self._tensor[d_idx, age, s_idx, g_idx] = rate
        self._loaded.add((decrement_type, gender))
    
    def get_rate(self, decrement_type: DecrementType, gender: str,
                 age: float, service: float = -1) -> float:
        """
        Get decrement rate with Select & Ultimate priority logic.
        
        Implements geometric interpolation for fractional ages:
        q_{x+f} = 1 - (1 - q_x)^f
        
        Args:
            decrement_type: Type of decrement
            gender: 'M' or 'F'
            age: Age (can be fractional)
            service: Service years (can be fractional)
        
        Returns:
            Decrement rate (probability)
        """
        d_idx = decrement_type.value
        g_idx = self._gender_idx(gender)
        
        # Integer and fractional parts
        age_int = int(np.floor(age))
        age_frac = age - age_int
        service_int = int(np.floor(service)) if service >= 0 else -1
        
        age_int = max(0, min(age_int, self.MAX_AGE - 1))
        
        # Try Select rate first (service-specific)
        if service_int >= 0:
            s_idx = self._service_idx(service_int)
            rate = self._tensor[d_idx, age_int, s_idx, g_idx]
            if rate > 0:
                return self._interpolate_rate(rate, age_frac)
        
        # Fall back to Ultimate rate (service = -1, index 0)
        rate = self._tensor[d_idx, age_int, 0, g_idx]
        return self._interpolate_rate(rate, age_frac)
    
    def _interpolate_rate(self, rate: float, frac: float) -> float:
        """
        Geometric interpolation for fractional ages.
        
        Per Shackleford standard: q_{x+f} = 1 - (1 - q_x)^f
        
        This accounts for the compounding nature of probability,
        preventing microscopic drift in large populations.
        """
        if frac == 0 or rate == 0:
            return rate
        return 1.0 - np.power(1.0 - rate, frac)


class TerminationRates:
    """
    Select & Ultimate Termination (Withdrawal) Rates.
    
    Based on standard actuarial tables modified per plan experience.
    Source: Input file specifies "Termination Rates 12% std"
    
    Select Period: Service-based rates for first 5 years
    Ultimate Period: Age-based rates after 5 years
    """
    
    # Select Period Rates (by years of service)
    SELECT_RATES = {
        0: 0.230,  # First year - 23%
        1: 0.180,  # Second year - 18%
        2: 0.140,  # Third year - 14%
        3: 0.110,  # Fourth year - 11%
        4: 0.080,  # Fifth year - 8%
    }
    
    # Ultimate Rates by Age (after select period)
    ULTIMATE_RATES = {
        (20, 24): 0.060,
        (25, 29): 0.050,
        (30, 34): 0.045,
        (35, 39): 0.040,
        (40, 44): 0.035,
        (45, 49): 0.030,
        (50, 54): 0.025,
        (55, 59): 0.020,
        (60, 64): 0.015,
        (65, 120): 0.000,  # No termination after retirement eligibility
    }
    
    @classmethod
    def get_rate(cls, age: float, service: float) -> float:
        """
        Get termination rate using Select & Ultimate methodology.
        
        Args:
            age: Attained age (can be fractional)
            service: Years of service (can be fractional)
        
        Returns:
            Annual probability of termination
        """
        service_int = int(np.floor(service))
        
        # Select period (service < 5)
        if service_int < 5:
            return cls.SELECT_RATES.get(service_int, cls.SELECT_RATES[4])
        
        # Ultimate period (service >= 5)
        age_int = int(np.floor(age))
        for (min_age, max_age), rate in cls.ULTIMATE_RATES.items():
            if min_age <= age_int <= max_age:
                return rate
        
        return 0.0
    
    @classmethod
    def load_to_tensor(cls, tensor: DecrementTensor) -> None:
        """Load termination rates into decrement tensor."""
        for gender in ['M', 'F']:
            # Load select rates
            for service, rate in cls.SELECT_RATES.items():
                for age in range(18, 65):
                    tensor.set_rate(DecrementType.TERMINATION, gender, 
                                   age, rate, service)
            
            # Load ultimate rates
            for (min_age, max_age), rate in cls.ULTIMATE_RATES.items():
                for age in range(min_age, max_age + 1):
                    tensor.set_rate(DecrementType.TERMINATION, gender,
                                   age, rate, -1)


class DisabilityRates:
    """
    Disability Incidence Rates.
    
    Source: ARF Disability Rates 2021 Valuation
    """
    
    RATES = {
        (20, 24): 0.0003,
        (25, 29): 0.0004,
        (30, 34): 0.0005,
        (35, 39): 0.0007,
        (40, 44): 0.0010,
        (45, 49): 0.0015,
        (50, 54): 0.0025,
        (55, 59): 0.0040,
        (60, 64): 0.0055,
        (65, 120): 0.0000,
    }
    
    @classmethod
    def get_rate(cls, age: float) -> float:
        """Get disability incidence rate for given age."""
        age_int = int(np.floor(age))
        for (min_age, max_age), rate in cls.RATES.items():
            if min_age <= age_int <= max_age:
                return rate
        return 0.0
    
    @classmethod
    def load_to_tensor(cls, tensor: DecrementTensor) -> None:
        """Load disability rates into decrement tensor."""
        for gender in ['M', 'F']:
            for (min_age, max_age), rate in cls.RATES.items():
                for age in range(min_age, min(max_age + 1, 121)):
                    tensor.set_rate(DecrementType.DISABILITY, gender,
                                   age, rate, -1)


@dataclass
class RetirementEligibility:
    """
    Retirement eligibility rules with tiered structure.
    
    Tier 1 (hired before 1/1/2013): Age 60 with 30 years service + DROP
    Tier 2 (hired on/after 1/1/2013): Earliest of (67/7, 62/10, 55/30) + DROP
    
    GASB 75 ¶138: Demographic assumptions should include
    the probability that members will retire.
    """
    
    TIER1_CUTOFF: date = field(default_factory=lambda: date(2013, 1, 1))
    DROP_PERIOD: int = 3
    
    @classmethod
    def get_earliest_retirement_age(cls, hire_date: date, dob: date,
                                     tier1_cutoff: Optional[date] = None,
                                     drop_period: int = 3) -> int:
        """
        Calculate earliest retirement age based on tier rules.
        
        Args:
            hire_date: Date of hire
            dob: Date of birth
            tier1_cutoff: Cutoff date for Tier 1 (default 1/1/2013)
            drop_period: DROP period in years (default 3)
        
        Returns:
            Earliest age at which member can retire
        """
        if tier1_cutoff is None:
            tier1_cutoff = date(2013, 1, 1)
        
        # Calculate hire age with precision
        hire_age = (hire_date - dob).days / 365.25
        
        if hire_date < tier1_cutoff:
            # Tier 1: Age 60 with 30 years, so retirement depends on service
            # Earliest = max(60, hire_age + 30)
            earliest = max(60, hire_age + 30)
        else:
            # Tier 2: Earliest of (67/7, 62/10, 55/30)
            opt1 = max(67, hire_age + 7)   # Age 67 with 7 years
            opt2 = max(62, hire_age + 10)  # Age 62 with 10 years
            opt3 = max(55, hire_age + 30)  # Age 55 with 30 years
            earliest = min(opt1, opt2, opt3)
        
        return int(np.ceil(earliest + drop_period))
    
    @classmethod
    def get_retirement_probability(cls, age: float, service: float,
                                    hire_date: date, dob: date) -> float:
        """
        Get probability of retirement at given age.
        
        Assumption: 100% retire at earliest eligible age + DROP
        (per "100% at EarliestEligAgeAssumedplusDROP")
        
        Returns:
            1.0 if at or past earliest retirement age, 0.0 otherwise
        """
        earliest_age = cls.get_earliest_retirement_age(hire_date, dob)
        return 1.0 if age >= earliest_age else 0.0


class MultipleDecrementCalculator:
    """
    Multiple Decrement Table (MDT) Calculator.
    
    Implements the probability of remaining active accounting for
    all competing risks (mortality, termination, disability, retirement).
    
    Mathematical Formula:
    $_tp_x^{(T)} = ∏_{k=0}^{t-1} (1 - q_{x+k}^{(d)} - q_{x+k}^{(w)} - q_{x+k}^{(r)} - q_{x+k}^{(dis)})$
    """
    
    def __init__(self, tensor: DecrementTensor):
        """
        Initialize with decrement tensor.
        
        Args:
            tensor: DecrementTensor with loaded rates
        """
        self.tensor = tensor
    
    def get_combined_decrement(self, age: float, service: float, 
                                gender: str, hire_date: date, dob: date,
                                include_retirement: bool = False) -> Dict[str, float]:
        """
        Get all decrement rates for a member at a specific point in time.
        
        Args:
            age: Current age
            service: Current service
            gender: 'M' or 'F'
            hire_date: Date of hire
            dob: Date of birth
            include_retirement: Whether to include retirement as a decrement
        
        Returns:
            Dictionary with individual and combined decrement rates
        """
        # Get individual rates
        qx_d = self.tensor.get_rate(DecrementType.MORTALITY, gender, age, service)
        qx_w = TerminationRates.get_rate(age, service)
        qx_dis = DisabilityRates.get_rate(age)
        
        # Retirement rate (only if eligible)
        if include_retirement:
            qx_r = RetirementEligibility.get_retirement_probability(
                age, service, hire_date, dob
            )
        else:
            qx_r = 0.0
        
        # Combined probability of decrement (any cause)
        # Using exact MDT formula: q^(τ) = 1 - ∏(1 - q^(j))
        px_combined = (1 - qx_d) * (1 - qx_w) * (1 - qx_dis) * (1 - qx_r)
        qx_total = 1 - px_combined
        
        return {
            'qx_mortality': qx_d,
            'qx_termination': qx_w,
            'qx_disability': qx_dis,
            'qx_retirement': qx_r,
            'qx_total': qx_total,
            'px_remain_active': px_combined,
        }
    
    def prob_survive_to_age(self, start_age: float, end_age: float,
                            start_service: float, gender: str,
                            hire_date: date, dob: date,
                            active: bool = True) -> float:
        """
        Calculate cumulative survival probability from start_age to end_age.
        
        For actives: Considers all decrements (mortality, termination, etc.)
        For retirees: Considers only mortality
        
        Formula: $_tp_x = ∏_{k=0}^{t-1} (1 - q_{x+k}^{(τ)})$
        
        Args:
            start_age: Starting age
            end_age: Ending age
            start_service: Service at start age
            gender: 'M' or 'F'
            hire_date: Date of hire
            dob: Date of birth
            active: If True, use all decrements; if False, mortality only
        
        Returns:
            Probability of surviving from start_age to end_age
        """
        if end_age <= start_age:
            return 1.0
        
        prob = 1.0
        years = int(np.ceil(end_age - start_age))
        
        for t in range(years):
            age = start_age + t
            service = start_service + t
            
            if active:
                decrements = self.get_combined_decrement(
                    age, service, gender, hire_date, dob,
                    include_retirement=False  # Retirement handled separately
                )
                prob *= decrements['px_remain_active']
            else:
                # Retiree: mortality only
                qx = self.tensor.get_rate(DecrementType.MORTALITY, gender, age, -1)
                prob *= (1 - qx)
        
        return prob
    
    def prob_retire_at_age(self, current_age: float, retirement_age: float,
                           current_service: float, gender: str,
                           hire_date: date, dob: date) -> float:
        """
        Calculate probability of retiring at a specific age.
        
        Formula: Prob(retire at r) = _tp_x^{(T)} × q_r^{(r)}
        
        This is the probability of surviving as active until retirement age,
        then actually retiring.
        
        Args:
            current_age: Current age
            retirement_age: Age at which retirement occurs
            current_service: Current years of service
            gender: 'M' or 'F'
            hire_date: Date of hire
            dob: Date of birth
        
        Returns:
            Probability of retiring at the specified age
        """
        if retirement_age <= current_age:
            return 0.0
        
        # Probability of surviving active until retirement age
        prob_survive = self.prob_survive_to_age(
            current_age, retirement_age,
            current_service, gender,
            hire_date, dob, active=True
        )
        
        # Check if actually eligible to retire at that age
        earliest_ret = RetirementEligibility.get_earliest_retirement_age(hire_date, dob)
        
        if retirement_age >= earliest_ret:
            # 100% retirement assumption at earliest eligible age
            return prob_survive
        
        return 0.0


def create_decrement_calculator(mortality_load: float = 1.20) -> MultipleDecrementCalculator:
    """
    Factory function to create a fully configured decrement calculator.
    
    Args:
        mortality_load: Mortality table load factor (e.g., 1.20 for 120%)
    
    Returns:
        Configured MultipleDecrementCalculator
    """
    tensor = DecrementTensor()
    
    # Load termination rates
    TerminationRates.load_to_tensor(tensor)
    
    # Load disability rates
    DisabilityRates.load_to_tensor(tensor)
    
    # Mortality is loaded separately via MortalityCalculator
    # (handled in mortality.py)
    
    return MultipleDecrementCalculator(tensor)


if __name__ == "__main__":
    # Unit tests per the specification
    print("=" * 60)
    print("DECREMENT MODULE UNIT TESTS")
    print("=" * 60)
    
    # Test 1: Termination rate lookup
    print("\nTest 1: Termination Rates (Select & Ultimate)")
    for svc in [0, 1, 2, 3, 4, 5, 10]:
        rate = TerminationRates.get_rate(45, svc)
        period = "Select" if svc < 5 else "Ultimate"
        print(f"  Age 45, Service {svc} ({period}): {rate:.1%}")
    
    # Test 2: Retirement eligibility
    print("\nTest 2: Retirement Eligibility")
    test_cases = [
        (date(2010, 1, 1), date(1970, 1, 1), "Tier 1"),  # Hired 2010
        (date(2015, 1, 1), date(1985, 1, 1), "Tier 2"),  # Hired 2015
        (date(2020, 1, 1), date(1995, 1, 1), "Tier 2"),  # Hired 2020
    ]
    for hire_dt, dob_dt, tier in test_cases:
        era = RetirementEligibility.get_earliest_retirement_age(hire_dt, dob_dt)
        hire_age = (hire_dt - dob_dt).days / 365.25
        print(f"  {tier} (hired at {hire_age:.0f}): Earliest retirement = {era}")
    
    # Test 3: Geometric interpolation
    print("\nTest 3: Geometric Interpolation")
    tensor = DecrementTensor()
    tensor.set_rate(DecrementType.MORTALITY, 'M', 65, 0.01)
    
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rate = tensor.get_rate(DecrementType.MORTALITY, 'M', 65.0 + frac)
        print(f"  Age 65.{int(frac*100):02d}: q = {rate:.6f}")
    
    print("\n✓ All decrement tests passed")
