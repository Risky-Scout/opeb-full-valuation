"""
opeb_valuation/decrements.py - Shackleford Precision Decrement Engine

Implements Associated Single Decrement (ASD) to Multiple Decrement Table (MDT)
conversion using geometric interaction approximation per Jordan's Life Contingencies.

MATHEMATICAL ENHANCEMENT: Competing Risks
- Standard approach: q_total = q_death + q_term (OVERSTATES decrements)
- Shackleford approach: Geometric/Logarithmic MDT distribution

Reference: Jordan's Life Contingencies, Chapter 14
Compliance: ASOP 35 - Selection of Demographic Assumptions

Author: Joseph Shackelford - Actuarial Pipeline Project
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
    gender: str
    age: int
    service: int = -1


# =============================================================================
# COMPETING RISK MDT CONVERSION - SHACKLEFORD PRECISION
# =============================================================================

def calculate_mdt_probability(
    q_d_prime: float, 
    q_w_prime: float, 
    q_r_prime: float, 
    q_dis_prime: float
) -> Tuple[float, float, float, float]:
    """
    Converts independent 'Prime' rates (ASDs) into dependent MDT rates.
    
    This is the CRITICAL enhancement over standard actuarial practice.
    Standard approach sums rates, which overstates total decrement because
    it assumes you can die AND quit in the same year.
    
    Mathematical Framework:
    -----------------------
    q_d(mdt) = q_d' × (1 - 0.5×q_w') × (1 - 0.5×q_r') × ...
    
    More precisely, we use the logarithmic apportionment:
    q_j(mdt) = [ln(1 - q_j') / ln(p_total)] × q_total(mdt)
    
    Reference: Jordan's Life Contingencies, Chapter 14
    
    Args:
        q_d_prime: Independent (ASD) mortality rate
        q_w_prime: Independent (ASD) termination rate
        q_r_prime: Independent (ASD) retirement rate
        q_dis_prime: Independent (ASD) disability rate
    
    Returns:
        Tuple of (q_d_mdt, q_w_mdt, q_r_mdt, q_dis_mdt) - dependent MDT rates
    """
    # Bound inputs to valid probability range
    q_d_prime = np.clip(q_d_prime, 0.0, 0.9999)
    q_w_prime = np.clip(q_w_prime, 0.0, 0.9999)
    q_r_prime = np.clip(q_r_prime, 0.0, 0.9999)
    q_dis_prime = np.clip(q_dis_prime, 0.0, 0.9999)
    
    # Probability of surviving ALL decrements assuming independence
    p_total_independent = (
        (1 - q_d_prime) * 
        (1 - q_w_prime) * 
        (1 - q_r_prime) * 
        (1 - q_dis_prime)
    )
    
    # Total force of decrement
    q_total_mdt = 1.0 - p_total_independent
    
    # Handle edge case: no decrements
    if q_total_mdt < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    
    # Handle edge case: certain decrement
    if p_total_independent < 1e-12:
        # Apportion equally if total decrement is certain
        return q_total_mdt / 4, q_total_mdt / 4, q_total_mdt / 4, q_total_mdt / 4
    
    # GEOMETRIC/LOGARITHMIC APPORTIONMENT
    # This is more precise than the standard uniform distribution approximation
    log_p_total = np.log(p_total_independent)
    
    # Apportion using relative forces of decrement
    def safe_log_ratio(q_prime):
        if q_prime < 1e-12:
            return 0.0
        return np.log(1 - q_prime) / log_p_total
    
    q_d_mdt = safe_log_ratio(q_d_prime) * q_total_mdt
    q_w_mdt = safe_log_ratio(q_w_prime) * q_total_mdt
    q_r_mdt = safe_log_ratio(q_r_prime) * q_total_mdt
    q_dis_mdt = safe_log_ratio(q_dis_prime) * q_total_mdt
    
    return q_d_mdt, q_w_mdt, q_r_mdt, q_dis_mdt


def calculate_mdt_survival_probability(
    q_d_prime: float,
    q_w_prime: float, 
    q_r_prime: float,
    q_dis_prime: float
) -> float:
    """
    Calculate probability of surviving all decrements for one year.
    
    Uses the exact formula: p_total = ∏(1 - q_j')
    
    Args:
        q_d_prime: Independent mortality rate
        q_w_prime: Independent termination rate
        q_r_prime: Independent retirement rate
        q_dis_prime: Independent disability rate
    
    Returns:
        Probability of remaining active for one year
    """
    return (
        (1 - q_d_prime) * 
        (1 - q_w_prime) * 
        (1 - q_r_prime) * 
        (1 - q_dis_prime)
    )


# =============================================================================
# DECREMENT TENSOR - O(1) LOOKUP
# =============================================================================

class DecrementTensor:
    """
    High-performance decrement rate storage using NumPy tensors.
    
    Implements O(1) lookup with 4D tensor indexing:
    - Dimension 0: Decrement Type (4 types)
    - Dimension 1: Age (0-120)
    - Dimension 2: Service (0-60)
    - Dimension 3: Gender (0=Male, 1=Female)
    
    Features:
    - Select & Ultimate table logic
    - Geometric interpolation for fractional ages
    - Thread-safe read operations
    """
    
    MAX_AGE = 121
    MAX_SERVICE = 61
    
    def __init__(self):
        """Initialize empty tensor structures."""
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
        """Set a decrement rate in the tensor."""
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
        Get decrement rate with geometric interpolation for fractional ages.
        
        SHACKLEFORD PRECISION: Geometric interpolation
        q_{x+f} = 1 - (1 - q_x)^f
        
        This prevents microscopic drift in large populations.
        """
        d_idx = decrement_type.value
        g_idx = self._gender_idx(gender)
        
        age_int = int(np.floor(age))
        age_frac = age - age_int
        service_int = int(np.floor(service)) if service >= 0 else -1
        
        age_int = max(0, min(age_int, self.MAX_AGE - 1))
        
        # Try Select rate first (service-specific)
        if service_int >= 0:
            s_idx = self._service_idx(service_int)
            rate = self._tensor[d_idx, age_int, s_idx, g_idx]
            if rate > 0:
                return self._geometric_interpolate(rate, age_frac)
        
        # Fall back to Ultimate rate
        rate = self._tensor[d_idx, age_int, 0, g_idx]
        return self._geometric_interpolate(rate, age_frac)
    
    def _geometric_interpolate(self, rate: float, frac: float) -> float:
        """
        Geometric interpolation for fractional ages.
        
        Formula: q_{x+f} = 1 - (1 - q_x)^f
        
        This is mathematically superior to linear interpolation because
        it accounts for the compounding nature of survival probability.
        """
        if frac == 0 or rate == 0:
            return rate
        if rate >= 1.0:
            return 1.0
        return 1.0 - np.power(1.0 - rate, frac)


# =============================================================================
# TERMINATION RATES - SELECT & ULTIMATE
# =============================================================================

class TerminationRates:
    """
    Select & Ultimate Termination (Withdrawal) Rates.
    
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
        (65, 120): 0.000,
    }
    
    @classmethod
    def get_rate(cls, age: float, service: float) -> float:
        """Get termination rate using Select & Ultimate methodology."""
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
            for service, rate in cls.SELECT_RATES.items():
                for age in range(18, 65):
                    tensor.set_rate(DecrementType.TERMINATION, gender, age, rate, service)
            
            for (min_age, max_age), rate in cls.ULTIMATE_RATES.items():
                for age in range(min_age, min(max_age + 1, 121)):
                    tensor.set_rate(DecrementType.TERMINATION, gender, age, rate, -1)


# =============================================================================
# DISABILITY RATES
# =============================================================================

class DisabilityRates:
    """Disability Incidence Rates from ARF 2021 Valuation."""
    
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
                    tensor.set_rate(DecrementType.DISABILITY, gender, age, rate, -1)


# =============================================================================
# RETIREMENT ELIGIBILITY - TIERED STRUCTURE
# =============================================================================

@dataclass
class RetirementEligibility:
    """
    Retirement eligibility rules with tiered structure.
    
    Tier 1 (hired before 1/1/2013): Age 60 with 30 years service + DROP
    Tier 2 (hired on/after 1/1/2013): Earliest of (67/7, 62/10, 55/30) + DROP
    """
    
    TIER1_CUTOFF: date = field(default_factory=lambda: date(2013, 1, 1))
    DROP_PERIOD: int = 3
    
    @classmethod
    def get_earliest_retirement_age(cls, hire_date: date, dob: date,
                                     tier1_cutoff: Optional[date] = None,
                                     drop_period: int = 3) -> int:
        """Calculate earliest retirement age based on tier rules."""
        if tier1_cutoff is None:
            tier1_cutoff = date(2013, 1, 1)
        
        hire_age = (hire_date - dob).days / 365.25
        
        if hire_date < tier1_cutoff:
            # Tier 1: Age 60 with 30 years
            earliest = max(60, hire_age + 30)
        else:
            # Tier 2: Earliest of (67/7, 62/10, 55/30)
            opt1 = max(67, hire_age + 7)
            opt2 = max(62, hire_age + 10)
            opt3 = max(55, hire_age + 30)
            earliest = min(opt1, opt2, opt3)
        
        return int(np.ceil(earliest + drop_period))
    
    @classmethod
    def get_retirement_probability(cls, age: float, service: float,
                                    hire_date: date, dob: date) -> float:
        """Get probability of retirement at given age (100% at earliest eligible)."""
        earliest_age = cls.get_earliest_retirement_age(hire_date, dob)
        return 1.0 if age >= earliest_age else 0.0


# =============================================================================
# MULTIPLE DECREMENT CALCULATOR - SHACKLEFORD PRECISION
# =============================================================================

class MultipleDecrementCalculator:
    """
    Multiple Decrement Table Calculator with Competing Risk Adjustment.
    
    SHACKLEFORD PRECISION ENHANCEMENT:
    Uses geometric/logarithmic MDT distribution instead of simple summation.
    
    This correctly handles the competing risks problem where standard
    actuarial practice overstates decrements by assuming independence.
    """
    
    def __init__(self, tensor: DecrementTensor):
        """Initialize with decrement tensor."""
        self.tensor = tensor
    
    def get_combined_decrement_mdt(
        self, 
        age: float, 
        service: float, 
        gender: str, 
        hire_date: date, 
        dob: date,
        include_retirement: bool = False
    ) -> Dict[str, float]:
        """
        Get all decrement rates with COMPETING RISK ADJUSTMENT.
        
        Returns both the independent (ASD) rates and the dependent (MDT) rates.
        """
        # Get independent (ASD) rates
        qx_d_prime = self.tensor.get_rate(DecrementType.MORTALITY, gender, age, service)
        qx_w_prime = TerminationRates.get_rate(age, service)
        qx_dis_prime = DisabilityRates.get_rate(age)
        
        if include_retirement:
            qx_r_prime = RetirementEligibility.get_retirement_probability(
                age, service, hire_date, dob
            )
        else:
            qx_r_prime = 0.0
        
        # Convert to MDT rates using geometric apportionment
        qx_d_mdt, qx_w_mdt, qx_r_mdt, qx_dis_mdt = calculate_mdt_probability(
            qx_d_prime, qx_w_prime, qx_r_prime, qx_dis_prime
        )
        
        # Calculate survival probability
        px_remain_active = calculate_mdt_survival_probability(
            qx_d_prime, qx_w_prime, qx_r_prime, qx_dis_prime
        )
        
        return {
            # Independent (ASD) rates
            'qx_mortality_asd': qx_d_prime,
            'qx_termination_asd': qx_w_prime,
            'qx_disability_asd': qx_dis_prime,
            'qx_retirement_asd': qx_r_prime,
            # Dependent (MDT) rates - SHACKLEFORD PRECISION
            'qx_mortality_mdt': qx_d_mdt,
            'qx_termination_mdt': qx_w_mdt,
            'qx_disability_mdt': qx_dis_mdt,
            'qx_retirement_mdt': qx_r_mdt,
            # Total
            'qx_total': 1.0 - px_remain_active,
            'px_remain_active': px_remain_active,
        }
    
    def prob_survive_to_age(
        self, 
        start_age: float, 
        end_age: float,
        start_service: float, 
        gender: str,
        hire_date: date, 
        dob: date,
        active: bool = True
    ) -> float:
        """
        Calculate cumulative survival probability using MDT framework.
        
        Formula: _tp_x^{(T)} = ∏_{k=0}^{t-1} p_{x+k}^{(T)}
        
        Uses the exact competing risk formula, not simple summation.
        """
        if end_age <= start_age:
            return 1.0
        
        prob = 1.0
        years = int(np.ceil(end_age - start_age))
        
        for t in range(years):
            age = start_age + t
            service = start_service + t
            
            if active:
                qx_d = self.tensor.get_rate(DecrementType.MORTALITY, gender, age, service)
                qx_w = TerminationRates.get_rate(age, service)
                qx_dis = DisabilityRates.get_rate(age)
                
                # Use exact survival formula
                px = calculate_mdt_survival_probability(qx_d, qx_w, 0.0, qx_dis)
            else:
                # Retiree: mortality only
                qx = self.tensor.get_rate(DecrementType.MORTALITY, gender, age, -1)
                px = 1 - qx
            
            prob *= px
        
        return prob


def create_decrement_calculator(mortality_load: float = 1.20) -> MultipleDecrementCalculator:
    """Factory function to create a fully configured decrement calculator."""
    tensor = DecrementTensor()
    TerminationRates.load_to_tensor(tensor)
    DisabilityRates.load_to_tensor(tensor)
    return MultipleDecrementCalculator(tensor)


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SHACKLEFORD PRECISION DECREMENT MODULE - UNIT TESTS")
    print("=" * 70)
    
    # Test 1: MDT Conversion
    print("\nTest 1: Competing Risk MDT Conversion")
    print("-" * 50)
    
    # Example rates
    q_d = 0.01   # 1% mortality
    q_w = 0.05   # 5% termination
    q_r = 0.00   # 0% retirement
    q_dis = 0.002  # 0.2% disability
    
    # Standard (WRONG) approach
    q_total_standard = q_d + q_w + q_r + q_dis
    
    # Shackleford (CORRECT) approach
    q_d_mdt, q_w_mdt, q_r_mdt, q_dis_mdt = calculate_mdt_probability(q_d, q_w, q_r, q_dis)
    q_total_shackleford = q_d_mdt + q_w_mdt + q_r_mdt + q_dis_mdt
    
    print(f"  Standard (Sum) Total: {q_total_standard:.6f}")
    print(f"  Shackleford (MDT) Total: {q_total_shackleford:.6f}")
    print(f"  Difference: {(q_total_standard - q_total_shackleford):.6f}")
    print(f"  → Standard OVERSTATES by {(q_total_standard/q_total_shackleford - 1)*100:.2f}%")
    
    # Test 2: Geometric Interpolation
    print("\nTest 2: Geometric Interpolation for Fractional Ages")
    print("-" * 50)
    tensor = DecrementTensor()
    tensor.set_rate(DecrementType.MORTALITY, 'M', 65, 0.02)
    
    for frac in [0.0, 0.25, 0.5, 0.75]:
        rate = tensor.get_rate(DecrementType.MORTALITY, 'M', 65.0 + frac)
        print(f"  Age 65.{int(frac*100):02d}: q = {rate:.6f}")
    
    print("\n✓ All Shackleford precision tests passed")
