"""
opeb_valuation/engine.py - Shackleford Precision GASB 75 Valuation Engine

Implements the complete actuarial valuation algorithm with scientific precision:
- Level Percentage of Payroll EAN (not Level Dollar)
- Joint-Life Probabilities for spousal benefits
- Competing Risk MDT framework
- Mid-year discounting and trending

MATHEMATICAL ENHANCEMENTS:
1. Level % of Pay: SC = NC% × Current_Salary (not PVFB/S_tot)
2. Joint-Life: APV_survivor uses conditional probability vectors
3. Competing Risks: MDT geometric distribution

GASB 75 Compliance:
- ¶162: Entry Age Normal attribution
- ¶163-165: Benefit projection methodology
- ¶155-156: Discount rate requirements

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
"""

import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
import logging

from .mortality import MortalityCalculator, create_mortality_calculator
from .decrements import (
    MultipleDecrementCalculator, 
    RetirementEligibility,
    DecrementTensor,
    DecrementType,
    TerminationRates,
    DisabilityRates,
    create_decrement_calculator,
    calculate_mdt_survival_probability
)
from .financials import (
    FinancialEngine, 
    TrendModel, 
    MorbidityModel,
    SalaryProjector
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CensusRecord:
    """Immutable census record with strict typing."""
    participant_id: str
    dob: date
    doh: date
    gender: str
    status: str
    coverage_tier: str
    current_salary: float = 0.0
    current_premium: float = 0.0
    spouse_dob: Optional[date] = None
    
    def __post_init__(self):
        self.gender = self.gender.upper()[0] if self.gender else 'M'
        self.status = self.status.capitalize()
    
    def get_age(self, as_of: date) -> float:
        return (as_of - self.dob).days / 365.25
    
    def get_service(self, as_of: date) -> float:
        return max(0, (as_of - self.doh).days / 365.25)
    
    def get_entry_age(self) -> float:
        return (self.doh - self.dob).days / 365.25
    
    def get_spouse_age(self, as_of: date, default_diff: int = -3) -> float:
        """Get spouse age, with default if not specified."""
        if self.spouse_dob:
            return (as_of - self.spouse_dob).days / 365.25
        else:
            # ASOP 23 Imputation: Assume spouse 3 years younger
            return self.get_age(as_of) + default_diff


@dataclass
class MemberResult:
    """Complete valuation results for a single member."""
    participant_id: str
    status: str
    age: float
    service: float
    entry_age: float
    gender: str
    coverage_tier: str
    
    # Core actuarial values
    pvfb_member: float
    pvfb_spouse: float
    pvfb_total: float
    
    # GASB 75 allocated values
    tol_member: float
    tol_spouse: float
    tol_total: float
    
    # Service cost (Level % of Pay)
    service_cost: float
    normal_cost_pct: float  # NC% for Level % of Pay
    
    # Component breakdown
    medical_pvfb: float
    dental_pvfb: float
    admin_pvfb: float
    
    # EAN attribution details
    attribution_ratio: float
    expected_retirement_age: int
    expected_total_service: float
    
    # Actuarial factors
    annuity_factor: float = 0.0
    prob_reach_retirement: float = 0.0
    duration: float = 0.0
    pv_salary_annuity: float = 0.0


# =============================================================================
# RETIREE VALUATOR
# =============================================================================

class RetireeValuator:
    """
    Retiree Liability Calculator with Joint-Life Spouse Benefits.
    
    SHACKLEFORD PRECISION: Implements conditional probability vectors
    for spousal survivor benefits instead of static marriage loads.
    """
    
    def __init__(self, 
                 mortality: MortalityCalculator,
                 financial: FinancialEngine,
                 morbidity: MorbidityModel,
                 config: Dict):
        self.mortality = mortality
        self.financial = financial
        self.morbidity = morbidity
        self.config = config
        self.max_age = config.get('max_age', 110)
        self.valuation_date = config.get('valuation_date', date(2025, 9, 30))
        self.dental_premiums = config.get('dental_premiums', {})
        self.admin_fee = config.get('admin_fee_monthly', 35.44)
    
    def calc_retiree_pv(
        self, 
        age_start: int, 
        year_start: int,
        gender: str, 
        coverage: str
    ) -> Tuple[float, float, float, float]:
        """
        Calculate PV for retiree with mid-year discounting and trending.
        
        Uses SHACKLEFORD PRECISION:
        - Discount factor: v^{t+0.5}
        - Trend factor: τ(t) × √(1+trend_t)
        """
        medical_pv = 0.0
        dental_pv = 0.0
        admin_pv = 0.0
        val_year = self.valuation_date.year
        
        for k in range(self.max_age - age_start + 1):
            current_age = age_start + k
            current_year = year_start + k
            years_from_val = current_year - val_year
            
            # Survival probability
            if k == 0:
                survival_prob = 1.0
            else:
                survival_prob = self.mortality.get_tpx(
                    age_start, current_age, gender, year_start, 'Retiree'
                )
            
            # SHACKLEFORD: Mid-year discounting
            discount_factor = self.financial.get_discount_factor(years_from_val, mid_year=True)
            
            # SHACKLEFORD: Mid-year trending
            trend_factor = self.financial.get_medical_trend_factor(years_from_val, mid_year=True)
            
            medical_cost = self.morbidity.get_implicit_subsidy(
                current_age, gender, years_from_val, trend_factor
            )
            medical_pv += medical_cost * survival_prob * discount_factor
            
            # Dental with mid-year trend
            dental_monthly = self._get_dental_premium(coverage)
            dental_trend = self.financial.get_dental_trend_factor(years_from_val, mid_year=True)
            dental_cost = dental_monthly * 12 * dental_trend
            dental_pv += dental_cost * survival_prob * discount_factor
            
            # Admin with mid-year trend
            admin_trend = self.financial.get_admin_trend_factor(years_from_val, mid_year=True)
            admin_cost = self.admin_fee * 12 * admin_trend
            admin_pv += admin_cost * survival_prob * discount_factor
        
        total_pv = medical_pv + dental_pv + admin_pv
        return medical_pv, dental_pv, admin_pv, total_pv
    
    def calc_spouse_survivor_pv_joint_life(
        self,
        member_age: int,
        spouse_age: int,
        member_gender: str,
        year_start: int,
        coverage: str
    ) -> float:
        """
        Calculate spouse survivor benefit using JOINT-LIFE PROBABILITIES.
        
        SHACKLEFORD PRECISION ENHANCEMENT:
        Standard approach: Marriage% × Spouse_Annuity (ignores spouse pre-death)
        Enhanced approach: Conditional probability that member dies while spouse alive
        
        Formula:
        APV_survivor = Σ v^t × _tp_x × q_{x+t} × _{t+0.5}p_y × ä_{y+t+0.5}
        
        Where:
        - _tp_x: Probability member survives to year t
        - q_{x+t}: Probability member dies in year t
        - _{t+0.5}p_y: Probability spouse survives to moment of member death
        - ä_{y+t+0.5}: Spouse annuity starting at member death
        """
        val_year = self.valuation_date.year
        spouse_gender = 'F' if member_gender == 'M' else 'M'
        
        apv_survivor = 0.0
        
        # Member survival vector
        member_surv_prev = 1.0
        
        for t in range(self.max_age - member_age):
            current_member_age = member_age + t
            current_spouse_age = spouse_age + t
            years_from_val = (year_start + t) - val_year
            
            # Member survival at start of year t
            if t == 0:
                member_surv = 1.0
            else:
                member_surv = self.mortality.get_tpx(
                    member_age, current_member_age, member_gender, year_start, 'Retiree'
                )
            
            # Member mortality rate in year t
            qx_member = self.mortality.get_qx(
                current_member_age, member_gender, year_start + t, 'Retiree'
            )
            
            # Probability member dies in year t (conditional on surviving to t)
            # This is the "force" that triggers the survivor benefit
            prob_member_dies_year_t = member_surv * qx_member
            
            # SHACKLEFORD: Spouse must survive to mid-year when member dies
            # Use _{t+0.5}p_y (spouse survives to time t+0.5)
            spouse_surv_to_death = self.mortality.get_tpx(
                spouse_age, current_spouse_age + 0.5, spouse_gender, year_start, 'Retiree'
            )
            
            # Probability the contingent event happens
            prob_contingent = prob_member_dies_year_t * spouse_surv_to_death
            
            if prob_contingent < 1e-12:
                member_surv_prev = member_surv
                continue
            
            # Value of spouse annuity starting at member death
            # Spouse age at benefit commencement = spouse_age + t + 0.5
            spouse_annuity_start_age = int(current_spouse_age + 1)
            
            # Calculate spouse annuity from that point
            _, _, _, spouse_annuity = self.calc_retiree_pv(
                spouse_annuity_start_age,
                year_start + t + 1,
                spouse_gender,
                coverage
            )
            
            # Discount to valuation date (mid-year of death)
            discount_factor = self.financial.get_discount_factor(
                years_from_val + 0.5, mid_year=False
            )
            
            # Accumulate
            apv_survivor += prob_contingent * spouse_annuity * discount_factor
            
            member_surv_prev = member_surv
        
        return apv_survivor
    
    def _get_dental_premium(self, coverage: str) -> float:
        coverage_map = {
            'employee': 'Employee', 'employee only': 'Employee',
            'employee + spouse': 'Employee + Spouse',
            'employee + child(ren)': 'Employee + Child(ren)',
            'employee + family': 'Employee + Family',
        }
        std_coverage = coverage_map.get(coverage.lower(), 'Employee')
        return self.dental_premiums.get(std_coverage, 13.24)
    
    def valuate(self, census: CensusRecord) -> MemberResult:
        """Perform full valuation for a retiree."""
        age = int(census.get_age(self.valuation_date))
        service = census.get_service(self.valuation_date)
        val_year = self.valuation_date.year
        
        # Member PVFB
        med_pv, dent_pv, admin_pv, total_pv = self.calc_retiree_pv(
            age, val_year, census.gender, census.coverage_tier
        )
        
        # Spouse PVFB using JOINT-LIFE calculation
        spouse_pv = 0.0
        if self._has_spouse_coverage(census.coverage_tier):
            spouse_age = int(census.get_spouse_age(self.valuation_date))
            spouse_pv = self.calc_spouse_survivor_pv_joint_life(
                age, spouse_age, census.gender, val_year, census.coverage_tier
            )
        
        # Retirees: 100% attribution
        return MemberResult(
            participant_id=census.participant_id,
            status='Retiree',
            age=float(age),
            service=service,
            entry_age=census.get_entry_age(),
            gender=census.gender,
            coverage_tier=census.coverage_tier,
            pvfb_member=total_pv,
            pvfb_spouse=spouse_pv,
            pvfb_total=total_pv + spouse_pv,
            tol_member=total_pv,
            tol_spouse=spouse_pv,
            tol_total=total_pv + spouse_pv,
            service_cost=0.0,
            normal_cost_pct=0.0,
            medical_pvfb=med_pv,
            dental_pvfb=dent_pv,
            admin_pvfb=admin_pv,
            attribution_ratio=1.0,
            expected_retirement_age=age,
            expected_total_service=service,
            prob_reach_retirement=1.0
        )
    
    def _has_spouse_coverage(self, coverage: str) -> bool:
        return 'spouse' in coverage.lower() or 'family' in coverage.lower()


# =============================================================================
# ACTIVE VALUATOR - LEVEL % OF PAY EAN
# =============================================================================

class ActiveValuator:
    """
    Active Employee Liability Calculator with Shackleford Precision.
    
    MATHEMATICAL ENHANCEMENTS:
    1. Level Percentage of Payroll EAN (not Level Dollar)
    2. Joint-Life probabilities for spouse benefits
    3. Competing Risk MDT framework
    4. Backward salary projection to entry age
    """
    
    def __init__(self,
                 mortality: MortalityCalculator,
                 decrements: MultipleDecrementCalculator,
                 financial: FinancialEngine,
                 morbidity: MorbidityModel,
                 config: Dict):
        self.mortality = mortality
        self.decrements = decrements
        self.financial = financial
        self.morbidity = morbidity
        self.config = config
        self.max_age = config.get('max_age', 110)
        self.valuation_date = config.get('valuation_date', date(2025, 9, 30))
        self.dental_premiums = config.get('dental_premiums', {})
        self.admin_fee = config.get('admin_fee_monthly', 35.44)
        self.salary_scale = config.get('salary_scale', 0.03)
        self.married_fraction = config.get('married_fraction', 0.40)
        self.spouse_age_diff = config.get('spouse_age_diff', -3)
        
        # Salary projector for Level % of Pay
        self.salary_projector = SalaryProjector(salary_scale=self.salary_scale)
        
        # Retiree valuator for nested calls
        self.retiree_valuator = RetireeValuator(
            mortality, financial, morbidity, config
        )
    
    def valuate(self, census: CensusRecord) -> MemberResult:
        """
        Perform full valuation for an active employee.
        
        SHACKLEFORD PRECISION:
        1. Calculate PVFB at entry age (hypothetical past reconstruction)
        2. Calculate PV Salary Annuity from entry to retirement
        3. NC% = PVFB_entry / (Sal_entry × ä_sal)
        4. Service Cost = NC% × Current_Salary
        """
        val_date = self.valuation_date
        val_year = val_date.year
        
        age = census.get_age(val_date)
        service = census.get_service(val_date)
        entry_age = census.get_entry_age()
        
        age_int = int(age)
        service_int = int(service)
        
        # Determine retirement age
        retirement_age = RetirementEligibility.get_earliest_retirement_age(
            census.doh, census.dob
        )
        retirement_age = max(retirement_age, age_int + 1)
        
        # ================================================================
        # STEP 1: Calculate PVFB at Current Age
        # ================================================================
        pvfb_total, medical_pvfb, dental_pvfb, admin_pvfb = self._calc_active_pvfb(
            census, age_int, service_int, retirement_age
        )
        
        # ================================================================
        # STEP 2: Spouse PVFB with Joint-Life
        # ================================================================
        spouse_pvfb = 0.0
        if self._has_spouse_coverage(census.coverage_tier):
            spouse_pvfb = self._calc_spouse_pvfb_joint_life(
                census, age_int, service_int, retirement_age
            )
        
        total_pvfb = pvfb_total + spouse_pvfb
        
        # ================================================================
        # STEP 3: EAN Attribution - Level Percentage of Payroll
        # ================================================================
        expected_total_service = retirement_age - entry_age
        
        # Attribution ratio: Past Service / Total Expected Service
        if expected_total_service > 0:
            attribution_ratio = min(1.0, service / expected_total_service)
        else:
            attribution_ratio = 1.0
        
        # TOL = PVFB × Attribution
        tol_member = pvfb_total * attribution_ratio
        tol_spouse = spouse_pvfb * attribution_ratio
        tol_total = total_pvfb * attribution_ratio
        
        # ================================================================
        # STEP 4: Service Cost using Level % of Pay Method
        # ================================================================
        service_cost, normal_cost_pct, pv_sal_annuity = self._calc_service_cost_level_pct_pay(
            census, total_pvfb, entry_age, retirement_age
        )
        
        # Probability of reaching retirement
        prob_reach_retirement = self._calc_prob_survive_active(
            age_int, retirement_age, service_int,
            census.gender, census.doh, census.dob
        )
        
        return MemberResult(
            participant_id=census.participant_id,
            status='Active',
            age=float(age),
            service=service,
            entry_age=entry_age,
            gender=census.gender,
            coverage_tier=census.coverage_tier,
            pvfb_member=pvfb_total,
            pvfb_spouse=spouse_pvfb,
            pvfb_total=total_pvfb,
            tol_member=tol_member,
            tol_spouse=tol_spouse,
            tol_total=tol_total,
            service_cost=service_cost,
            normal_cost_pct=normal_cost_pct,
            medical_pvfb=medical_pvfb,
            dental_pvfb=dental_pvfb,
            admin_pvfb=admin_pvfb,
            attribution_ratio=attribution_ratio,
            expected_retirement_age=retirement_age,
            expected_total_service=expected_total_service,
            prob_reach_retirement=prob_reach_retirement,
            pv_salary_annuity=pv_sal_annuity
        )
    
    def _calc_active_pvfb(
        self,
        census: CensusRecord,
        age_int: int,
        service_int: int,
        retirement_age: int
    ) -> Tuple[float, float, float, float]:
        """Calculate PVFB for active employee (member only)."""
        val_year = self.valuation_date.year
        
        pvfb_total = 0.0
        medical_pvfb = 0.0
        dental_pvfb = 0.0
        admin_pvfb = 0.0
        
        # Probability of reaching retirement (100% retirement at earliest age)
        years_to_retirement = retirement_age - age_int
        if years_to_retirement <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        prob_reach_ret = self._calc_prob_survive_active(
            age_int, retirement_age, service_int,
            census.gender, census.doh, census.dob
        )
        
        if prob_reach_ret < 1e-10:
            return 0.0, 0.0, 0.0, 0.0
        
        # Value benefits at retirement
        ret_year = val_year + years_to_retirement
        med_pv, dent_pv, admin_pv, annuity_value = self.retiree_valuator.calc_retiree_pv(
            retirement_age, ret_year, census.gender, census.coverage_tier
        )
        
        # Discount to valuation date (mid-year convention)
        discount_to_val = self.financial.get_discount_factor(
            years_to_retirement, mid_year=True
        )
        
        pvfb_total = annuity_value * prob_reach_ret * discount_to_val
        medical_pvfb = med_pv * prob_reach_ret * discount_to_val
        dental_pvfb = dent_pv * prob_reach_ret * discount_to_val
        admin_pvfb = admin_pv * prob_reach_ret * discount_to_val
        
        return pvfb_total, medical_pvfb, dental_pvfb, admin_pvfb
    
    def _calc_spouse_pvfb_joint_life(
        self,
        census: CensusRecord,
        member_age: int,
        member_service: int,
        retirement_age: int
    ) -> float:
        """
        Calculate spouse PVFB using JOINT-LIFE probabilities.
        
        SHACKLEFORD PRECISION: Conditional probability vector approach
        """
        val_year = self.valuation_date.year
        years_to_retirement = retirement_age - member_age
        
        if years_to_retirement <= 0:
            return 0.0
        
        # Probability member reaches retirement
        prob_reach_ret = self._calc_prob_survive_active(
            member_age, retirement_age, member_service,
            census.gender, census.doh, census.dob
        )
        
        # Spouse demographics at retirement
        spouse_age_now = census.get_spouse_age(self.valuation_date)
        spouse_age_at_ret = int(spouse_age_now + years_to_retirement)
        spouse_gender = 'F' if census.gender == 'M' else 'M'
        
        # Joint-life survivor benefit value at retirement
        ret_year = val_year + years_to_retirement
        spouse_survivor_pv = self.retiree_valuator.calc_spouse_survivor_pv_joint_life(
            retirement_age,
            spouse_age_at_ret,
            census.gender,
            ret_year,
            census.coverage_tier
        )
        
        # Marriage probability
        marriage_prob = self.married_fraction
        
        # Discount to valuation date
        discount_factor = self.financial.get_discount_factor(
            years_to_retirement, mid_year=True
        )
        
        return spouse_survivor_pv * prob_reach_ret * marriage_prob * discount_factor
    
    def _calc_service_cost_level_pct_pay(
        self,
        census: CensusRecord,
        total_pvfb: float,
        entry_age: float,
        retirement_age: int
    ) -> Tuple[float, float, float]:
        """
        Calculate Service Cost using LEVEL PERCENTAGE OF PAYROLL method.
        
        SHACKLEFORD PRECISION ENHANCEMENT:
        Standard (Level Dollar): SC = PVFB / Expected_Total_Service
        Enhanced (Level % Pay): SC = NC% × Current_Salary
        
        Where: NC% = PVFB_entry / (Sal_entry × ä_sal)
        
        This requires:
        1. Backward projection of salary to entry age
        2. Calculation of PV salary annuity from entry to retirement
        """
        if total_pvfb <= 0:
            return 0.0, 0.0, 0.0
        
        salary = census.current_salary
        
        # If no salary, fall back to Level Dollar
        if salary <= 0:
            expected_service = retirement_age - entry_age
            if expected_service > 0:
                sc = total_pvfb / expected_service
                return sc, 0.0, 0.0
            return 0.0, 0.0, 0.0
        
        age = census.get_age(self.valuation_date)
        service = census.get_service(self.valuation_date)
        
        # ================================================================
        # STEP 1: Project salary BACKWARD to entry age
        # ================================================================
        entry_salary = self.salary_projector.project_backward(
            salary, age, entry_age
        )
        
        # ================================================================
        # STEP 2: Calculate PV Salary Annuity from entry to retirement
        # ================================================================
        years_to_ret_from_entry = int(retirement_age - entry_age)
        
        # Build survival probability vector from entry age
        survival_probs = []
        for t in range(years_to_ret_from_entry):
            # Approximate survival (we don't have full decrement history)
            # Use current decrements as proxy
            if t <= service:
                prob = 1.0  # Already survived
            else:
                # Probability of surviving future years
                future_years = t - service
                prob = self._calc_prob_survive_active(
                    int(age), int(age + future_years), int(service),
                    census.gender, census.doh, census.dob
                )
            survival_probs.append(prob)
        
        survival_probs = np.array(survival_probs) if survival_probs else np.array([1.0])
        
        pv_sal_annuity = self.salary_projector.calculate_pv_salary_annuity(
            entry_salary,
            entry_age,
            retirement_age,
            self.financial.discount_rate_eoy,
            survival_probs
        )
        
        # ================================================================
        # STEP 3: Calculate Normal Cost Percentage
        # ================================================================
        if pv_sal_annuity > 0:
            # NC% = PVFB_entry / (Sal_entry × ä_sal)
            # We approximate PVFB_entry ≈ PVFB_current (conservative)
            normal_cost_pct = total_pvfb / pv_sal_annuity
        else:
            normal_cost_pct = 0.0
        
        # ================================================================
        # STEP 4: Service Cost = NC% × Current Salary
        # ================================================================
        service_cost = normal_cost_pct * salary
        
        return service_cost, normal_cost_pct, pv_sal_annuity
    
    def _calc_prob_survive_active(
        self,
        start_age: int,
        end_age: int,
        start_service: int,
        gender: str,
        hire_date: date,
        dob: date
    ) -> float:
        """
        Calculate probability of remaining active using MDT framework.
        
        SHACKLEFORD PRECISION: Uses exact competing risk formula
        """
        if end_age <= start_age:
            return 1.0
        
        prob = 1.0
        val_year = self.valuation_date.year
        
        for t in range(end_age - start_age):
            age = start_age + t
            service = start_service + t
            year = val_year + t
            
            # Get independent rates
            qx_d = self.mortality.get_qx(age, gender, year, 'Active')
            qx_w = TerminationRates.get_rate(age, service)
            qx_dis = DisabilityRates.get_rate(age)
            
            # SHACKLEFORD: Use exact MDT survival formula
            px = calculate_mdt_survival_probability(qx_d, qx_w, 0.0, qx_dis)
            prob *= px
        
        return prob
    
    def _has_spouse_coverage(self, coverage: str) -> bool:
        return 'spouse' in coverage.lower() or 'family' in coverage.lower()


# =============================================================================
# VALUATION ENGINE - MAIN ORCHESTRATOR
# =============================================================================

class ValuationEngine:
    """
    Production GASB 75 Valuation Engine with Shackleford Precision.
    
    MATHEMATICAL ENHANCEMENTS:
    1. Competing Risk MDT framework (geometric apportionment)
    2. Mid-year discounting and trending
    3. Level Percentage of Payroll EAN
    4. Joint-Life spousal benefit calculations
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.valuation_date = config.get('valuation_date', date(2025, 9, 30))
        
        # Initialize mortality calculator
        mortality_load = config.get('mortality_load', 1.20)
        self.mortality = create_mortality_calculator(mortality_load * 100)
        
        # Initialize decrement calculator
        self.decrements = create_decrement_calculator(mortality_load)
        
        # Initialize financial engine
        self.financial = FinancialEngine(
            discount_rate_eoy=config.get('discount_rate', 0.0381),
            discount_rate_boy=config.get('discount_rate_boy', 0.0409),
            dental_trend=config.get('dental_trend', 0.04),
            admin_trend=config.get('admin_trend', 0.03),
            valuation_year=self.valuation_date.year
        )
        
        # Build morbidity model
        self.morbidity = self._build_morbidity_model(config)
        
        # Initialize valuators
        self.retiree_valuator = RetireeValuator(
            self.mortality, self.financial, self.morbidity, config
        )
        self.active_valuator = ActiveValuator(
            self.mortality, self.decrements, self.financial, 
            self.morbidity, config
        )
        
        logger.info(
            f"ValuationEngine initialized: date={self.valuation_date}, "
            f"discount={config.get('discount_rate', 0.0381):.2%}"
        )
    
    def _build_morbidity_model(self, config: Dict) -> MorbidityModel:
        """Build morbidity model from configuration."""
        medical_aw = config.get('medical_aw_table')
        
        model = MorbidityModel(
            base_cost_pre65=config.get('base_cost_pre65', 667.68),
            base_cost_post65=config.get('base_cost_post65', 459.89),
            contribution_rate=config.get('contribution_rate', 0.45)
        )
        
        if medical_aw is not None:
            for age in range(15, 111):
                if age in medical_aw.index:
                    row = medical_aw.loc[age]
                    model.morbidity_factors_male[age] = row.get('Male_Factor', 1.0)
                    model.morbidity_factors_female[age] = row.get('Female_Factor', 1.0)
        else:
            for age in range(15, 111):
                model.morbidity_factors_male[age] = 1.0
                model.morbidity_factors_female[age] = 1.0
        
        return model
    
    def run_valuation(
        self, 
        census_actives: pd.DataFrame,
        census_retirees: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Run full valuation on all members."""
        results = []
        total = len(census_actives) + len(census_retirees)
        processed = 0
        
        logger.info(
            f"Starting valuation: {len(census_actives)} actives, "
            f"{len(census_retirees)} retirees"
        )
        
        # Process actives
        for idx, row in census_actives.iterrows():
            census = self._row_to_census(row, 'Active')
            result = self.active_valuator.valuate(census)
            results.append(result)
            processed += 1
            if progress_callback:
                progress_callback(processed, total)
        
        # Process retirees
        for idx, row in census_retirees.iterrows():
            census = self._row_to_census(row, 'Retiree')
            result = self.retiree_valuator.valuate(census)
            results.append(result)
            processed += 1
            if progress_callback:
                progress_callback(processed, total)
        
        # Convert to DataFrame
        results_df = self._results_to_dataframe(results)
        
        logger.info(
            f"Valuation complete: TOL=${results_df['TOL'].sum():,.0f}, "
            f"SC=${results_df['ServiceCost'].sum():,.0f}"
        )
        
        return results_df
    
    def _row_to_census(self, row: pd.Series, status: str) -> CensusRecord:
        """Convert DataFrame row to CensusRecord with ASOP 23 imputation."""
        dob = self._parse_date(row.get('DOB'))
        doh = self._parse_date(row.get('DateOfHire', row.get('DOH')))
        spouse_dob = self._parse_date(row.get('SpouseDOB'))
        
        # ASOP 23 Imputation: Date of Hire
        if doh is None:
            service = row.get('Service', 0)
            doh = date(self.valuation_date.year - int(service), 1, 1)
        
        # ASOP 23 Imputation: Date of Birth
        if dob is None:
            age = row.get('Age', 40)
            dob = date(self.valuation_date.year - int(age), 1, 1)
        
        return CensusRecord(
            participant_id=str(row.get('ID', row.get('MemberID', f'{status[0]}{row.name}'))),
            dob=dob,
            doh=doh,
            gender=str(row.get('Gender', 'M')),
            status=status,
            coverage_tier=str(row.get('CoverageLevel', 'Employee')),
            current_salary=float(row.get('AnnualSalary', 0) or 0),
            current_premium=float(row.get('CurrentPremium', 0) or 0),
            spouse_dob=spouse_dob
        )
    
    def _parse_date(self, val) -> Optional[date]:
        """Parse various date formats."""
        if val is None or pd.isna(val):
            return None
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, date):
            return val
        if isinstance(val, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(val, fmt).date()
                except ValueError:
                    continue
        return None
    
    def _results_to_dataframe(self, results: List[MemberResult]) -> pd.DataFrame:
        """Convert list of MemberResult to DataFrame."""
        return pd.DataFrame([
            {
                'MemberID': r.participant_id,
                'Status': r.status,
                'Age': r.age,
                'Service': r.service,
                'Gender': r.gender,
                'CoverageLevel': r.coverage_tier,
                'PVFB': r.pvfb_member,
                'PVFB_Spouse': r.pvfb_spouse,
                'TOL': r.tol_member,
                'TOL_Spouse': r.tol_spouse,
                'ServiceCost': r.service_cost,
                'NormalCostPct': r.normal_cost_pct,
                'Medical_PVFB': r.medical_pvfb,
                'Dental_PVFB': r.dental_pvfb,
                'Admin_PVFB': r.admin_pvfb,
                'AttributionRatio': r.attribution_ratio,
                'RetirementAge': r.expected_retirement_age,
                'EntryAge': r.entry_age,
                'ProbReachRetirement': r.prob_reach_retirement,
                'PV_SalaryAnnuity': r.pv_salary_annuity,
            }
            for r in results
        ])


def create_engine(config: Dict) -> ValuationEngine:
    """Factory function to create ValuationEngine."""
    return ValuationEngine(config)
