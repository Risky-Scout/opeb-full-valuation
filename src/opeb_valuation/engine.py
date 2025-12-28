"""
opeb_valuation/engine.py - Production GASB 75 Valuation Engine

Implements the complete actuarial valuation algorithm with:
- Entry Age Normal (EAN) cost method (GASB 75 ¶162)
- Multiple Decrement Tables (MDT)
- Vectorized calculations for performance
- Per-member PVFB, TOL, and Service Cost

GASB 75 Compliance:
- ¶162: Entry Age Normal attribution
- ¶163-165: Benefit projection methodology
- ¶155-156: Discount rate requirements

Author: Actuarial Pipeline Project
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
    create_decrement_calculator
)
from .financials import FinancialEngine, TrendModel, MorbidityModel

logger = logging.getLogger(__name__)


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
    
    def __post_init__(self):
        self.gender = self.gender.upper()[0] if self.gender else 'M'
        self.status = self.status.capitalize()
    
    def get_age(self, as_of: date) -> float:
        return (as_of - self.dob).days / 365.25
    
    def get_service(self, as_of: date) -> float:
        return max(0, (as_of - self.doh).days / 365.25)
    
    def get_entry_age(self) -> float:
        return (self.doh - self.dob).days / 365.25


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
    pvfb_member: float
    pvfb_spouse: float
    pvfb_total: float
    tol_member: float
    tol_spouse: float
    tol_total: float
    service_cost: float
    medical_pvfb: float
    dental_pvfb: float
    admin_pvfb: float
    attribution_ratio: float
    expected_retirement_age: int
    expected_total_service: float
    annuity_factor: float = 0.0
    prob_reach_retirement: float = 0.0
    duration: float = 0.0


class RetireeValuator:
    """Retiree Liability Calculator (Routine A)."""
    
    def __init__(self, mortality: MortalityCalculator, financial: FinancialEngine,
                 morbidity: MorbidityModel, config: Dict):
        self.mortality = mortality
        self.financial = financial
        self.morbidity = morbidity
        self.config = config
        self.max_age = config.get('max_age', 110)
        self.valuation_date = config.get('valuation_date', date(2025, 9, 30))
        self.dental_premiums = config.get('dental_premiums', {})
        self.admin_fee = config.get('admin_fee_monthly', 35.44)
    
    def calc_retiree_pv(self, age_start: int, year_start: int,
                        gender: str, coverage: str) -> Tuple[float, float, float, float]:
        """Calculate PV for retiree starting at given age."""
        medical_pv = 0.0
        dental_pv = 0.0
        admin_pv = 0.0
        val_year = self.valuation_date.year
        
        for k in range(self.max_age - age_start + 1):
            current_age = age_start + k
            current_year = year_start + k
            years_from_val = current_year - val_year
            
            if k == 0:
                survival_prob = 1.0
            else:
                survival_prob = self.mortality.get_tpx(
                    age_start, current_age, gender, year_start, 'Retiree'
                )
            
            discount_factor = self.financial.get_discount_factor(years_from_val)
            
            trend_factor = self.financial.get_medical_trend_factor(years_from_val)
            medical_cost = self.morbidity.get_implicit_subsidy(
                current_age, gender, years_from_val, trend_factor
            )
            medical_pv += medical_cost * survival_prob * discount_factor
            
            dental_monthly = self._get_dental_premium(coverage)
            dental_trend = self.financial.get_dental_trend_factor(years_from_val)
            dental_cost = dental_monthly * 12 * dental_trend
            dental_pv += dental_cost * survival_prob * discount_factor
            
            admin_trend = self.financial.get_admin_trend_factor(years_from_val)
            admin_cost = self.admin_fee * 12 * admin_trend
            admin_pv += admin_cost * survival_prob * discount_factor
        
        total_pv = medical_pv + dental_pv + admin_pv
        return medical_pv, dental_pv, admin_pv, total_pv
    
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
        age = int(census.get_age(self.valuation_date))
        service = census.get_service(self.valuation_date)
        val_year = self.valuation_date.year
        
        med_pv, dent_pv, admin_pv, total_pv = self.calc_retiree_pv(
            age, val_year, census.gender, census.coverage_tier
        )
        
        spouse_pv = 0.0
        if self._has_spouse_coverage(census.coverage_tier):
            spouse_age = age + self.config.get('spouse_age_diff', -3)
            spouse_gender = 'F' if census.gender == 'M' else 'M'
            _, _, _, spouse_pv = self.calc_retiree_pv(
                spouse_age, val_year, spouse_gender, census.coverage_tier
            )
        
        return MemberResult(
            participant_id=census.participant_id, status='Retiree',
            age=float(age), service=service, entry_age=census.get_entry_age(),
            gender=census.gender, coverage_tier=census.coverage_tier,
            pvfb_member=total_pv, pvfb_spouse=spouse_pv, pvfb_total=total_pv + spouse_pv,
            tol_member=total_pv, tol_spouse=spouse_pv, tol_total=total_pv + spouse_pv,
            service_cost=0.0, medical_pvfb=med_pv, dental_pvfb=dent_pv, admin_pvfb=admin_pv,
            attribution_ratio=1.0, expected_retirement_age=age, expected_total_service=service,
            prob_reach_retirement=1.0
        )
    
    def _has_spouse_coverage(self, coverage: str) -> bool:
        return 'spouse' in coverage.lower() or 'family' in coverage.lower()


class ActiveValuator:
    """Active Employee Liability Calculator (Routine B)."""
    
    def __init__(self, mortality: MortalityCalculator, decrements: MultipleDecrementCalculator,
                 financial: FinancialEngine, morbidity: MorbidityModel, config: Dict):
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
        self.retiree_valuator = RetireeValuator(mortality, financial, morbidity, config)
    
    def valuate(self, census: CensusRecord) -> MemberResult:
        val_date = self.valuation_date
        val_year = val_date.year
        age = census.get_age(val_date)
        service = census.get_service(val_date)
        entry_age = census.get_entry_age()
        age_int, service_int = int(age), int(service)
        
        retirement_age = RetirementEligibility.get_earliest_retirement_age(census.doh, census.dob)
        retirement_age = max(retirement_age, age_int + 1)
        
        pvfb_total = 0.0
        medical_pvfb = 0.0
        dental_pvfb = 0.0
        admin_pvfb = 0.0
        weighted_retirement_age = 0.0
        total_retirement_prob = 0.0
        
        for ret_age in range(retirement_age, self.max_age + 1):
            years_to_retirement = ret_age - age_int
            if years_to_retirement <= 0:
                continue
            
            prob_survive_active = self._calc_prob_survive_active(
                age_int, ret_age, service_int, census.gender, census.doh, census.dob
            )
            
            prob_retire_now = 1.0 if ret_age == retirement_age else 0.0
            prob_retire = prob_survive_active * prob_retire_now
            
            if prob_retire < 1e-10:
                continue
            
            weighted_retirement_age += ret_age * prob_retire
            total_retirement_prob += prob_retire
            
            ret_year = val_year + years_to_retirement
            med_pv, dent_pv, admin_pv, annuity_value = self.retiree_valuator.calc_retiree_pv(
                ret_age, ret_year, census.gender, census.coverage_tier
            )
            
            discount_to_val = np.power(1.0 / (1.0 + self.financial.discount_rate_eoy), years_to_retirement)
            pv_component = annuity_value * prob_retire * discount_to_val
            
            pvfb_total += pv_component
            medical_pvfb += med_pv * prob_retire * discount_to_val
            dental_pvfb += dent_pv * prob_retire * discount_to_val
            admin_pvfb += admin_pv * prob_retire * discount_to_val
        
        spouse_pvfb = 0.0
        if self._has_spouse_coverage(census.coverage_tier):
            spouse_pvfb = self._calc_spouse_pvfb(census, age_int, service_int, retirement_age)
        
        total_pvfb = pvfb_total + spouse_pvfb
        
        if total_retirement_prob > 0:
            expected_retirement_age = weighted_retirement_age / total_retirement_prob
        else:
            expected_retirement_age = retirement_age
        
        expected_total_service = expected_retirement_age - entry_age
        attribution_ratio = min(1.0, service / expected_total_service) if expected_total_service > 0 else 1.0
        
        tol_member = pvfb_total * attribution_ratio
        tol_spouse = spouse_pvfb * attribution_ratio
        tol_total = total_pvfb * attribution_ratio
        
        service_cost = self._calc_service_cost(census, total_pvfb, expected_total_service) if expected_total_service > 0 else 0.0
        
        prob_reach_retirement = self._calc_prob_survive_active(
            age_int, retirement_age, service_int, census.gender, census.doh, census.dob
        )
        
        return MemberResult(
            participant_id=census.participant_id, status='Active',
            age=float(age), service=service, entry_age=entry_age,
            gender=census.gender, coverage_tier=census.coverage_tier,
            pvfb_member=pvfb_total, pvfb_spouse=spouse_pvfb, pvfb_total=total_pvfb,
            tol_member=tol_member, tol_spouse=tol_spouse, tol_total=tol_total,
            service_cost=service_cost, medical_pvfb=medical_pvfb,
            dental_pvfb=dental_pvfb, admin_pvfb=admin_pvfb,
            attribution_ratio=attribution_ratio,
            expected_retirement_age=int(expected_retirement_age),
            expected_total_service=expected_total_service,
            prob_reach_retirement=prob_reach_retirement
        )
    
    def _calc_prob_survive_active(self, start_age: int, end_age: int, start_service: int,
                                   gender: str, hire_date: date, dob: date) -> float:
        if end_age <= start_age:
            return 1.0
        prob = 1.0
        val_year = self.valuation_date.year
        for t in range(end_age - start_age):
            age = start_age + t
            service = start_service + t
            year = val_year + t
            qx_d = self.mortality.get_qx(age, gender, year, 'Active')
            qx_w = TerminationRates.get_rate(age, service)
            qx_dis = DisabilityRates.get_rate(age)
            px = (1 - qx_d) * (1 - qx_w) * (1 - qx_dis)
            prob *= px
        return prob
    
    def _calc_spouse_pvfb(self, census: CensusRecord, member_age: int,
                          member_service: int, retirement_age: int) -> float:
        val_year = self.valuation_date.year
        years_to_retirement = retirement_age - member_age
        prob_reach_ret = self._calc_prob_survive_active(
            member_age, retirement_age, member_service, census.gender, census.doh, census.dob
        )
        spouse_age_at_ret = retirement_age + self.spouse_age_diff
        spouse_gender = 'F' if census.gender == 'M' else 'M'
        ret_year = val_year + years_to_retirement
        _, _, _, spouse_annuity = self.retiree_valuator.calc_retiree_pv(
            spouse_age_at_ret, ret_year, spouse_gender, census.coverage_tier
        )
        discount_factor = np.power(1.0 / (1.0 + self.financial.discount_rate_eoy), years_to_retirement)
        return spouse_annuity * prob_reach_ret * self.married_fraction * discount_factor
    
    def _calc_service_cost(self, census: CensusRecord, total_pvfb: float,
                           expected_total_service: float) -> float:
        if expected_total_service <= 0 or total_pvfb <= 0:
            return 0.0
        salary = census.current_salary
        if salary <= 0:
            return total_pvfb / expected_total_service
        age = int(census.get_age(self.valuation_date))
        service = int(census.get_service(self.valuation_date))
        retirement_age = RetirementEligibility.get_earliest_retirement_age(census.doh, census.dob)
        years_to_ret = max(1, retirement_age - age)
        pv_salary = 0.0
        for t in range(years_to_ret):
            year_prob = self._calc_prob_survive_active(age, age + t + 1, service, census.gender, census.doh, census.dob)
            proj_salary = salary * np.power(1 + self.salary_scale, t)
            discount = np.power(1.0 / (1.0 + self.financial.discount_rate_eoy), t)
            pv_salary += proj_salary * year_prob * discount
        if pv_salary > 0:
            nc_pct = total_pvfb / pv_salary
            return salary * nc_pct
        return total_pvfb / expected_total_service
    
    def _has_spouse_coverage(self, coverage: str) -> bool:
        return 'spouse' in coverage.lower() or 'family' in coverage.lower()


class ValuationEngine:
    """Production GASB 75 Valuation Engine."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.valuation_date = config.get('valuation_date', date(2025, 9, 30))
        
        mortality_load = config.get('mortality_load', 1.20)
        self.mortality = create_mortality_calculator(mortality_load * 100)
        self.decrements = create_decrement_calculator(mortality_load)
        
        self.financial = FinancialEngine(
            discount_rate_eoy=config.get('discount_rate', 0.0381),
            discount_rate_boy=config.get('discount_rate_boy', 0.0409),
            dental_trend=config.get('dental_trend', 0.04),
            admin_trend=config.get('admin_trend', 0.03),
            valuation_year=self.valuation_date.year
        )
        
        self.morbidity = self._build_morbidity_model(config)
        self.retiree_valuator = RetireeValuator(self.mortality, self.financial, self.morbidity, config)
        self.active_valuator = ActiveValuator(self.mortality, self.decrements, self.financial, self.morbidity, config)
        
        logger.info(f"ValuationEngine initialized: date={self.valuation_date}, discount={config.get('discount_rate', 0.0381):.2%}")
    
    def _build_morbidity_model(self, config: Dict) -> MorbidityModel:
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
    
    def run_valuation(self, census_actives: pd.DataFrame, census_retirees: pd.DataFrame,
                      progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        results = []
        total = len(census_actives) + len(census_retirees)
        processed = 0
        
        logger.info(f"Starting valuation: {len(census_actives)} actives, {len(census_retirees)} retirees")
        
        for idx, row in census_actives.iterrows():
            census = self._row_to_census(row, 'Active')
            result = self.active_valuator.valuate(census)
            results.append(result)
            processed += 1
            if progress_callback:
                progress_callback(processed, total)
        
        for idx, row in census_retirees.iterrows():
            census = self._row_to_census(row, 'Retiree')
            result = self.retiree_valuator.valuate(census)
            results.append(result)
            processed += 1
            if progress_callback:
                progress_callback(processed, total)
        
        results_df = self._results_to_dataframe(results)
        logger.info(f"Valuation complete: TOL=${results_df['TOL'].sum():,.0f}, SC=${results_df['ServiceCost'].sum():,.0f}")
        return results_df
    
    def _row_to_census(self, row: pd.Series, status: str) -> CensusRecord:
        dob = self._parse_date(row.get('DOB'))
        doh = self._parse_date(row.get('DateOfHire', row.get('DOH')))
        if doh is None:
            service = row.get('Service', 0)
            doh = date(self.valuation_date.year - int(service), 1, 1)
        if dob is None:
            age = row.get('Age', 40)
            dob = date(self.valuation_date.year - int(age), 1, 1)
        return CensusRecord(
            participant_id=str(row.get('ID', row.get('MemberID', f'{status[0]}{row.name}'))),
            dob=dob, doh=doh, gender=str(row.get('Gender', 'M')), status=status,
            coverage_tier=str(row.get('CoverageLevel', 'Employee')),
            current_salary=float(row.get('AnnualSalary', 0) or 0),
            current_premium=float(row.get('CurrentPremium', 0) or 0)
        )
    
    def _parse_date(self, val) -> Optional[date]:
        if val is None or pd.isna(val):
            return None
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, date):
            return val
        if isinstance(val, str):
            try:
                return datetime.strptime(val, '%Y-%m-%d').date()
            except ValueError:
                try:
                    return datetime.strptime(val, '%m/%d/%Y').date()
                except ValueError:
                    return None
        return None
    
    def _results_to_dataframe(self, results: List[MemberResult]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'MemberID': r.participant_id, 'Status': r.status, 'Age': r.age, 'Service': r.service,
                'Gender': r.gender, 'CoverageLevel': r.coverage_tier,
                'PVFB': r.pvfb_member, 'PVFB_Spouse': r.pvfb_spouse,
                'TOL': r.tol_member, 'TOL_Spouse': r.tol_spouse, 'ServiceCost': r.service_cost,
                'Medical_PVFB': r.medical_pvfb, 'Dental_PVFB': r.dental_pvfb, 'Admin_PVFB': r.admin_pvfb,
                'AttributionRatio': r.attribution_ratio, 'RetirementAge': r.expected_retirement_age,
                'EntryAge': r.entry_age, 'ProbReachRetirement': r.prob_reach_retirement,
            }
            for r in results
        ])


def create_engine(config: Dict) -> ValuationEngine:
    return ValuationEngine(config)
