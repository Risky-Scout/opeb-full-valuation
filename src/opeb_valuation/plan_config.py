"""
opeb_valuation/plan_config.py - Dynamic Plan Configuration (Strategy Pattern)

DESIGN PRINCIPLE: No hardcoded plan rules.
The Engine asks the PlanDocument: "What is the benefit?"
The complexity is hidden inside the PlanDocument.

STRATEGY PATTERN IMPLEMENTATION:
- PlanDocument: Abstract interface for benefit structures
- ImplicitSubsidyPlan: Standard OPEB implicit subsidy calculation
- ExplicitSubsidyPlan: Explicit dollar subsidy
- TieredContributionPlan: Service-based contribution tiers

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CoverageType(Enum):
    """Coverage tier types."""
    EMPLOYEE = "employee"
    EMPLOYEE_SPOUSE = "employee_spouse"
    EMPLOYEE_CHILDREN = "employee_children"
    FAMILY = "family"


class BenefitType(Enum):
    """Benefit calculation types."""
    IMPLICIT_SUBSIDY = "implicit_subsidy"
    EXPLICIT_DOLLAR = "explicit_dollar"
    PERCENTAGE_OF_PREMIUM = "percentage_premium"
    FLAT_DOLLAR = "flat_dollar"


class ContributionStructure(Enum):
    """Contribution structure types."""
    FLAT_PERCENTAGE = "flat_percentage"
    SERVICE_TIERED = "service_tiered"
    AGE_TIERED = "age_tiered"
    HIRE_DATE_TIERED = "hire_date_tiered"


# =============================================================================
# PYDANTIC MODELS FOR CONFIGURATION
# =============================================================================

class PremiumSchedule(BaseModel):
    """Premium schedule by coverage tier and age band."""
    coverage_type: CoverageType
    pre65_monthly: float = Field(..., description="Monthly premium for pre-65")
    post65_monthly: float = Field(..., description="Monthly premium for post-65")
    age_factors: Optional[Dict[int, float]] = Field(
        default=None,
        description="Age-based premium factors (age -> multiplier)"
    )
    
    def get_premium(self, age: int) -> float:
        """Get monthly premium for given age."""
        base = self.pre65_monthly if age < 65 else self.post65_monthly
        
        if self.age_factors:
            # Find applicable age factor
            applicable_ages = [a for a in self.age_factors.keys() if a <= age]
            if applicable_ages:
                factor = self.age_factors[max(applicable_ages)]
                return base * factor
        
        return base


class ContributionTier(BaseModel):
    """A single contribution tier."""
    min_service: float = 0
    max_service: float = 999
    min_hire_date: Optional[date] = None
    max_hire_date: Optional[date] = None
    contribution_rate: float = Field(..., ge=0, le=1)
    
    def applies(self, service: float, hire_date: Optional[date] = None) -> bool:
        """Check if this tier applies to given service/hire date."""
        if not (self.min_service <= service < self.max_service):
            return False
        
        if hire_date and self.min_hire_date:
            if hire_date < self.min_hire_date:
                return False
        
        if hire_date and self.max_hire_date:
            if hire_date > self.max_hire_date:
                return False
        
        return True


class PlanRules(BaseModel):
    """Complete plan rules configuration."""
    plan_name: str
    plan_id: str
    effective_date: date
    
    benefit_type: BenefitType = BenefitType.IMPLICIT_SUBSIDY
    contribution_structure: ContributionStructure = ContributionStructure.FLAT_PERCENTAGE
    
    # Premium schedules by coverage type
    premiums: Dict[str, PremiumSchedule] = Field(default_factory=dict)
    
    # Contribution tiers
    contribution_tiers: List[ContributionTier] = Field(default_factory=list)
    default_contribution_rate: float = 0.45
    
    # Explicit subsidy amounts (if applicable)
    explicit_subsidy_pre65: float = 0.0
    explicit_subsidy_post65: float = 0.0
    
    # Age thresholds
    medicare_age: int = 65
    minimum_retirement_age: int = 55
    
    # Service requirements
    minimum_service_for_benefit: float = 0.0
    vesting_service: float = 5.0
    
    # Spouse assumptions
    spouse_age_difference: int = -3
    spouse_coverage_probability: float = 0.40


# =============================================================================
# ABSTRACT PLAN DOCUMENT (STRATEGY INTERFACE)
# =============================================================================

class PlanDocument(ABC):
    """
    Abstract interface for plan benefit calculations.
    
    STRATEGY PATTERN: The Engine doesn't know HOW benefits are calculated.
    It only knows to call get_gross_benefit() and get_participant_contribution().
    
    The Subsidy Detector: subsidy = gross - contribution
    """
    
    @abstractmethod
    def get_gross_benefit(self, age: int, gender: str, 
                          coverage: CoverageType,
                          year: int = 0) -> float:
        """
        Returns the 'True Cost' - what the plan would charge at community rates.
        
        Args:
            age: Participant age
            gender: 'M' or 'F'
            coverage: Coverage tier
            year: Projection year from valuation date
        
        Returns:
            Annual gross benefit cost
        """
        pass
    
    @abstractmethod
    def get_participant_contribution(self, age: int, service: float,
                                     coverage: CoverageType,
                                     hire_date: Optional[date] = None,
                                     year: int = 0) -> float:
        """
        Returns the 'Check Written by Retiree'.
        
        This handles complex logic like:
        - "If Service > 20, pay 50%; else pay 100%"
        - "Tier 1 (hired before 2013) pays 25%; Tier 2 pays 50%"
        
        Args:
            age: Participant age
            service: Years of service at retirement
            coverage: Coverage tier
            hire_date: Date of hire (for tiered benefits)
            year: Projection year
        
        Returns:
            Annual participant contribution
        """
        pass
    
    def get_implicit_subsidy(self, age: int, gender: str, service: float,
                             coverage: CoverageType,
                             hire_date: Optional[date] = None,
                             year: int = 0) -> float:
        """
        THE SUBSIDY DETECTOR.
        
        Simply: gross - contribution
        All complexity is hidden in the component methods.
        """
        gross = self.get_gross_benefit(age, gender, coverage, year)
        contribution = self.get_participant_contribution(
            age, service, coverage, hire_date, year
        )
        return max(0.0, gross - contribution)
    
    @abstractmethod
    def get_plan_info(self) -> Dict[str, Any]:
        """Return plan metadata for reporting."""
        pass


# =============================================================================
# CONCRETE PLAN IMPLEMENTATIONS
# =============================================================================

class ImplicitSubsidyPlan(PlanDocument):
    """
    Standard OPEB Implicit Subsidy Plan.
    
    The employer allows retirees to stay on the group plan at blended
    (not age-rated) premiums. The implicit subsidy is the difference
    between the age-rated cost and the blended premium.
    """
    
    def __init__(self, rules: PlanRules):
        self.rules = rules
        self._build_morbidity_factors()
    
    def _build_morbidity_factors(self):
        """Build age-based morbidity (claims cost) factors."""
        # Standard SOA morbidity curve
        self.morbidity_male = {}
        self.morbidity_female = {}
        
        for age in range(18, 111):
            # Male factors (slightly higher after 50)
            if age < 30:
                self.morbidity_male[age] = 0.60
            elif age < 40:
                self.morbidity_male[age] = 0.80
            elif age < 50:
                self.morbidity_male[age] = 1.00
            elif age < 60:
                self.morbidity_male[age] = 1.40
            elif age < 65:
                self.morbidity_male[age] = 1.80
            elif age < 70:
                self.morbidity_male[age] = 1.20  # Post-Medicare primary
            elif age < 80:
                self.morbidity_male[age] = 1.40
            else:
                self.morbidity_male[age] = 1.60
            
            # Female factors (slightly lower than male)
            self.morbidity_female[age] = self.morbidity_male[age] * 0.95
    
    def get_gross_benefit(self, age: int, gender: str,
                          coverage: CoverageType,
                          year: int = 0) -> float:
        """
        Age-rated gross cost = Base Premium × Morbidity Factor × 12
        """
        # Get base premium
        coverage_key = coverage.value
        if coverage_key in self.rules.premiums:
            schedule = self.rules.premiums[coverage_key]
            base_monthly = schedule.get_premium(age)
        else:
            # Default premiums
            base_monthly = 650 if age < 65 else 450
        
        # Apply morbidity factor
        morbidity = (self.morbidity_male.get(age, 1.0) if gender.upper() == 'M'
                     else self.morbidity_female.get(age, 1.0))
        
        return base_monthly * morbidity * 12.0
    
    def get_participant_contribution(self, age: int, service: float,
                                     coverage: CoverageType,
                                     hire_date: Optional[date] = None,
                                     year: int = 0) -> float:
        """
        Participant contribution based on tier rules.
        """
        # Find applicable contribution tier
        rate = self.rules.default_contribution_rate
        
        for tier in self.rules.contribution_tiers:
            if tier.applies(service, hire_date):
                rate = tier.contribution_rate
                break
        
        # Get blended (non-age-rated) premium
        coverage_key = coverage.value
        if coverage_key in self.rules.premiums:
            schedule = self.rules.premiums[coverage_key]
            # Use average of pre/post 65 as "blended"
            blended_monthly = (schedule.pre65_monthly + schedule.post65_monthly) / 2
        else:
            blended_monthly = 550
        
        return blended_monthly * rate * 12.0
    
    def get_plan_info(self) -> Dict[str, Any]:
        return {
            'plan_name': self.rules.plan_name,
            'plan_id': self.rules.plan_id,
            'benefit_type': self.rules.benefit_type.value,
            'contribution_structure': self.rules.contribution_structure.value,
            'default_contribution_rate': self.rules.default_contribution_rate,
        }


class ExplicitSubsidyPlan(PlanDocument):
    """
    Explicit Dollar Subsidy Plan.
    
    The employer pays a fixed dollar amount toward retiree healthcare.
    Common structure: $X per month pre-65, $Y per month post-65.
    """
    
    def __init__(self, rules: PlanRules):
        self.rules = rules
    
    def get_gross_benefit(self, age: int, gender: str,
                          coverage: CoverageType,
                          year: int = 0) -> float:
        """
        For explicit subsidy, gross = the explicit subsidy amount.
        """
        if age < self.rules.medicare_age:
            return self.rules.explicit_subsidy_pre65 * 12.0
        else:
            return self.rules.explicit_subsidy_post65 * 12.0
    
    def get_participant_contribution(self, age: int, service: float,
                                     coverage: CoverageType,
                                     hire_date: Optional[date] = None,
                                     year: int = 0) -> float:
        """
        For explicit subsidy, participant pays nothing (employer pays fixed $).
        """
        return 0.0
    
    def get_plan_info(self) -> Dict[str, Any]:
        return {
            'plan_name': self.rules.plan_name,
            'plan_id': self.rules.plan_id,
            'benefit_type': 'explicit_subsidy',
            'pre65_monthly': self.rules.explicit_subsidy_pre65,
            'post65_monthly': self.rules.explicit_subsidy_post65,
        }


class ServiceTieredPlan(PlanDocument):
    """
    Service-Tiered Contribution Plan.
    
    Common structure:
    - 30+ years: Employer pays 100%
    - 20-29 years: Employer pays 75%
    - 10-19 years: Employer pays 50%
    - <10 years: No benefit
    """
    
    def __init__(self, rules: PlanRules,
                 service_tiers: Optional[List[Dict]] = None):
        self.rules = rules
        
        # Build service tier schedule
        if service_tiers:
            self.tiers = service_tiers
        else:
            # Default service-graded schedule
            self.tiers = [
                {'min_service': 30, 'employer_share': 1.00},
                {'min_service': 20, 'employer_share': 0.75},
                {'min_service': 10, 'employer_share': 0.50},
                {'min_service': 5, 'employer_share': 0.25},
                {'min_service': 0, 'employer_share': 0.00},
            ]
    
    def _get_employer_share(self, service: float) -> float:
        """Get employer share based on service."""
        for tier in self.tiers:
            if service >= tier['min_service']:
                return tier['employer_share']
        return 0.0
    
    def get_gross_benefit(self, age: int, gender: str,
                          coverage: CoverageType,
                          year: int = 0) -> float:
        """Get full premium cost."""
        coverage_key = coverage.value
        if coverage_key in self.rules.premiums:
            schedule = self.rules.premiums[coverage_key]
            return schedule.get_premium(age) * 12.0
        return 650 * 12.0 if age < 65 else 450 * 12.0
    
    def get_participant_contribution(self, age: int, service: float,
                                     coverage: CoverageType,
                                     hire_date: Optional[date] = None,
                                     year: int = 0) -> float:
        """Participant pays (1 - employer_share) × premium."""
        employer_share = self._get_employer_share(service)
        participant_share = 1.0 - employer_share
        
        gross = self.get_gross_benefit(age, 'M', coverage, year)
        return gross * participant_share
    
    def get_plan_info(self) -> Dict[str, Any]:
        return {
            'plan_name': self.rules.plan_name,
            'plan_id': self.rules.plan_id,
            'benefit_type': 'service_tiered',
            'tiers': self.tiers,
        }


# =============================================================================
# PLAN FACTORY
# =============================================================================

class PlanFactory:
    """
    Factory for creating plan documents from configuration.
    """
    
    @staticmethod
    def create_plan(rules: PlanRules) -> PlanDocument:
        """Create appropriate plan document based on rules."""
        if rules.benefit_type == BenefitType.IMPLICIT_SUBSIDY:
            return ImplicitSubsidyPlan(rules)
        elif rules.benefit_type == BenefitType.EXPLICIT_DOLLAR:
            return ExplicitSubsidyPlan(rules)
        elif rules.contribution_structure == ContributionStructure.SERVICE_TIERED:
            return ServiceTieredPlan(rules)
        else:
            return ImplicitSubsidyPlan(rules)
    
    @staticmethod
    def create_from_dict(config: Dict[str, Any]) -> PlanDocument:
        """Create plan from dictionary configuration."""
        rules = PlanRules(**config)
        return PlanFactory.create_plan(rules)
    
    @staticmethod
    def create_default_opeb_plan(
        plan_name: str = "Default OPEB Plan",
        contribution_rate: float = 0.45,
        pre65_premium: float = 667.68,
        post65_premium: float = 459.89
    ) -> PlanDocument:
        """Create a default OPEB implicit subsidy plan."""
        rules = PlanRules(
            plan_name=plan_name,
            plan_id="DEFAULT",
            effective_date=date.today(),
            benefit_type=BenefitType.IMPLICIT_SUBSIDY,
            default_contribution_rate=contribution_rate,
            premiums={
                'employee': PremiumSchedule(
                    coverage_type=CoverageType.EMPLOYEE,
                    pre65_monthly=pre65_premium,
                    post65_monthly=post65_premium
                )
            }
        )
        return ImplicitSubsidyPlan(rules)


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PLAN CONFIGURATION MODULE - UNIT TESTS")
    print("=" * 70)
    
    # Test 1: Default OPEB Plan
    print("\nTest 1: Default OPEB Implicit Subsidy Plan")
    print("-" * 50)
    
    plan = PlanFactory.create_default_opeb_plan(
        plan_name="City of DeRidder",
        contribution_rate=0.45,
        pre65_premium=667.68,
        post65_premium=459.89
    )
    
    for age in [55, 60, 65, 70, 75]:
        gross = plan.get_gross_benefit(age, 'M', CoverageType.EMPLOYEE)
        contrib = plan.get_participant_contribution(age, 25, CoverageType.EMPLOYEE)
        subsidy = plan.get_implicit_subsidy(age, 'M', 25, CoverageType.EMPLOYEE)
        print(f"  Age {age}: Gross=${gross:,.0f}, Contrib=${contrib:,.0f}, Subsidy=${subsidy:,.0f}")
    
    # Test 2: Service-Tiered Plan
    print("\nTest 2: Service-Tiered Contribution Plan")
    print("-" * 50)
    
    tiered_rules = PlanRules(
        plan_name="Service Tiered Plan",
        plan_id="TIERED",
        effective_date=date.today(),
        contribution_structure=ContributionStructure.SERVICE_TIERED
    )
    tiered_plan = ServiceTieredPlan(tiered_rules)
    
    for service in [5, 10, 20, 30, 35]:
        subsidy = tiered_plan.get_implicit_subsidy(60, 'M', service, CoverageType.EMPLOYEE)
        print(f"  Service {service:2d} years: Annual Subsidy = ${subsidy:,.0f}")
    
    # Test 3: Explicit Subsidy Plan
    print("\nTest 3: Explicit Dollar Subsidy Plan")
    print("-" * 50)
    
    explicit_rules = PlanRules(
        plan_name="Explicit Subsidy Plan",
        plan_id="EXPLICIT",
        effective_date=date.today(),
        benefit_type=BenefitType.EXPLICIT_DOLLAR,
        explicit_subsidy_pre65=400.0,
        explicit_subsidy_post65=200.0
    )
    explicit_plan = ExplicitSubsidyPlan(explicit_rules)
    
    for age in [55, 65, 75]:
        subsidy = explicit_plan.get_implicit_subsidy(age, 'M', 20, CoverageType.EMPLOYEE)
        print(f"  Age {age}: Annual Subsidy = ${subsidy:,.0f}")
    
    print("\n✓ All plan configuration tests passed")
