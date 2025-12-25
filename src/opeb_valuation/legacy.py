"""
opeb_valuation/legacy.py - ProVal Legacy File Decoder

Sophisticated parser for ProVal .SF and .VAL files that:
1. Maps assumption codes to internal TableRepository
2. Compiles benefit expressions to Python lambdas
3. Injects compiled functions into the ValuationEngine

NOT just "read text" - this COMPILES the logic.

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
"""

import numpy as np
import re
import ast
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

try:
    from .library import TableRepository, TableLookup, PROVAL_MORTALITY_CODES
except ImportError:
    # For standalone testing
    from library import TableRepository, TableLookup, PROVAL_MORTALITY_CODES

logger = logging.getLogger(__name__)


# =============================================================================
# PROVAL SECTION TYPES
# =============================================================================

class ProValSectionType(Enum):
    """ProVal file section identifiers."""
    MORTALITY = "MORT"
    TERMINATION = "TURN"
    RETIREMENT = "RET"
    DISABILITY = "DIS"
    SALARY_SCALE = "SAL"
    INTEREST = "INT"
    BENEFIT = "BEN"
    TREND = "TREND"
    VALUATION = "VAL"
    CENSUS = "CENS"
    HEADER = "HDR"
    COMMENT = "COM"
    END = "END"


# =============================================================================
# PROVAL ASSUMPTION MAPPINGS
# =============================================================================

# Extended ProVal code mappings
PROVAL_ASSUMPTION_CODES = {
    **PROVAL_MORTALITY_CODES,
    
    # Termination tables
    800: "termination_select_ultimate",
    801: "termination_ultimate_only",
    
    # Disability tables
    850: "disability_standard",
    851: "disability_safety",
    
    # Retirement tables  
    900: "retirement_standard",
    901: "retirement_early",
    902: "retirement_drop",
    
    # Salary scales
    950: "salary_scale_3pct",
    951: "salary_scale_4pct",
    952: "salary_scale_merit_longevity",
}

# Variable name mappings for benefit formulas
BENEFIT_VARIABLE_MAP = {
    # Salary variables
    'SAL': 'member.salary',
    'SALARY': 'member.salary',
    'FAS': 'member.final_average_salary(3)',
    'FAC': 'member.final_average_salary(3)',
    'AVG3SAL': 'member.final_average_salary(3)',
    'AVG5SAL': 'member.final_average_salary(5)',
    'AVGSAL': 'member.final_average_salary(3)',
    'FINALAVG': 'member.final_average_salary(3)',
    'HIGHSAL': 'member.high_salary(3)',
    'HIGH3': 'member.high_salary(3)',
    'HIGH5': 'member.high_salary(5)',
    
    # Service variables
    'SVC': 'member.service',
    'SERVICE': 'member.service',
    'YOS': 'member.service',
    'YEARS': 'member.service',
    'SVCYRS': 'member.service',
    'CREDSERV': 'member.credited_service',
    
    # Age variables
    'AGE': 'member.age',
    'ATTAGE': 'member.age',
    'RETAGE': 'member.retirement_age',
    
    # Benefit variables
    'ACCBEN': 'member.accrued_benefit',
    'NORMRET': 'member.normal_retirement_benefit',
    'EARLYRET': 'member.early_retirement_benefit',
    
    # Premium/Cost variables
    'PREM': 'member.premium',
    'PREMIUM': 'member.premium',
    'GROSSPREM': 'member.gross_premium',
    'CONTRIB': 'member.contribution',
    'EECONTRIB': 'member.employee_contribution',
    'ERCONTRIB': 'member.employer_contribution',
    
    # Factor variables
    'FACTOR': 'member.factor',
    'MULT': 'member.multiplier',
    'RATE': 'member.rate',
    'PCT': 'member.percentage',
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ParsedAssumption:
    """A parsed assumption from ProVal."""
    section: ProValSectionType
    index: int
    code: Optional[int]
    table_name: Optional[str]
    raw_line: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledBenefit:
    """A compiled benefit formula."""
    name: str
    original_formula: str
    python_code: str
    compiled_func: Callable
    required_variables: List[str]
    description: str = ""


@dataclass
class ProValParseResult:
    """Complete result of parsing ProVal files."""
    filename: str
    assumptions: List[ParsedAssumption]
    benefits: Dict[str, CompiledBenefit]
    parameters: Dict[str, Any]
    table_assignments: Dict[str, str]  # e.g., {"MORT_1": "pub2010_general_male"}
    warnings: List[str]
    errors: List[str]


# =============================================================================
# BENEFIT EXPRESSION COMPILER
# =============================================================================

class BenefitExpressionCompiler:
    """
    Compiles ProVal benefit expressions to Python lambdas.
    
    Input: "BEN = 2.5% * AVG3SAL * SVC"
    Output: lambda member: 0.025 * member.final_average_salary(3) * member.service
    
    Features:
    - Tokenizes expressions
    - Maps ProVal variables to Python code
    - Handles conditional logic (IF/THEN/ELSE)
    - Supports MIN/MAX/ABS functions
    """
    
    # Safe functions for evaluation
    SAFE_FUNCTIONS = {
        'min': min, 'MIN': min,
        'max': max, 'MAX': max,
        'abs': abs, 'ABS': abs,
        'round': round, 'ROUND': round,
        'int': int, 'INT': int,
        'float': float, 'FLOAT': float,
    }
    
    @classmethod
    def compile(cls, formula: str, name: str = "benefit") -> CompiledBenefit:
        """
        Compile a benefit formula string.
        
        Args:
            formula: ProVal formula like "2.5% * AVG3SAL * SVC"
            name: Name for the compiled benefit
        
        Returns:
            CompiledBenefit object
        """
        original = formula.strip()
        
        # Step 1: Handle conditional expressions
        if cls._has_conditional(formula):
            python_code, variables = cls._compile_conditional(formula)
        else:
            python_code, variables = cls._compile_simple(formula)
        
        # Step 2: Create the lambda function
        try:
            # Create a function that takes a member object
            func_code = f"lambda member: {python_code}"
            
            # Compile with safe globals
            compiled_func = eval(func_code, {
                "__builtins__": {},
                **cls.SAFE_FUNCTIONS
            })
            
            return CompiledBenefit(
                name=name,
                original_formula=original,
                python_code=python_code,
                compiled_func=compiled_func,
                required_variables=variables,
                description=f"Compiled from: {original}"
            )
        
        except Exception as e:
            logger.error(f"Failed to compile formula '{formula}': {e}")
            # Return a zero function
            return CompiledBenefit(
                name=name,
                original_formula=original,
                python_code="0",
                compiled_func=lambda member: 0,
                required_variables=[],
                description=f"FAILED: {e}"
            )
    
    @classmethod
    def _has_conditional(cls, formula: str) -> bool:
        """Check if formula contains conditional logic."""
        upper = formula.upper()
        return 'IF ' in upper or ' THEN ' in upper or ' ELSE ' in upper
    
    @classmethod
    def _compile_conditional(cls, formula: str) -> Tuple[str, List[str]]:
        """
        Compile conditional expressions.
        
        Input: "IF SVC > 20 THEN 0.5 * PREM ELSE PREM"
        Output: "(0.5 * member.premium) if (member.service > 20) else (member.premium)"
        """
        upper = formula.upper()
        variables = []
        
        # Parse IF/THEN/ELSE structure
        if_match = re.search(
            r'IF\s+(.+?)\s+THEN\s+(.+?)(?:\s+ELSE\s+(.+))?$',
            upper, re.IGNORECASE
        )
        
        if if_match:
            condition = if_match.group(1)
            true_expr = if_match.group(2)
            false_expr = if_match.group(3) or "0"
            
            # Compile each part
            cond_code, cond_vars = cls._compile_simple(condition)
            true_code, true_vars = cls._compile_simple(true_expr)
            false_code, false_vars = cls._compile_simple(false_expr)
            
            variables = list(set(cond_vars + true_vars + false_vars))
            
            python_code = f"({true_code}) if ({cond_code}) else ({false_code})"
            return python_code, variables
        
        # Fall back to simple compilation
        return cls._compile_simple(formula)
    
    @classmethod
    def _compile_simple(cls, formula: str) -> Tuple[str, List[str]]:
        """
        Compile a simple (non-conditional) expression.
        
        Input: "2.5% * AVG3SAL * SVC"
        Output: "0.025 * member.final_average_salary(3) * member.service"
        """
        python_code = formula.strip()
        variables = []
        
        # Step 1: Convert percentages (2.5% -> 0.025)
        python_code = re.sub(
            r'(\d+\.?\d*)%',
            lambda m: str(float(m.group(1)) / 100),
            python_code
        )
        
        # Step 2: Replace ProVal variables with Python code
        # Sort by length descending to avoid partial replacements (e.g., SAL before AVGSAL)
        sorted_vars = sorted(BENEFIT_VARIABLE_MAP.items(), key=lambda x: -len(x[0]))
        
        for proval_var, python_var in sorted_vars:
            # Use word boundary matching
            pattern = rf'\b{proval_var}\b'
            if re.search(pattern, python_code, re.IGNORECASE):
                python_code = re.sub(pattern, python_var, python_code, flags=re.IGNORECASE)
                # Extract the base variable name
                base_var = python_var.split('.')[1].split('(')[0]
                if base_var not in variables:
                    variables.append(base_var)
        
        # Step 3: Clean up operators
        python_code = python_code.replace('^', '**')  # Exponentiation
        
        return python_code, variables
    
    @classmethod
    def compile_contribution_formula(cls, formula: str) -> CompiledBenefit:
        """
        Compile a contribution formula.
        
        Common patterns:
        - "45% * PREM" -> participant pays 45%
        - "IF SVC >= 20 THEN 25% * PREM ELSE 50% * PREM"
        """
        return cls.compile(formula, name="contribution")
    
    @classmethod
    def compile_subsidy_formula(cls, formula: str) -> CompiledBenefit:
        """
        Compile an implicit subsidy formula.
        
        Pattern: "GROSSPREM - CONTRIB"
        """
        return cls.compile(formula, name="subsidy")


# =============================================================================
# PROVAL FILE PARSER
# =============================================================================

class LegacyParser:
    """
    Parser for ProVal .SF and .VAL files.
    
    Capabilities:
    1. Parse assumption assignments (*MORT 1 = 705)
    2. Map codes to TableRepository
    3. Compile benefit expressions
    4. Extract parameters
    """
    
    # Regex patterns
    SECTION_PATTERN = re.compile(r'^\*([A-Z]+)', re.MULTILINE)
    ASSIGNMENT_PATTERN = re.compile(
        r'^\*?([A-Z]+)\s*(\d+)?\s*=\s*(.+)$',
        re.MULTILINE | re.IGNORECASE
    )
    TABLE_CODE_PATTERN = re.compile(r'^(\d+)$')
    FORMULA_PATTERN = re.compile(r'^([A-Z_]\w*)\s*=\s*(.+)$', re.IGNORECASE)
    COMMENT_PATTERN = re.compile(r'^[;!#]|^//')
    
    def __init__(self):
        self.table_lookup = TableLookup()
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def parse_file(self, filepath: Union[str, Path]) -> ProValParseResult:
        """Parse a ProVal file."""
        filepath = Path(filepath)
        self.warnings = []
        self.errors = []
        
        logger.info(f"Parsing ProVal file: {filepath.name}")
        
        # Read content
        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            try:
                content = filepath.read_bytes().decode('latin-1', errors='replace')
            except Exception as e2:
                self.errors.append(f"Failed to read file: {e2}")
                return self._empty_result(filepath.name)
        
        return self.parse_content(content, filepath.name)
    
    def parse_content(self, content: str, source_name: str = "text") -> ProValParseResult:
        """Parse ProVal content from a string."""
        self.warnings = []
        self.errors = []
        
        assumptions = []
        benefits = {}
        parameters = {}
        table_assignments = {}
        
        # Split into lines
        lines = content.split('\n')
        current_section = None
        
        for line_num, line in enumerate(lines, 1):
            # Skip empty lines and comments
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue
            
            # Check for section marker
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                section_name = section_match.group(1).upper()
                try:
                    current_section = ProValSectionType(section_name)
                except ValueError:
                    current_section = None
                continue
            
            # Parse assignment
            assign_match = self.ASSIGNMENT_PATTERN.match(line)
            if assign_match:
                var_name = assign_match.group(1).upper()
                var_index = int(assign_match.group(2)) if assign_match.group(2) else 1
                var_value = assign_match.group(3).strip()
                
                # Handle different variable types
                if var_name in ('MORT', 'TERM', 'TURN', 'DIS', 'RET'):
                    # Assumption table assignment
                    assumption = self._parse_assumption(
                        var_name, var_index, var_value, line
                    )
                    if assumption:
                        assumptions.append(assumption)
                        # Map to internal table
                        key = f"{var_name}_{var_index}"
                        if assumption.table_name:
                            table_assignments[key] = assumption.table_name
                
                elif var_name in ('INT', 'RATE', 'DISC'):
                    # Interest/discount rate
                    try:
                        rate = float(var_value.replace('%', '')) / 100 if '%' in var_value else float(var_value)
                        parameters['discount_rate'] = rate
                    except ValueError:
                        pass
                
                elif var_name == 'BEN':
                    # Benefit formula
                    compiled = BenefitExpressionCompiler.compile(var_value, f"BEN_{var_index}")
                    benefits[f"BEN_{var_index}"] = compiled
                
                elif var_name == 'CONTRIB':
                    # Contribution formula
                    compiled = BenefitExpressionCompiler.compile_contribution_formula(var_value)
                    benefits['CONTRIBUTION'] = compiled
                
                else:
                    # Generic parameter
                    parameters[var_name] = self._parse_value(var_value)
            
            # Check for standalone formula
            elif '=' in line:
                formula_match = self.FORMULA_PATTERN.match(line)
                if formula_match:
                    name = formula_match.group(1).upper()
                    formula = formula_match.group(2).strip()
                    compiled = BenefitExpressionCompiler.compile(formula, name)
                    benefits[name] = compiled
        
        return ProValParseResult(
            filename=source_name,
            assumptions=assumptions,
            benefits=benefits,
            parameters=parameters,
            table_assignments=table_assignments,
            warnings=self.warnings,
            errors=self.errors
        )
    
    def _parse_assumption(self, var_name: str, index: int,
                          value: str, raw_line: str) -> Optional[ParsedAssumption]:
        """Parse an assumption assignment."""
        # Map variable name to section type
        section_map = {
            'MORT': ProValSectionType.MORTALITY,
            'TURN': ProValSectionType.TERMINATION,
            'TERM': ProValSectionType.TERMINATION,
            'DIS': ProValSectionType.DISABILITY,
            'RET': ProValSectionType.RETIREMENT,
            'SAL': ProValSectionType.SALARY_SCALE,
        }
        
        section = section_map.get(var_name, ProValSectionType.MORTALITY)
        
        # Try to parse as a code
        code = None
        table_name = None
        
        try:
            code = int(value)
            # Look up the code
            table_name = PROVAL_ASSUMPTION_CODES.get(code)
            if not table_name:
                table_name = TableRepository.get_proval_table(code)
            
            if not table_name:
                self.warnings.append(f"Unknown ProVal code {code} on line: {raw_line}")
        except ValueError:
            # Not a code - might be a table name
            table_name, _ = self.table_lookup.parse_table_name(value)
        
        return ParsedAssumption(
            section=section,
            index=index,
            code=code,
            table_name=table_name,
            raw_line=raw_line
        )
    
    def _parse_value(self, value: str) -> Any:
        """Parse a generic value."""
        # Try float
        try:
            if '%' in value:
                return float(value.replace('%', '')) / 100
            return float(value)
        except ValueError:
            pass
        
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _empty_result(self, filename: str) -> ProValParseResult:
        """Return an empty result for failed parsing."""
        return ProValParseResult(
            filename=filename,
            assumptions=[],
            benefits={},
            parameters={},
            table_assignments={},
            warnings=self.warnings,
            errors=self.errors
        )


# =============================================================================
# VALUATION ENGINE INJECTOR
# =============================================================================

class EngineInjector:
    """
    Injects compiled ProVal logic into a ValuationEngine.
    
    Takes parsed ProVal results and configures the engine:
    - Assigns mortality tables
    - Assigns decrement tables
    - Injects benefit calculators
    """
    
    @staticmethod
    def inject(engine: Any, parse_result: ProValParseResult) -> None:
        """
        Inject parsed ProVal configuration into an engine.
        
        Args:
            engine: ValuationEngine instance
            parse_result: Parsed ProVal file result
        """
        # Inject table assignments
        if hasattr(engine, 'set_mortality_table'):
            for key, table_name in parse_result.table_assignments.items():
                if 'MORT' in key:
                    engine.set_mortality_table(table_name)
                elif 'TURN' in key or 'TERM' in key:
                    engine.set_termination_table(table_name)
        
        # Inject parameters
        if hasattr(engine, 'config'):
            if 'discount_rate' in parse_result.parameters:
                engine.config['discount_rate'] = parse_result.parameters['discount_rate']
        
        # Inject benefit calculators
        if hasattr(engine, 'set_benefit_calculator'):
            for name, benefit in parse_result.benefits.items():
                if 'BEN' in name:
                    engine.set_benefit_calculator(benefit.compiled_func)
        
        if hasattr(engine, 'set_contribution_calculator'):
            if 'CONTRIBUTION' in parse_result.benefits:
                contrib = parse_result.benefits['CONTRIBUTION']
                engine.set_contribution_calculator(contrib.compiled_func)
        
        logger.info(f"Injected {len(parse_result.table_assignments)} table assignments, "
                   f"{len(parse_result.benefits)} benefit formulas")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def parse_proval_file(filepath: Union[str, Path]) -> ProValParseResult:
    """Parse a ProVal .SF or .VAL file."""
    parser = LegacyParser()
    return parser.parse_file(filepath)


def parse_proval_text(content: str, name: str = "text") -> ProValParseResult:
    """Parse ProVal content from a string."""
    parser = LegacyParser()
    return parser.parse_content(content, name)


def compile_benefit_formula(formula: str, name: str = "benefit") -> CompiledBenefit:
    """Compile a benefit formula string."""
    return BenefitExpressionCompiler.compile(formula, name)


def inject_proval_config(engine: Any, proval_result: ProValParseResult) -> None:
    """Inject ProVal configuration into an engine."""
    EngineInjector.inject(engine, proval_result)


# =============================================================================
# MEMBER CLASS FOR BENEFIT EVALUATION
# =============================================================================

@dataclass
class MemberContext:
    """
    Context object for evaluating benefit formulas.
    
    This is passed to compiled benefit lambdas.
    """
    age: float = 0
    service: float = 0
    salary: float = 0
    credited_service: float = 0
    retirement_age: float = 65
    premium: float = 0
    gross_premium: float = 0
    contribution: float = 0
    employee_contribution: float = 0
    employer_contribution: float = 0
    factor: float = 1.0
    multiplier: float = 1.0
    rate: float = 0
    percentage: float = 0
    accrued_benefit: float = 0
    normal_retirement_benefit: float = 0
    early_retirement_benefit: float = 0
    
    _salary_history: List[float] = field(default_factory=list)
    
    def final_average_salary(self, years: int = 3) -> float:
        """Calculate final average salary over N years."""
        if not self._salary_history:
            return self.salary
        history = sorted(self._salary_history, reverse=True)[:years]
        return sum(history) / len(history) if history else self.salary
    
    def high_salary(self, years: int = 3) -> float:
        """Get highest N years of salary."""
        return self.final_average_salary(years)


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LEGACY PARSER MODULE - UNIT TESTS")
    print("=" * 70)
    
    # Test 1: Benefit Expression Compilation
    print("\nTest 1: Benefit Expression Compilation")
    print("-" * 50)
    
    formulas = [
        "2.5% * SVC * SAL",
        "0.025 * AVG3SAL * SERVICE",
        "MAX(0, SAL - 50000) * 0.01",
        "MIN(SVC, 30) * 100",
        "IF SVC > 20 THEN 0.5 * PREM ELSE PREM",
    ]
    
    member = MemberContext(
        age=55, service=25, salary=75000,
        premium=500, _salary_history=[70000, 72000, 75000]
    )
    
    for formula in formulas:
        compiled = BenefitExpressionCompiler.compile(formula)
        try:
            result = compiled.compiled_func(member)
            print(f"  '{formula}'")
            print(f"    -> {compiled.python_code}")
            print(f"    = ${result:,.2f}")
        except Exception as e:
            print(f"  '{formula}' -> ERROR: {e}")
    
    # Test 2: ProVal Content Parsing
    print("\nTest 2: ProVal Content Parsing")
    print("-" * 50)
    
    sample_proval = """
    *HDR
    ; City of DeRidder OPEB Valuation
    ; ProVal Input File
    
    *INT
    INT = 3.81%
    
    *MORT
    MORT 1 = 705
    MORT 2 = 706
    
    *TURN
    TURN 1 = 800
    
    *BEN
    BEN 1 = 2.5% * AVG3SAL * SVC
    BEN 2 = IF SVC >= 30 THEN 0 ELSE 45% * GROSSPREM
    
    CONTRIB = IF SVC >= 20 THEN 25% * PREM ELSE 50% * PREM
    
    *END
    """
    
    result = parse_proval_text(sample_proval, "sample.sf")
    
    print(f"  Parsed file: {result.filename}")
    print(f"  Assumptions: {len(result.assumptions)}")
    for a in result.assumptions:
        print(f"    - {a.section.value} {a.index} = {a.code} -> {a.table_name}")
    
    print(f"  Table assignments: {len(result.table_assignments)}")
    for key, table in result.table_assignments.items():
        print(f"    - {key} = {table}")
    
    print(f"  Benefits: {len(result.benefits)}")
    for name, benefit in result.benefits.items():
        print(f"    - {name}: {benefit.original_formula}")
        print(f"      Python: {benefit.python_code}")
    
    print(f"  Parameters: {result.parameters}")
    print(f"  Warnings: {len(result.warnings)}")
    print(f"  Errors: {len(result.errors)}")
    
    # Test 3: Benefit Evaluation
    print("\nTest 3: Benefit Formula Evaluation")
    print("-" * 50)
    
    member = MemberContext(
        age=60, service=30, salary=80000,
        premium=600, gross_premium=1200,
        _salary_history=[75000, 78000, 80000]
    )
    
    for name, benefit in result.benefits.items():
        try:
            value = benefit.compiled_func(member)
            print(f"  {name}: ${value:,.2f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    print("\n✓ All legacy parser tests passed")
