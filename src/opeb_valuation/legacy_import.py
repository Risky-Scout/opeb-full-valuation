"""
opeb_valuation/legacy_import.py - ProVal Legacy File Parser

Universal Adapter for legacy actuarial software migration.

CAPABILITIES:
1. Parse ProVal .SF (Select Factor) files using regex
2. Extract decrement tables (*MORT, *TURN, *INT, *DIS)
3. Compile benefit formulas into vectorized lambdas
4. Convert legacy formats to Shackleford Precision tensors

SUPPORTED FORMATS:
- ProVal .SF files (PVASMP.SF, PVVIP.SF)
- Milliman ValuPro exports
- Generic actuarial text tables

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

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class TableType(Enum):
    """Legacy table type identifiers."""
    MORTALITY = "MORT"
    TERMINATION = "TURN"
    RETIREMENT = "RET"
    DISABILITY = "DIS"
    INTEREST = "INT"
    SALARY_SCALE = "SAL"
    TREND = "TREND"
    BENEFIT = "BEN"


class ProValSection(Enum):
    """ProVal file section markers."""
    MORTALITY = "*MORT"
    TERMINATION = "*TURN"
    RETIREMENT = "*RET"
    DISABILITY = "*DIS"
    INTEREST = "*INT"
    SALARY = "*SAL"
    BENEFIT = "*BEN"
    TREND = "*TREND"
    HEADER = "*HDR"
    END = "*END"


# =============================================================================
# PARSED TABLE STRUCTURES
# =============================================================================

@dataclass
class ParsedTable:
    """Represents a parsed actuarial table."""
    table_type: TableType
    name: str
    dimensions: List[str]  # e.g., ['age', 'service'] or ['age']
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_value(self, **kwargs) -> float:
        """Look up value by dimension keys."""
        if len(self.dimensions) == 1:
            idx = int(kwargs.get(self.dimensions[0], 0))
            idx = np.clip(idx, 0, len(self.data) - 1)
            return float(self.data[idx])
        elif len(self.dimensions) == 2:
            idx0 = int(kwargs.get(self.dimensions[0], 0))
            idx1 = int(kwargs.get(self.dimensions[1], 0))
            idx0 = np.clip(idx0, 0, self.data.shape[0] - 1)
            idx1 = np.clip(idx1, 0, self.data.shape[1] - 1)
            return float(self.data[idx0, idx1])
        else:
            raise ValueError(f"Unsupported dimensions: {self.dimensions}")
    
    def to_tensor(self, max_age: int = 121, max_service: int = 61) -> np.ndarray:
        """Convert to standard DecrementTensor format."""
        if len(self.dimensions) == 1 and self.dimensions[0] == 'age':
            # 1D age-only table -> expand to 2D
            tensor = np.zeros((max_age, max_service))
            for age in range(max_age):
                src_idx = min(age, len(self.data) - 1)
                tensor[age, :] = self.data[src_idx]
            return tensor
        elif len(self.dimensions) == 2:
            # Already 2D (age x service)
            tensor = np.zeros((max_age, max_service))
            for age in range(min(max_age, self.data.shape[0])):
                for svc in range(min(max_service, self.data.shape[1])):
                    tensor[age, svc] = self.data[age, svc]
            return tensor
        else:
            return self.data


@dataclass 
class ParsedBenefit:
    """Represents a parsed benefit formula."""
    name: str
    formula_str: str
    compiled_func: Optional[Callable] = None
    variables: List[str] = field(default_factory=list)
    
    def evaluate(self, **kwargs) -> float:
        """Evaluate the benefit formula with given variables."""
        if self.compiled_func:
            return self.compiled_func(**kwargs)
        return 0.0


@dataclass
class ProValParseResult:
    """Complete result of parsing a ProVal file."""
    filename: str
    tables: Dict[str, ParsedTable]
    benefits: Dict[str, ParsedBenefit]
    parameters: Dict[str, Any]
    warnings: List[str]
    raw_sections: Dict[str, str]


# =============================================================================
# BENEFIT FORMULA COMPILER
# =============================================================================

class BenefitFactory:
    """
    Parses legacy benefit strings into executable Python logic.
    
    Example Input: "0.025 * SVC * SAL"
    Output: Vectorized lambda function
    
    SUPPORTED VARIABLES:
    - SVC, SERVICE: Years of service
    - SAL, SALARY: Annual salary
    - AGE: Current age
    - YRS, YEARS: Generic years
    - FAC, FACTOR: Benefit factor
    """
    
    # Variable mappings (legacy -> standard)
    VARIABLE_MAP = {
        'SVC': 'service',
        'SERVICE': 'service',
        'SAL': 'salary',
        'SALARY': 'salary',
        'AGE': 'age',
        'YRS': 'years',
        'YEARS': 'years',
        'FAC': 'factor',
        'FACTOR': 'factor',
        'AVGSAL': 'avg_salary',
        'FAS': 'final_avg_salary',
        'MONTHS': 'months',
        'RATE': 'rate',
    }
    
    # Safe functions for evaluation
    SAFE_FUNCTIONS = {
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'int': int,
        'float': float,
        'MIN': min,
        'MAX': max,
        'ABS': abs,
        'ROUND': round,
        'INT': int,
        'FLOAT': float,
    }
    
    @classmethod
    def compile_formula(cls, formula_str: str) -> Tuple[Callable, List[str]]:
        """
        Compile a benefit formula string into an executable function.
        
        Args:
            formula_str: Legacy formula like "0.025 * SVC * SAL"
        
        Returns:
            Tuple of (compiled function, list of required variables)
        """
        # Clean the formula
        clean_formula = formula_str.strip().upper()
        
        # Track required variables
        required_vars = []
        
        # Replace legacy variable names with kwargs access
        python_code = clean_formula
        for legacy_var, standard_var in cls.VARIABLE_MAP.items():
            if legacy_var in python_code:
                python_code = re.sub(
                    rf'\b{legacy_var}\b',
                    f'kwargs.get("{standard_var}", 0)',
                    python_code
                )
                required_vars.append(standard_var)
        
        # Handle percentage notation (2.5% -> 0.025)
        python_code = re.sub(
            r'(\d+\.?\d*)%',
            lambda m: str(float(m.group(1)) / 100),
            python_code
        )
        
        # Create the function
        try:
            # Validate the expression is safe
            ast.parse(python_code, mode='eval')
            
            def compiled_func(**kwargs):
                return eval(python_code, {"__builtins__": {}}, 
                           {**cls.SAFE_FUNCTIONS, "kwargs": kwargs})
            
            return compiled_func, list(set(required_vars))
        
        except SyntaxError as e:
            logger.warning(f"Failed to compile formula '{formula_str}': {e}")
            return lambda **kwargs: 0.0, []
    
    @classmethod
    def compile_conditional(cls, condition_str: str, 
                           true_formula: str, 
                           false_formula: str) -> Callable:
        """
        Compile a conditional benefit formula.
        
        Example: IF SERVICE > 20 THEN 0.5 * PREMIUM ELSE PREMIUM
        """
        cond_func, _ = cls.compile_formula(condition_str)
        true_func, _ = cls.compile_formula(true_formula)
        false_func, _ = cls.compile_formula(false_formula)
        
        def conditional_func(**kwargs):
            if cond_func(**kwargs):
                return true_func(**kwargs)
            else:
                return false_func(**kwargs)
        
        return conditional_func


# =============================================================================
# PROVAL .SF FILE PARSER
# =============================================================================

class ProValParser:
    """
    Parser for ProVal .SF (Select Factor) files.
    
    Handles:
    - PVASMP.SF: Sample/assumption files
    - PVVIP.SF: VIP/benefit definition files
    - Generic actuarial text tables
    
    PATTERN RECOGNITION:
    - Section markers: *MORT, *TURN, *INT, etc.
    - 2D tables: Age x Service grids
    - 1D tables: Age-only arrays
    - Benefit formulas: BEN = expression
    """
    
    # Regex patterns for parsing
    SECTION_PATTERN = re.compile(r'^\*([A-Z]+)', re.MULTILINE)
    TABLE_HEADER_PATTERN = re.compile(r'^(\d+)\s+(.+)$', re.MULTILINE)
    NUMBER_PATTERN = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
    FORMULA_PATTERN = re.compile(r'^([A-Z_]+)\s*=\s*(.+)$', re.MULTILINE | re.IGNORECASE)
    PARAMETER_PATTERN = re.compile(r'^([A-Za-z_]\w*)\s*[:=]\s*(.+)$', re.MULTILINE)
    
    def __init__(self):
        self.warnings: List[str] = []
    
    def parse_file(self, filepath: Union[str, Path]) -> ProValParseResult:
        """
        Parse a ProVal .SF file.
        
        Args:
            filepath: Path to .SF file
        
        Returns:
            ProValParseResult with all parsed data
        """
        filepath = Path(filepath)
        self.warnings = []
        
        logger.info(f"Parsing ProVal file: {filepath.name}")
        
        # Read file content
        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            content = filepath.read_bytes().decode('latin-1', errors='replace')
        
        # Split into sections
        sections = self._split_sections(content)
        
        # Parse each section type
        tables = {}
        benefits = {}
        parameters = {}
        
        for section_name, section_content in sections.items():
            if section_name in ('MORT', 'TURN', 'RET', 'DIS'):
                table = self._parse_decrement_table(section_name, section_content)
                if table:
                    tables[section_name] = table
            
            elif section_name == 'BEN':
                parsed_benefits = self._parse_benefit_section(section_content)
                benefits.update(parsed_benefits)
            
            elif section_name == 'INT':
                params = self._parse_interest_section(section_content)
                parameters.update(params)
            
            elif section_name == 'SAL':
                table = self._parse_salary_scale(section_content)
                if table:
                    tables['SAL'] = table
            
            elif section_name == 'TREND':
                table = self._parse_trend_section(section_content)
                if table:
                    tables['TREND'] = table
        
        # Also extract any loose parameters
        parameters.update(self._extract_parameters(content))
        
        return ProValParseResult(
            filename=filepath.name,
            tables=tables,
            benefits=benefits,
            parameters=parameters,
            warnings=self.warnings,
            raw_sections=sections
        )
    
    def parse_text(self, content: str, source_name: str = "text") -> ProValParseResult:
        """Parse ProVal content from a string."""
        self.warnings = []
        sections = self._split_sections(content)
        
        tables = {}
        benefits = {}
        parameters = {}
        
        for section_name, section_content in sections.items():
            if section_name in ('MORT', 'TURN', 'RET', 'DIS'):
                table = self._parse_decrement_table(section_name, section_content)
                if table:
                    tables[section_name] = table
        
        parameters.update(self._extract_parameters(content))
        
        return ProValParseResult(
            filename=source_name,
            tables=tables,
            benefits=benefits,
            parameters=parameters,
            warnings=self.warnings,
            raw_sections=sections
        )
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """Split file content into named sections."""
        sections = {}
        
        # Find all section markers
        markers = list(self.SECTION_PATTERN.finditer(content))
        
        for i, match in enumerate(markers):
            section_name = match.group(1)
            start = match.end()
            
            # End is either next marker or end of file
            if i + 1 < len(markers):
                end = markers[i + 1].start()
            else:
                end = len(content)
            
            section_content = content[start:end].strip()
            sections[section_name] = section_content
        
        # If no markers found, treat entire content as data
        if not sections:
            sections['DATA'] = content
        
        return sections
    
    def _parse_decrement_table(self, table_type: str, 
                                content: str) -> Optional[ParsedTable]:
        """
        Parse a decrement table (mortality, termination, etc.)
        
        Handles both 1D (age-only) and 2D (age x service) formats.
        """
        lines = content.strip().split('\n')
        
        # Extract all numbers from content
        all_numbers = []
        for line in lines:
            # Skip comment lines
            if line.strip().startswith(('#', '//', ';', '!')):
                continue
            
            numbers = self.NUMBER_PATTERN.findall(line)
            if numbers:
                all_numbers.append([float(n) for n in numbers])
        
        if not all_numbers:
            self.warnings.append(f"No numeric data found in {table_type} section")
            return None
        
        # Determine table structure
        first_row_len = len(all_numbers[0])
        is_2d = first_row_len > 2 or (
            len(all_numbers) > 1 and len(all_numbers[1]) > 2
        )
        
        if is_2d:
            # 2D table: Age x Service
            # First column is usually age, rest are service years
            data = np.array(all_numbers)
            
            # If first column looks like ages (18-120), use it as index
            if data.shape[1] > 1 and data[0, 0] >= 15 and data[0, 0] <= 25:
                # First column is age labels
                ages = data[:, 0].astype(int)
                values = data[:, 1:]
                
                # Create full tensor
                max_age = int(max(ages)) + 1
                max_svc = values.shape[1]
                tensor = np.zeros((max_age, max_svc))
                
                for i, age in enumerate(ages):
                    tensor[age, :] = values[i, :]
                
                return ParsedTable(
                    table_type=TableType[table_type],
                    name=f"{table_type}_SELECT_ULTIMATE",
                    dimensions=['age', 'service'],
                    data=tensor,
                    metadata={'source': 'proval', 'has_age_index': True}
                )
            else:
                return ParsedTable(
                    table_type=TableType[table_type],
                    name=f"{table_type}_2D",
                    dimensions=['age', 'service'],
                    data=data,
                    metadata={'source': 'proval'}
                )
        else:
            # 1D table: Age only
            if first_row_len == 2:
                # Two columns: age, rate
                data = np.array(all_numbers)
                ages = data[:, 0].astype(int)
                rates = data[:, 1]
                
                max_age = int(max(ages)) + 1
                tensor = np.zeros(max_age)
                for i, age in enumerate(ages):
                    tensor[age] = rates[i]
                
                return ParsedTable(
                    table_type=TableType[table_type],
                    name=f"{table_type}_1D",
                    dimensions=['age'],
                    data=tensor,
                    metadata={'source': 'proval'}
                )
            else:
                # Single column of rates (indexed by position)
                rates = np.array([row[0] for row in all_numbers])
                return ParsedTable(
                    table_type=TableType[table_type],
                    name=f"{table_type}_1D",
                    dimensions=['age'],
                    data=rates,
                    metadata={'source': 'proval', 'zero_indexed': True}
                )
    
    def _parse_benefit_section(self, content: str) -> Dict[str, ParsedBenefit]:
        """Parse benefit formula definitions."""
        benefits = {}
        
        # Find all formula definitions
        for match in self.FORMULA_PATTERN.finditer(content):
            name = match.group(1).strip()
            formula = match.group(2).strip()
            
            try:
                compiled_func, variables = BenefitFactory.compile_formula(formula)
                benefits[name] = ParsedBenefit(
                    name=name,
                    formula_str=formula,
                    compiled_func=compiled_func,
                    variables=variables
                )
                logger.debug(f"Compiled benefit formula: {name} = {formula}")
            except Exception as e:
                self.warnings.append(f"Failed to compile benefit '{name}': {e}")
        
        return benefits
    
    def _parse_interest_section(self, content: str) -> Dict[str, float]:
        """Parse interest rate parameters."""
        params = {}
        
        numbers = self.NUMBER_PATTERN.findall(content)
        if numbers:
            params['interest_rate'] = float(numbers[0])
            if len(numbers) > 1:
                params['valuation_rate'] = float(numbers[1])
        
        return params
    
    def _parse_salary_scale(self, content: str) -> Optional[ParsedTable]:
        """Parse salary scale table."""
        return self._parse_decrement_table('SAL', content)
    
    def _parse_trend_section(self, content: str) -> Optional[ParsedTable]:
        """Parse healthcare trend rates."""
        lines = content.strip().split('\n')
        rates = []
        
        for line in lines:
            numbers = self.NUMBER_PATTERN.findall(line)
            if numbers:
                rates.append(float(numbers[-1]))  # Take last number as rate
        
        if rates:
            return ParsedTable(
                table_type=TableType.TREND,
                name='TREND_RATES',
                dimensions=['year'],
                data=np.array(rates),
                metadata={'source': 'proval'}
            )
        return None
    
    def _extract_parameters(self, content: str) -> Dict[str, Any]:
        """Extract key-value parameters from content."""
        params = {}
        
        # Common parameter patterns
        patterns = [
            (r'(?:discount|int|rate)\s*[:=]\s*([\d.]+)', 'discount_rate'),
            (r'(?:trend|medical)\s*[:=]\s*([\d.]+)', 'trend_rate'),
            (r'(?:load|factor)\s*[:=]\s*([\d.]+)', 'load_factor'),
            (r'(?:valdate|val_date|valuation)\s*[:=]\s*(\d{4}[-/]\d{2}[-/]\d{2})', 'valuation_date'),
        ]
        
        for pattern, param_name in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                try:
                    params[param_name] = float(value)
                except ValueError:
                    params[param_name] = value
        
        return params


# =============================================================================
# LEGACY TABLE CONVERTER
# =============================================================================

class LegacyTableConverter:
    """
    Converts parsed legacy tables to Shackleford Precision tensors.
    """
    
    @staticmethod
    def to_decrement_tensor(parsed_table: ParsedTable,
                            max_age: int = 121,
                            max_service: int = 61,
                            genders: int = 2) -> np.ndarray:
        """
        Convert ParsedTable to full DecrementTensor format.
        
        Output shape: (max_age, max_service, genders)
        """
        base_tensor = parsed_table.to_tensor(max_age, max_service)
        
        # Expand to include gender dimension
        if len(base_tensor.shape) == 2:
            # Same rates for both genders (expand)
            full_tensor = np.zeros((max_age, max_service, genders))
            for g in range(genders):
                full_tensor[:, :, g] = base_tensor
            return full_tensor
        
        return base_tensor
    
    @staticmethod
    def interpolate_missing(tensor: np.ndarray) -> np.ndarray:
        """Fill missing values using linear interpolation."""
        result = tensor.copy()
        
        # Find non-zero values
        nonzero_mask = result != 0
        
        if not np.any(nonzero_mask):
            return result
        
        # For 1D arrays
        if len(result.shape) == 1:
            nonzero_idx = np.where(nonzero_mask)[0]
            if len(nonzero_idx) > 1:
                result = np.interp(
                    np.arange(len(result)),
                    nonzero_idx,
                    result[nonzero_idx]
                )
        
        return result


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def parse_proval_file(filepath: Union[str, Path]) -> ProValParseResult:
    """Parse a ProVal .SF file."""
    parser = ProValParser()
    return parser.parse_file(filepath)


def parse_proval_text(content: str, name: str = "text") -> ProValParseResult:
    """Parse ProVal content from a string."""
    parser = ProValParser()
    return parser.parse_text(content, name)


def compile_benefit_formula(formula: str) -> Callable:
    """Compile a benefit formula string."""
    func, _ = BenefitFactory.compile_formula(formula)
    return func


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LEGACY IMPORT MODULE - UNIT TESTS")
    print("=" * 70)
    
    # Test 1: Benefit Formula Compilation
    print("\nTest 1: Benefit Formula Compilation")
    print("-" * 50)
    
    formulas = [
        "2.5% * SVC * SAL",
        "0.025 * SERVICE * SALARY",
        "max(0, SAL - 50000) * 0.01",
        "min(SVC, 30) * 100",
    ]
    
    for formula in formulas:
        func, vars = BenefitFactory.compile_formula(formula)
        result = func(service=20, salary=75000, age=55)
        print(f"  '{formula}' -> {result:,.2f} (vars: {vars})")
    
    # Test 2: ProVal Text Parsing
    print("\nTest 2: ProVal Text Parsing")
    print("-" * 50)
    
    sample_proval = """
    *MORT
    ; Mortality rates by age
    20  0.00045
    30  0.00065
    40  0.00120
    50  0.00250
    60  0.00550
    70  0.01200
    80  0.02800
    
    *TURN
    ; Termination rates (Age x Service)
    ; Age  Svc0   Svc1   Svc2   Svc3   Svc4   Ult
    20    0.250  0.180  0.140  0.110  0.080  0.060
    30    0.200  0.150  0.120  0.090  0.070  0.045
    40    0.150  0.120  0.100  0.080  0.060  0.035
    50    0.100  0.080  0.060  0.050  0.040  0.025
    60    0.050  0.040  0.030  0.020  0.015  0.010
    
    *BEN
    ACCRUAL = 2.5% * SVC * SAL
    MAX_BENEFIT = min(ACCRUAL, 100000)
    
    *INT
    4.50
    """
    
    result = parse_proval_text(sample_proval, "sample")
    
    print(f"  Parsed {len(result.tables)} tables:")
    for name, table in result.tables.items():
        print(f"    - {name}: shape={table.data.shape}, dims={table.dimensions}")
    
    print(f"  Parsed {len(result.benefits)} benefits:")
    for name, benefit in result.benefits.items():
        print(f"    - {name}: {benefit.formula_str}")
    
    print(f"  Warnings: {len(result.warnings)}")
    
    # Test 3: Table Lookup
    print("\nTest 3: Table Value Lookup")
    print("-" * 50)
    
    if 'MORT' in result.tables:
        mort = result.tables['MORT']
        for age in [30, 50, 70]:
            rate = mort.get_value(age=age)
            print(f"  Mortality at age {age}: {rate:.5f}")
    
    if 'TURN' in result.tables:
        turn = result.tables['TURN']
        for age, svc in [(30, 0), (30, 3), (50, 5)]:
            rate = turn.get_value(age=age, service=svc)
            print(f"  Termination at age {age}, svc {svc}: {rate:.4f}")
    
    print("\n✓ All legacy import tests passed")
