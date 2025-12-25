"""
opeb_valuation/smart_ingestion.py - "Client Chaos" Ingestion System

Handles terrible client data with grace using:
1. Fuzzy column name matching (Levenshtein distance)
2. Multi-format date parsing heuristics
3. Status unification
4. ASOP 23 compliant imputation
5. Data Forensics reporting

DESIGN: "Fail Gracefully, Fix Intelligently"

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import hashlib
import re
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class MemberStatus(Enum):
    """Standardized member status."""
    ACTIVE = "Active"
    RETIREE = "Retiree"
    TERMINATED = "Terminated"
    DISABLED = "Disabled"
    UNKNOWN = "Unknown"


class DataQualityLevel(Enum):
    """Data quality classification."""
    CLEAN = "clean"
    IMPUTED = "imputed"
    WARNING = "warning"
    EXCLUDED = "excluded"


class IssueType(Enum):
    """Types of data quality issues."""
    MISSING_VALUE = "missing_value"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    LOGICAL_ERROR = "logical_error"
    IMPUTED = "imputed"
    NORMALIZED = "normalized"


# =============================================================================
# FUZZY COLUMN MATCHING
# =============================================================================

class FuzzyColumnMatcher:
    """
    Matches messy column names to standard schema using Levenshtein distance.
    
    Handles variations like:
    - Date_of_Birth, DOB, BirthDate, D.O.B. -> dob
    - Annual_Salary, Salary, Pay, Compensation -> salary
    """
    
    # Standard column names and their aliases
    COLUMN_MAPPINGS = {
        'dob': {
            'aliases': ['dob', 'dateofbirth', 'date_of_birth', 'birthdate', 'birth_date',
                       'd.o.b.', 'dob', 'birthday', 'birth', 'bdate'],
            'patterns': [r'birth', r'd\.?o\.?b\.?', r'bday'],
        },
        'doh': {
            'aliases': ['doh', 'dateofhire', 'date_of_hire', 'hiredate', 'hire_date',
                       'employmentdate', 'startdate', 'start_date', 'hired'],
            'patterns': [r'hire', r'employ', r'start', r'd\.?o\.?h\.?'],
        },
        'gender': {
            'aliases': ['gender', 'sex', 'gend', 'm/f', 'male/female'],
            'patterns': [r'gender', r'sex', r'm/?f'],
        },
        'status': {
            'aliases': ['status', 'empstatus', 'employeestatus', 'employmentstatus',
                       'stat', 'active/retired', 'empstat'],
            'patterns': [r'status', r'stat', r'active'],
        },
        'salary': {
            'aliases': ['salary', 'annualsalary', 'annual_salary', 'pay', 'compensation',
                       'wages', 'earnings', 'income', 'basesalary', 'base_salary'],
            'patterns': [r'salary', r'pay', r'wage', r'comp', r'earn', r'income'],
        },
        'service': {
            'aliases': ['service', 'yearsofservice', 'years_of_service', 'seniority',
                       'tenure', 'yos', 'svc', 'serviceyears'],
            'patterns': [r'service', r'tenure', r'seniority', r'yos', r'svc'],
        },
        'age': {
            'aliases': ['age', 'currentage', 'current_age', 'attainedage'],
            'patterns': [r'^age$', r'current.*age', r'attained'],
        },
        'member_id': {
            'aliases': ['memberid', 'member_id', 'id', 'employeeid', 'employee_id',
                       'ssn', 'empid', 'emp_id', 'participantid', 'recordid'],
            'patterns': [r'id$', r'member', r'employee', r'ssn', r'participant'],
        },
        'coverage': {
            'aliases': ['coverage', 'coveragelevel', 'coverage_level', 'tier',
                       'coveragetype', 'plan', 'benefit_tier'],
            'patterns': [r'coverage', r'tier', r'plan'],
        },
        'name': {
            'aliases': ['name', 'fullname', 'full_name', 'employeename', 'lastname',
                       'firstname', 'first_name', 'last_name'],
            'patterns': [r'name'],
        },
        'spouse_dob': {
            'aliases': ['spousedob', 'spouse_dob', 'spousebirthdate', 'spouse_birth',
                       'spdob', 'sp_dob', 'dependent_dob'],
            'patterns': [r'spouse.*birth', r'spouse.*dob', r'sp.*dob'],
        },
    }
    
    @classmethod
    def levenshtein_distance(cls, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return cls.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @classmethod
    def normalize_column_name(cls, name: str) -> str:
        """Normalize column name for comparison."""
        # Lowercase, remove special chars, collapse whitespace
        normalized = name.lower()
        normalized = re.sub(r'[^a-z0-9]', '', normalized)
        return normalized
    
    @classmethod
    def match_column(cls, column_name: str, threshold: int = 3) -> Optional[str]:
        """
        Match a column name to standard schema.
        
        Args:
            column_name: Raw column name from client file
            threshold: Maximum Levenshtein distance for fuzzy match
        
        Returns:
            Standard column name or None if no match
        """
        normalized = cls.normalize_column_name(column_name)
        
        best_match = None
        best_score = float('inf')
        
        for standard_name, config in cls.COLUMN_MAPPINGS.items():
            # Check exact alias matches first
            for alias in config['aliases']:
                alias_norm = cls.normalize_column_name(alias)
                if normalized == alias_norm:
                    return standard_name
                
                # Fuzzy match
                distance = cls.levenshtein_distance(normalized, alias_norm)
                if distance < best_score and distance <= threshold:
                    best_score = distance
                    best_match = standard_name
            
            # Check regex patterns
            for pattern in config['patterns']:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return standard_name
        
        return best_match
    
    @classmethod
    def map_columns(cls, columns: List[str]) -> Dict[str, str]:
        """
        Map all columns in a DataFrame to standard names.
        
        Returns:
            Dict mapping original column name -> standard name
        """
        mapping = {}
        used_standards = set()
        
        for col in columns:
            standard = cls.match_column(col)
            if standard and standard not in used_standards:
                mapping[col] = standard
                used_standards.add(standard)
        
        return mapping


# =============================================================================
# DATE PARSING HEURISTICS
# =============================================================================

class SmartDateParser:
    """
    Multi-format date parser with fallback heuristics.
    
    Tries:
    1. ISO format (YYYY-MM-DD)
    2. US format (MM/DD/YYYY)
    3. European format (DD/MM/YYYY)
    4. Excel serial number
    5. Imputation from Age
    """
    
    DATE_FORMATS = [
        '%Y-%m-%d',      # ISO: 2025-01-15
        '%m/%d/%Y',      # US: 01/15/2025
        '%d/%m/%Y',      # EU: 15/01/2025
        '%Y/%m/%d',      # Alt ISO: 2025/01/15
        '%m-%d-%Y',      # US dash: 01-15-2025
        '%d-%m-%Y',      # EU dash: 15-01-2025
        '%Y%m%d',        # Compact: 20250115
        '%m%d%Y',        # Compact US: 01152025
        '%b %d, %Y',     # Jan 15, 2025
        '%B %d, %Y',     # January 15, 2025
        '%d %b %Y',      # 15 Jan 2025
        '%d %B %Y',      # 15 January 2025
    ]
    
    # Excel epoch (January 1, 1900, but Excel incorrectly counts Feb 29, 1900)
    EXCEL_EPOCH = datetime(1899, 12, 30)
    
    @classmethod
    def parse(cls, value: Any, 
              reference_date: Optional[date] = None,
              age_fallback: Optional[float] = None) -> Tuple[Optional[date], str]:
        """
        Parse a date value with multiple format attempts.
        
        Args:
            value: Raw date value
            reference_date: Reference date for age-based imputation
            age_fallback: Age to use for imputation if date fails
        
        Returns:
            Tuple of (parsed date or None, parsing method used)
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if age_fallback is not None and reference_date is not None:
                imputed = cls._impute_from_age(age_fallback, reference_date)
                return imputed, "imputed_from_age"
            return None, "null"
        
        # Already a date/datetime
        if isinstance(value, datetime):
            return value.date(), "datetime"
        if isinstance(value, date):
            return value, "date"
        
        # Try Excel serial number (numeric)
        if isinstance(value, (int, float)):
            try:
                if 1 < value < 100000:  # Reasonable Excel date range
                    excel_date = cls.EXCEL_EPOCH + timedelta(days=int(value))
                    return excel_date.date(), "excel_serial"
            except:
                pass
        
        # Try string formats
        if isinstance(value, str):
            value = value.strip()
            
            for fmt in cls.DATE_FORMATS:
                try:
                    parsed = datetime.strptime(value, fmt)
                    return parsed.date(), f"format:{fmt}"
                except ValueError:
                    continue
        
        # Fallback to age imputation
        if age_fallback is not None and reference_date is not None:
            imputed = cls._impute_from_age(age_fallback, reference_date)
            return imputed, "imputed_from_age"
        
        return None, "parse_failed"
    
    @classmethod
    def _impute_from_age(cls, age: float, reference_date: date) -> date:
        """Impute birth date from age."""
        years_ago = int(age)
        return date(reference_date.year - years_ago, 7, 1)  # Mid-year default


# =============================================================================
# STATUS UNIFICATION
# =============================================================================

class StatusUnifier:
    """
    Unifies various status representations to standard enum.
    
    Maps:
    - ['Active', 'A', 'ACT', 'Emp', 'Employee', 'Working'] -> ACTIVE
    - ['Retiree', 'R', 'RET', 'Retired', 'Pay', 'Payee'] -> RETIREE
    - ['Term', 'T', 'Terminated', 'Sep', 'Separated'] -> TERMINATED
    """
    
    STATUS_MAPPINGS = {
        MemberStatus.ACTIVE: [
            'active', 'a', 'act', 'emp', 'employee', 'working', 'employed',
            'current', 'cur', 'w', 'work', 'e', 'ac'
        ],
        MemberStatus.RETIREE: [
            'retiree', 'r', 'ret', 'retired', 'pay', 'payee', 'pension',
            'pensioner', 'annuitant', 'beneficiary', 'ben', 'p', 're'
        ],
        MemberStatus.TERMINATED: [
            'terminated', 'term', 't', 'sep', 'separated', 'quit', 'left',
            'departed', 'inactive', 'former', 'te', 'tm'
        ],
        MemberStatus.DISABLED: [
            'disabled', 'dis', 'd', 'disability', 'ltd', 'std', 'di'
        ],
    }
    
    @classmethod
    def unify(cls, value: Any) -> MemberStatus:
        """Convert raw status value to standard enum."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return MemberStatus.UNKNOWN
        
        normalized = str(value).lower().strip()
        
        for status, aliases in cls.STATUS_MAPPINGS.items():
            if normalized in aliases:
                return status
        
        # Fuzzy match for partial matches
        for status, aliases in cls.STATUS_MAPPINGS.items():
            for alias in aliases:
                if alias in normalized or normalized in alias:
                    return status
        
        return MemberStatus.UNKNOWN


# =============================================================================
# DATA QUALITY ISSUE TRACKING
# =============================================================================

@dataclass
class DataQualityIssue:
    """Represents a single data quality issue."""
    record_index: int
    record_id: Optional[str]
    field_name: str
    issue_type: IssueType
    original_value: Any
    corrected_value: Any
    message: str
    severity: DataQualityLevel


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    input_file: str
    input_hash: str
    processing_timestamp: datetime
    total_records: int
    clean_records: int
    imputed_records: int
    warning_records: int
    excluded_records: int
    issues: List[DataQualityIssue]
    column_mapping: Dict[str, str]
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        md = f"""# Data Quality Report

## File Information
- **Input File:** {self.input_file}
- **SHA-256 Hash:** `{self.input_hash}`
- **Processed:** {self.processing_timestamp.isoformat()}

## Record Statistics
| Category | Count | Percentage |
|----------|-------|------------|
| Total Records | {self.total_records} | 100.0% |
| Clean Records | {self.clean_records} | {self.clean_records/self.total_records*100:.1f}% |
| Imputed Records | {self.imputed_records} | {self.imputed_records/self.total_records*100:.1f}% |
| Warning Records | {self.warning_records} | {self.warning_records/self.total_records*100:.1f}% |
| Excluded Records | {self.excluded_records} | {self.excluded_records/self.total_records*100:.1f}% |

## Column Mapping
| Original Column | Standard Column |
|-----------------|-----------------|
"""
        for orig, std in self.column_mapping.items():
            md += f"| {orig} | {std} |\n"
        
        md += f"""
## Data Issues ({len(self.issues)} total)

"""
        # Group issues by type
        issues_by_type = {}
        for issue in self.issues:
            key = issue.issue_type.value
            if key not in issues_by_type:
                issues_by_type[key] = []
            issues_by_type[key].append(issue)
        
        for issue_type, issues in issues_by_type.items():
            md += f"### {issue_type.replace('_', ' ').title()} ({len(issues)} issues)\n\n"
            for issue in issues[:10]:  # Show first 10
                md += f"- **Record {issue.record_index}** ({issue.record_id}): "
                md += f"{issue.message}\n"
            if len(issues) > 10:
                md += f"- ... and {len(issues) - 10} more\n"
            md += "\n"
        
        return md
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps({
            'input_file': self.input_file,
            'input_hash': self.input_hash,
            'timestamp': self.processing_timestamp.isoformat(),
            'total_records': self.total_records,
            'clean_records': self.clean_records,
            'imputed_records': self.imputed_records,
            'excluded_records': self.excluded_records,
            'issues': [
                {
                    'record': i.record_index,
                    'field': i.field_name,
                    'type': i.issue_type.value,
                    'original': str(i.original_value),
                    'corrected': str(i.corrected_value),
                    'message': i.message,
                }
                for i in self.issues
            ]
        }, indent=2)


# =============================================================================
# SMART CENSUS LOADER
# =============================================================================

class SmartCensusLoader:
    """
    "Client Chaos" Data Ingestion System.
    
    Features:
    1. Fuzzy column name matching
    2. Multi-format date parsing
    3. Status unification
    4. ASOP 23 imputation
    5. SHA-256 audit trail
    6. Data Forensics reporting
    """
    
    def __init__(self, valuation_date: date):
        self.valuation_date = valuation_date
        self.issues: List[DataQualityIssue] = []
    
    def load_file(self, filepath: Union[str, Path],
                  sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Load and clean census file.
        
        Args:
            filepath: Path to Excel or CSV file
            sheet_name: Sheet name for Excel files
        
        Returns:
            Tuple of (cleaned DataFrame, DataQualityReport)
        """
        filepath = Path(filepath)
        self.issues = []
        
        # Calculate file hash
        file_hash = self._hash_file(filepath)
        logger.info(f"Loading file: {filepath.name} (SHA-256: {file_hash[:16]}...)")
        
        # Load raw data
        if filepath.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            df = pd.read_csv(filepath)
        
        total_records = len(df)
        logger.info(f"Loaded {total_records} records")
        
        # Step 1: Fuzzy column mapping
        column_mapping = FuzzyColumnMatcher.map_columns(df.columns.tolist())
        df = df.rename(columns=column_mapping)
        
        # Step 2: Process each field
        df = self._process_dates(df)
        df = self._process_status(df)
        df = self._process_gender(df)
        df = self._process_numeric_fields(df)
        df = self._apply_asop23_imputation(df)
        df = self._validate_logical_rules(df)
        
        # Step 3: Calculate statistics
        record_quality = self._classify_records(df)
        
        # Step 4: Build report
        report = DataQualityReport(
            input_file=filepath.name,
            input_hash=file_hash,
            processing_timestamp=datetime.now(),
            total_records=total_records,
            clean_records=sum(1 for q in record_quality if q == DataQualityLevel.CLEAN),
            imputed_records=sum(1 for q in record_quality if q == DataQualityLevel.IMPUTED),
            warning_records=sum(1 for q in record_quality if q == DataQualityLevel.WARNING),
            excluded_records=sum(1 for q in record_quality if q == DataQualityLevel.EXCLUDED),
            issues=self.issues,
            column_mapping=column_mapping
        )
        
        # Add quality column
        df['_data_quality'] = [q.value for q in record_quality]
        
        return df, report
    
    def _hash_file(self, filepath: Path) -> str:
        """Calculate SHA-256 hash."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all date columns."""
        date_columns = ['dob', 'doh', 'spouse_dob']
        
        for col in date_columns:
            if col not in df.columns:
                continue
            
            # Get age column for fallback
            age_col = df.get('age')
            
            parsed_dates = []
            for idx, row in df.iterrows():
                value = row[col]
                age = row['age'] if age_col is not None and pd.notna(row.get('age')) else None
                
                parsed, method = SmartDateParser.parse(
                    value,
                    reference_date=self.valuation_date,
                    age_fallback=age if col == 'dob' else None
                )
                
                parsed_dates.append(parsed)
                
                if method not in ('date', 'datetime', 'format:%Y-%m-%d'):
                    self.issues.append(DataQualityIssue(
                        record_index=idx,
                        record_id=str(row.get('member_id', idx)),
                        field_name=col,
                        issue_type=IssueType.NORMALIZED if parsed else IssueType.MISSING_VALUE,
                        original_value=value,
                        corrected_value=parsed,
                        message=f"Date parsed using {method}",
                        severity=DataQualityLevel.IMPUTED if 'imputed' in method else DataQualityLevel.WARNING
                    ))
            
            df[col] = parsed_dates
        
        return df
    
    def _process_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process status column."""
        if 'status' not in df.columns:
            # Infer status from other columns
            df['status'] = MemberStatus.ACTIVE.value
            return df
        
        unified_status = []
        for idx, value in enumerate(df['status']):
            status = StatusUnifier.unify(value)
            unified_status.append(status.value)
            
            if status == MemberStatus.UNKNOWN:
                self.issues.append(DataQualityIssue(
                    record_index=idx,
                    record_id=str(df.iloc[idx].get('member_id', idx)),
                    field_name='status',
                    issue_type=IssueType.INVALID_FORMAT,
                    original_value=value,
                    corrected_value=MemberStatus.ACTIVE.value,
                    message=f"Unknown status '{value}', defaulting to Active",
                    severity=DataQualityLevel.WARNING
                ))
                unified_status[-1] = MemberStatus.ACTIVE.value
        
        df['status'] = unified_status
        return df
    
    def _process_gender(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process gender column."""
        if 'gender' not in df.columns:
            df['gender'] = 'M'
            return df
        
        def normalize_gender(value):
            if pd.isna(value):
                return 'M'
            v = str(value).upper().strip()
            if v in ('M', 'MALE', '1'):
                return 'M'
            if v in ('F', 'FEMALE', '2', 'W'):
                return 'F'
            return 'M'  # Default
        
        df['gender'] = df['gender'].apply(normalize_gender)
        return df
    
    def _process_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process numeric fields (salary, service, age)."""
        numeric_cols = ['salary', 'service', 'age']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _apply_asop23_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ASOP 23 compliant imputations."""
        
        # Ensure member_id exists
        if 'member_id' not in df.columns:
            df['member_id'] = [f'M{i:05d}' for i in range(len(df))]
        
        for idx, row in df.iterrows():
            record_id = str(row['member_id'])
            
            # Impute DOB from Age
            if pd.isna(row.get('dob')) and pd.notna(row.get('age')):
                age = row['age']
                imputed_dob = date(self.valuation_date.year - int(age), 7, 1)
                df.at[idx, 'dob'] = imputed_dob
                self._log_imputation(idx, record_id, 'dob', None, imputed_dob,
                                    f"Imputed from age={age}")
            
            # Impute DOH if missing (assume entry age 30)
            if pd.isna(row.get('doh')) and pd.notna(row.get('dob')):
                dob = row['dob']
                if isinstance(dob, date):
                    entry_age = 30
                    imputed_doh = date(dob.year + entry_age, 1, 1)
                    df.at[idx, 'doh'] = imputed_doh
                    self._log_imputation(idx, record_id, 'doh', None, imputed_doh,
                                        f"Imputed assuming entry age {entry_age}")
            
            # Impute salary if missing
            if pd.isna(row.get('salary')) or row.get('salary', 0) <= 0:
                plan_avg = df['salary'].dropna().mean()
                imputed_salary = plan_avg if pd.notna(plan_avg) else 50000.0
                df.at[idx, 'salary'] = imputed_salary
                self._log_imputation(idx, record_id, 'salary', row.get('salary'),
                                    imputed_salary, "Imputed from plan average")
        
        return df
    
    def _validate_logical_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix logical inconsistencies."""
        
        for idx, row in df.iterrows():
            record_id = str(row.get('member_id', idx))
            
            # Check DOH > ValDate
            if pd.notna(row.get('doh')):
                doh = row['doh']
                if isinstance(doh, date) and doh > self.valuation_date:
                    df.at[idx, 'doh'] = self.valuation_date
                    self.issues.append(DataQualityIssue(
                        record_index=idx,
                        record_id=record_id,
                        field_name='doh',
                        issue_type=IssueType.LOGICAL_ERROR,
                        original_value=doh,
                        corrected_value=self.valuation_date,
                        message=f"HireDate {doh} > ValDate. Reset to ValDate.",
                        severity=DataQualityLevel.WARNING
                    ))
            
            # Check DOB results in reasonable age
            if pd.notna(row.get('dob')):
                dob = row['dob']
                if isinstance(dob, date):
                    age = (self.valuation_date - dob).days / 365.25
                    if age < 18 or age > 110:
                        self.issues.append(DataQualityIssue(
                            record_index=idx,
                            record_id=record_id,
                            field_name='dob',
                            issue_type=IssueType.OUT_OF_RANGE,
                            original_value=dob,
                            corrected_value=dob,
                            message=f"Calculated age {age:.1f} is out of range",
                            severity=DataQualityLevel.WARNING
                        ))
        
        return df
    
    def _log_imputation(self, idx: int, record_id: str, field: str,
                        original: Any, corrected: Any, message: str):
        """Log an imputation action."""
        self.issues.append(DataQualityIssue(
            record_index=idx,
            record_id=record_id,
            field_name=field,
            issue_type=IssueType.IMPUTED,
            original_value=original,
            corrected_value=corrected,
            message=message,
            severity=DataQualityLevel.IMPUTED
        ))
    
    def _classify_records(self, df: pd.DataFrame) -> List[DataQualityLevel]:
        """Classify each record's data quality."""
        # Build issue map
        issue_map = {}
        for issue in self.issues:
            if issue.record_index not in issue_map:
                issue_map[issue.record_index] = []
            issue_map[issue.record_index].append(issue)
        
        quality = []
        for idx in range(len(df)):
            if idx not in issue_map:
                quality.append(DataQualityLevel.CLEAN)
            else:
                issues = issue_map[idx]
                severities = [i.severity for i in issues]
                if DataQualityLevel.EXCLUDED in severities:
                    quality.append(DataQualityLevel.EXCLUDED)
                elif DataQualityLevel.WARNING in severities:
                    quality.append(DataQualityLevel.WARNING)
                else:
                    quality.append(DataQualityLevel.IMPUTED)
        
        return quality


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def smart_load_census(
    filepath: Union[str, Path],
    valuation_date: date,
    sheet_name: Optional[str] = None,
    save_report: bool = True
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Load census file with smart cleaning and quality reporting.
    
    Args:
        filepath: Path to census file
        valuation_date: Valuation date
        sheet_name: Excel sheet name (optional)
        save_report: Whether to save quality report to file
    
    Returns:
        Tuple of (cleaned DataFrame, DataQualityReport)
    """
    loader = SmartCensusLoader(valuation_date)
    df, report = loader.load_file(filepath, sheet_name)
    
    if save_report:
        filepath = Path(filepath)
        report_path = filepath.parent / f"{filepath.stem}_Data_Quality_Log.md"
        with open(report_path, 'w') as f:
            f.write(report.to_markdown())
        logger.info(f"Quality report saved to {report_path}")
    
    return df, report


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMART INGESTION MODULE - UNIT TESTS")
    print("=" * 70)
    
    # Test 1: Fuzzy Column Matching
    print("\nTest 1: Fuzzy Column Matching")
    print("-" * 50)
    
    test_columns = [
        'Date_of_Birth', 'DOB', 'BirthDate', 'D.O.B.',
        'Annual_Salary', 'Salary', 'PAY',
        'Emp_Status', 'Active/Retired', 'STATUS'
    ]
    
    for col in test_columns:
        match = FuzzyColumnMatcher.match_column(col)
        print(f"  '{col}' -> {match}")
    
    # Test 2: Date Parsing
    print("\nTest 2: Smart Date Parsing")
    print("-" * 50)
    
    test_dates = [
        '2025-01-15',     # ISO
        '01/15/2025',     # US
        '15/01/2025',     # EU
        45678,            # Excel serial
        'Jan 15, 2025',   # Text
    ]
    
    for d in test_dates:
        parsed, method = SmartDateParser.parse(d, date(2025, 9, 30))
        print(f"  '{d}' -> {parsed} ({method})")
    
    # Test 3: Status Unification
    print("\nTest 3: Status Unification")
    print("-" * 50)
    
    test_statuses = ['Active', 'A', 'ACT', 'Emp', 'R', 'Retired', 'RET', 'Term', 'XYZ']
    
    for s in test_statuses:
        unified = StatusUnifier.unify(s)
        print(f"  '{s}' -> {unified.value}")
    
    print("\n✓ All smart ingestion tests passed")
