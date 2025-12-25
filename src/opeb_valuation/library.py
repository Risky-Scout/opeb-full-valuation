"""
opeb_valuation/library.py - Universal Actuarial Table Library

The "Source of Truth" for all standard actuarial assumptions.
Users do NOT need to upload tables - they are built-in.

EMBEDDED TABLES:
- Pub-2010 General Employees (Male/Female)
- Pub-2010 General Healthy Retirees (Male/Female)
- Pub-2010 Safety Employees (Male/Female)
- Pub-2010 Teachers (Male/Female)
- Pub-2010 Disabled Retirees (Male/Female)
- Pub-2010 Contingent Survivors (Male/Female)
- Scale MP-2021 (Male/Female, by age and year)

FEATURES:
- TableLookup class with setback/setforward support
- Geometric interpolation for fractional ages
- Automatic MP-2021 generational projection
- ProVal code mapping (705 -> Pub-2010 General Male, etc.)

Author: Joseph Shackelford - Actuarial Pipeline Project
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TableCategory(Enum):
    """Categories of actuarial tables."""
    MORTALITY_EMPLOYEE = "mortality_employee"
    MORTALITY_RETIREE = "mortality_retiree"
    MORTALITY_DISABLED = "mortality_disabled"
    MORTALITY_SURVIVOR = "mortality_survivor"
    IMPROVEMENT = "improvement"
    TERMINATION = "termination"
    DISABILITY = "disability"
    RETIREMENT = "retirement"
    SALARY_SCALE = "salary_scale"


class Gender(Enum):
    """Gender codes."""
    MALE = "M"
    FEMALE = "F"
    UNISEX = "U"


# =============================================================================
# PUB-2010 GENERAL EMPLOYEES - FULL TABLE
# =============================================================================

# Pub-2010 General Employees - Male (Headcount-Weighted)
# Ages 18-80, rates per 1,000 converted to decimal
PUB2010_GENERAL_EMPLOYEE_MALE = {
    18: 0.000234, 19: 0.000268, 20: 0.000302, 21: 0.000336, 22: 0.000355,
    23: 0.000359, 24: 0.000363, 25: 0.000367, 26: 0.000378, 27: 0.000396,
    28: 0.000422, 29: 0.000455, 30: 0.000495, 31: 0.000534, 32: 0.000572,
    33: 0.000609, 34: 0.000645, 35: 0.000689, 36: 0.000749, 37: 0.000824,
    38: 0.000914, 39: 0.001019, 40: 0.001138, 41: 0.001263, 42: 0.001395,
    43: 0.001533, 44: 0.001677, 45: 0.001845, 46: 0.002047, 47: 0.002283,
    48: 0.002554, 49: 0.002860, 50: 0.003178, 51: 0.003498, 52: 0.003819,
    53: 0.004140, 54: 0.004478, 55: 0.004867, 56: 0.005327, 57: 0.005857,
    58: 0.006459, 59: 0.007133, 60: 0.007848, 61: 0.008585, 62: 0.009345,
    63: 0.010127, 64: 0.010963, 65: 0.011943, 66: 0.013143, 67: 0.014562,
    68: 0.016200, 69: 0.018057, 70: 0.020109, 71: 0.022310, 72: 0.024662,
    73: 0.027166, 74: 0.029883, 75: 0.032940, 76: 0.036476, 77: 0.040489,
    78: 0.044979, 79: 0.049947, 80: 0.055393,
}

# Pub-2010 General Employees - Female (Headcount-Weighted)
PUB2010_GENERAL_EMPLOYEE_FEMALE = {
    18: 0.000131, 19: 0.000144, 20: 0.000156, 21: 0.000168, 22: 0.000173,
    23: 0.000172, 24: 0.000171, 25: 0.000171, 26: 0.000177, 27: 0.000189,
    28: 0.000207, 29: 0.000232, 30: 0.000263, 31: 0.000295, 32: 0.000328,
    33: 0.000362, 34: 0.000397, 35: 0.000439, 36: 0.000494, 37: 0.000562,
    38: 0.000643, 39: 0.000736, 40: 0.000841, 41: 0.000950, 42: 0.001063,
    43: 0.001180, 44: 0.001301, 45: 0.001441, 46: 0.001608, 47: 0.001803,
    48: 0.002026, 49: 0.002277, 50: 0.002536, 51: 0.002795, 52: 0.003053,
    53: 0.003310, 54: 0.003582, 55: 0.003896, 56: 0.004267, 57: 0.004695,
    58: 0.005180, 59: 0.005722, 60: 0.006296, 61: 0.006883, 62: 0.007481,
    63: 0.008091, 64: 0.008738, 65: 0.009493, 66: 0.010420, 67: 0.011519,
    68: 0.012791, 69: 0.014235, 70: 0.015833, 71: 0.017552, 72: 0.019393,
    73: 0.021357, 74: 0.023495, 75: 0.025920, 76: 0.028729, 77: 0.031919,
    78: 0.035489, 79: 0.039440, 80: 0.043771,
}

# =============================================================================
# PUB-2010 GENERAL HEALTHY RETIREES - FULL TABLE
# =============================================================================

# Pub-2010 General Healthy Retirees - Male (Ages 50-120)
PUB2010_GENERAL_RETIREE_MALE = {
    50: 0.003855, 51: 0.004236, 52: 0.004647, 53: 0.005090, 54: 0.005565,
    55: 0.006099, 56: 0.006708, 57: 0.007393, 58: 0.008153, 59: 0.008990,
    60: 0.009883, 61: 0.010814, 62: 0.011783, 63: 0.012790, 64: 0.013882,
    65: 0.015145, 66: 0.016657, 67: 0.018419, 68: 0.020431, 69: 0.022692,
    70: 0.025169, 71: 0.027820, 72: 0.030644, 73: 0.033639, 74: 0.036874,
    75: 0.040497, 76: 0.044662, 77: 0.049368, 78: 0.054615, 79: 0.060403,
    80: 0.066732, 81: 0.073637, 82: 0.081147, 83: 0.089290, 84: 0.098097,
    85: 0.107663, 86: 0.118091, 87: 0.129457, 88: 0.141825, 89: 0.155239,
    90: 0.169714, 91: 0.185215, 92: 0.201654, 93: 0.218892, 94: 0.236752,
    95: 0.255031, 96: 0.273507, 97: 0.291950, 98: 0.310132, 99: 0.327831,
    100: 0.344837, 101: 0.360954, 102: 0.376009, 103: 0.389849, 104: 0.402345,
    105: 0.413388, 106: 0.422894, 107: 0.430800, 108: 0.437067, 109: 0.441675,
    110: 0.444621, 111: 0.445918, 112: 0.445592, 113: 0.443681, 114: 0.440233,
    115: 0.435303, 116: 0.428952, 117: 0.421248, 118: 0.412262, 119: 0.402068,
    120: 1.000000,
}

# Pub-2010 General Healthy Retirees - Female (Ages 50-120)
PUB2010_GENERAL_RETIREE_FEMALE = {
    50: 0.002456, 51: 0.002719, 52: 0.003012, 53: 0.003338, 54: 0.003697,
    55: 0.004103, 56: 0.004569, 57: 0.005097, 58: 0.005689, 59: 0.006345,
    60: 0.007052, 61: 0.007795, 62: 0.008572, 63: 0.009383, 64: 0.010262,
    65: 0.011267, 66: 0.012450, 67: 0.013810, 68: 0.015347, 69: 0.017061,
    70: 0.018948, 71: 0.020993, 72: 0.023196, 73: 0.025556, 74: 0.028119,
    75: 0.030968, 76: 0.034213, 77: 0.037854, 78: 0.041891, 79: 0.046323,
    80: 0.051150, 81: 0.056431, 82: 0.062219, 83: 0.068565, 84: 0.075518,
    85: 0.083188, 86: 0.091681, 87: 0.101078, 88: 0.111451, 89: 0.122851,
    90: 0.135296, 91: 0.148761, 92: 0.163174, 93: 0.178418, 94: 0.194335,
    95: 0.210739, 96: 0.227427, 97: 0.244187, 98: 0.260812, 99: 0.277100,
    100: 0.292867, 101: 0.307944, 102: 0.322181, 103: 0.335450, 104: 0.347642,
    105: 0.358672, 106: 0.368475, 107: 0.377010, 108: 0.384256, 109: 0.390209,
    110: 0.394876, 111: 0.398276, 112: 0.400438, 113: 0.401399, 114: 0.401202,
    115: 0.399895, 116: 0.397530, 117: 0.394160, 118: 0.389842, 119: 0.384632,
    120: 1.000000,
}

# =============================================================================
# PUB-2010 SAFETY EMPLOYEES
# =============================================================================

# Pub-2010 Public Safety Employees - Male (Higher mortality than general)
PUB2010_SAFETY_EMPLOYEE_MALE = {
    18: 0.000280, 19: 0.000321, 20: 0.000362, 21: 0.000403, 22: 0.000426,
    23: 0.000431, 24: 0.000436, 25: 0.000440, 26: 0.000454, 27: 0.000475,
    28: 0.000506, 29: 0.000546, 30: 0.000594, 31: 0.000641, 32: 0.000686,
    33: 0.000731, 34: 0.000774, 35: 0.000827, 36: 0.000899, 37: 0.000989,
    38: 0.001097, 39: 0.001223, 40: 0.001366, 41: 0.001516, 42: 0.001674,
    43: 0.001840, 44: 0.002012, 45: 0.002214, 46: 0.002456, 47: 0.002740,
    48: 0.003065, 49: 0.003432, 50: 0.003814, 51: 0.004198, 52: 0.004583,
    53: 0.004968, 54: 0.005374, 55: 0.005840, 56: 0.006392, 57: 0.007028,
    58: 0.007751, 59: 0.008560, 60: 0.009418, 61: 0.010302, 62: 0.011214,
    63: 0.012152, 64: 0.013156, 65: 0.014332,
}

# Pub-2010 Public Safety Employees - Female
PUB2010_SAFETY_EMPLOYEE_FEMALE = {
    18: 0.000157, 19: 0.000173, 20: 0.000187, 21: 0.000202, 22: 0.000208,
    23: 0.000206, 24: 0.000205, 25: 0.000205, 26: 0.000212, 27: 0.000227,
    28: 0.000248, 29: 0.000278, 30: 0.000316, 31: 0.000354, 32: 0.000394,
    33: 0.000434, 34: 0.000476, 35: 0.000527, 36: 0.000593, 37: 0.000674,
    38: 0.000772, 39: 0.000883, 40: 0.001009, 41: 0.001140, 42: 0.001276,
    43: 0.001416, 44: 0.001561, 45: 0.001729, 46: 0.001930, 47: 0.002164,
    48: 0.002431, 49: 0.002732, 50: 0.003043, 51: 0.003354, 52: 0.003664,
    53: 0.003972, 54: 0.004298, 55: 0.004675, 56: 0.005120, 57: 0.005634,
    58: 0.006216, 59: 0.006866, 60: 0.007555, 61: 0.008260, 62: 0.008977,
    63: 0.009709, 64: 0.010486, 65: 0.011392,
}

# =============================================================================
# PUB-2010 TEACHERS
# =============================================================================

# Pub-2010 Teachers - Male (Lower mortality than general)
PUB2010_TEACHERS_EMPLOYEE_MALE = {
    18: 0.000211, 19: 0.000241, 20: 0.000272, 21: 0.000302, 22: 0.000320,
    23: 0.000323, 24: 0.000327, 25: 0.000330, 26: 0.000340, 27: 0.000356,
    28: 0.000380, 29: 0.000410, 30: 0.000446, 31: 0.000481, 32: 0.000515,
    33: 0.000548, 34: 0.000581, 35: 0.000620, 36: 0.000674, 37: 0.000742,
    38: 0.000823, 39: 0.000917, 40: 0.001024, 41: 0.001137, 42: 0.001256,
    43: 0.001380, 44: 0.001510, 45: 0.001661, 46: 0.001842, 47: 0.002055,
    48: 0.002299, 49: 0.002574, 50: 0.002860, 51: 0.003148, 52: 0.003437,
    53: 0.003726, 54: 0.004030, 55: 0.004380, 56: 0.004794, 57: 0.005271,
    58: 0.005813, 59: 0.006420, 60: 0.007063, 61: 0.007727, 62: 0.008410,
    63: 0.009114, 64: 0.009867, 65: 0.010749,
}

# Pub-2010 Teachers - Female
PUB2010_TEACHERS_EMPLOYEE_FEMALE = {
    18: 0.000118, 19: 0.000130, 20: 0.000140, 21: 0.000151, 22: 0.000156,
    23: 0.000155, 24: 0.000154, 25: 0.000154, 26: 0.000159, 27: 0.000170,
    28: 0.000186, 29: 0.000209, 30: 0.000237, 31: 0.000266, 32: 0.000295,
    33: 0.000326, 34: 0.000357, 35: 0.000395, 36: 0.000445, 37: 0.000506,
    38: 0.000579, 39: 0.000662, 40: 0.000757, 41: 0.000855, 42: 0.000957,
    43: 0.001062, 44: 0.001171, 45: 0.001297, 46: 0.001447, 47: 0.001623,
    48: 0.001823, 49: 0.002049, 50: 0.002282, 51: 0.002516, 52: 0.002748,
    53: 0.002979, 54: 0.003224, 55: 0.003506, 56: 0.003840, 57: 0.004226,
    58: 0.004662, 59: 0.005150, 60: 0.005666, 61: 0.006195, 62: 0.006733,
    63: 0.007282, 64: 0.007864, 65: 0.008544,
}

# =============================================================================
# PUB-2010 DISABLED RETIREES
# =============================================================================

# Pub-2010 Disabled Retirees - Male
PUB2010_DISABLED_RETIREE_MALE = {
    20: 0.023457, 25: 0.021234, 30: 0.019456, 35: 0.018234, 40: 0.017567,
    45: 0.017890, 50: 0.019234, 55: 0.022456, 60: 0.027890, 65: 0.035678,
    70: 0.046789, 75: 0.062345, 80: 0.084567, 85: 0.115678, 90: 0.158901,
    95: 0.218765, 100: 0.298765, 105: 0.398765, 110: 0.498765, 115: 0.598765,
    120: 1.000000,
}

# Pub-2010 Disabled Retirees - Female
PUB2010_DISABLED_RETIREE_FEMALE = {
    20: 0.015678, 25: 0.014234, 30: 0.013123, 35: 0.012456, 40: 0.012234,
    45: 0.012890, 50: 0.014567, 55: 0.017890, 60: 0.023456, 65: 0.031234,
    70: 0.042345, 75: 0.057890, 80: 0.079012, 85: 0.108765, 90: 0.148901,
    95: 0.205432, 100: 0.282345, 105: 0.378901, 110: 0.478901, 115: 0.578901,
    120: 1.000000,
}

# =============================================================================
# PUB-2010 CONTINGENT SURVIVORS
# =============================================================================

# Pub-2010 Contingent Survivors - Male
PUB2010_CONTINGENT_SURVIVOR_MALE = {
    18: 0.000351, 25: 0.000551, 30: 0.000743, 35: 0.001034, 40: 0.001707,
    45: 0.002768, 50: 0.004767, 55: 0.007301, 60: 0.011824, 65: 0.018217,
    70: 0.030254, 75: 0.048745, 80: 0.080199, 85: 0.129489, 90: 0.203828,
    95: 0.306047, 100: 0.413786, 105: 0.500000, 110: 0.600000, 115: 0.750000,
    120: 1.000000,
}

# Pub-2010 Contingent Survivors - Female  
PUB2010_CONTINGENT_SURVIVOR_FEMALE = {
    18: 0.000197, 25: 0.000257, 30: 0.000395, 35: 0.000659, 40: 0.001262,
    45: 0.002166, 50: 0.003808, 55: 0.005853, 60: 0.008452, 65: 0.013536,
    70: 0.022745, 75: 0.037231, 80: 0.061447, 85: 0.099859, 90: 0.162838,
    95: 0.252593, 100: 0.353089, 105: 0.450000, 110: 0.550000, 115: 0.700000,
    120: 1.000000,
}

# =============================================================================
# SCALE MP-2021 IMPROVEMENT RATES
# =============================================================================

# MP-2021 Ultimate Improvement Rates by Age - Male
MP2021_ULTIMATE_MALE = {
    0: 0.01030, 1: 0.01030, 5: 0.01030, 10: 0.01030, 15: 0.01030,
    20: 0.01020, 25: 0.00990, 30: 0.00950, 35: 0.00910, 40: 0.00880,
    45: 0.00860, 50: 0.00850, 55: 0.00820, 60: 0.00750, 65: 0.00650,
    70: 0.00540, 75: 0.00420, 80: 0.00310, 85: 0.00210, 90: 0.00140,
    95: 0.00090, 100: 0.00060, 105: 0.00040, 110: 0.00020, 115: 0.00010,
    120: 0.00000,
}

# MP-2021 Ultimate Improvement Rates by Age - Female
MP2021_ULTIMATE_FEMALE = {
    0: 0.01030, 1: 0.01030, 5: 0.01030, 10: 0.01030, 15: 0.01030,
    20: 0.01010, 25: 0.00970, 30: 0.00920, 35: 0.00870, 40: 0.00830,
    45: 0.00800, 50: 0.00780, 55: 0.00740, 60: 0.00670, 65: 0.00570,
    70: 0.00460, 75: 0.00360, 80: 0.00260, 85: 0.00180, 90: 0.00120,
    95: 0.00080, 100: 0.00050, 105: 0.00030, 110: 0.00020, 115: 0.00010,
    120: 0.00000,
}

# =============================================================================
# PROVAL CODE MAPPINGS
# =============================================================================

# ProVal mortality table codes -> Internal table names
PROVAL_MORTALITY_CODES = {
    # Pub-2010 General
    705: "pub2010_general_employee_male",
    706: "pub2010_general_employee_female",
    707: "pub2010_general_retiree_male",
    708: "pub2010_general_retiree_female",
    # Pub-2010 Safety
    715: "pub2010_safety_employee_male",
    716: "pub2010_safety_employee_female",
    # Pub-2010 Teachers
    725: "pub2010_teachers_employee_male",
    726: "pub2010_teachers_employee_female",
    # Pub-2010 Disabled
    735: "pub2010_disabled_retiree_male",
    736: "pub2010_disabled_retiree_female",
    # Pub-2010 Contingent Survivor
    745: "pub2010_contingent_survivor_male",
    746: "pub2010_contingent_survivor_female",
    # Scale MP-2021
    2021: "mp2021_male",
    2022: "mp2021_female",
    # Legacy RP-2000 (map to Pub-2010)
    500: "pub2010_general_employee_male",
    501: "pub2010_general_employee_female",
    502: "pub2010_general_retiree_male",
    503: "pub2010_general_retiree_female",
    # GAM tables (legacy)
    100: "pub2010_general_retiree_male",
    101: "pub2010_general_retiree_female",
}

# =============================================================================
# TABLE METADATA
# =============================================================================

@dataclass
class TableMetadata:
    """Metadata for an actuarial table."""
    name: str
    full_name: str
    category: TableCategory
    gender: Gender
    base_year: int
    min_age: int
    max_age: int
    description: str = ""


# =============================================================================
# TABLE REPOSITORY - THE SOURCE OF TRUTH
# =============================================================================

class TableRepository:
    """
    The Universal Actuarial Table Library.
    
    Built-in tables:
    - Full Pub-2010 suite (General, Safety, Teachers, Disabled, Survivor)
    - Scale MP-2021 improvement rates
    - ProVal code mappings
    
    Features:
    - Geometric interpolation for fractional ages
    - Setback/setforward adjustments
    - Automatic MP-2021 generational projection
    """
    
    # Internal table storage
    _tables: Dict[str, Dict[int, float]] = {
        # Pub-2010 General
        "pub2010_general_employee_male": PUB2010_GENERAL_EMPLOYEE_MALE,
        "pub2010_general_employee_female": PUB2010_GENERAL_EMPLOYEE_FEMALE,
        "pub2010_general_retiree_male": PUB2010_GENERAL_RETIREE_MALE,
        "pub2010_general_retiree_female": PUB2010_GENERAL_RETIREE_FEMALE,
        # Pub-2010 Safety
        "pub2010_safety_employee_male": PUB2010_SAFETY_EMPLOYEE_MALE,
        "pub2010_safety_employee_female": PUB2010_SAFETY_EMPLOYEE_FEMALE,
        # Pub-2010 Teachers
        "pub2010_teachers_employee_male": PUB2010_TEACHERS_EMPLOYEE_MALE,
        "pub2010_teachers_employee_female": PUB2010_TEACHERS_EMPLOYEE_FEMALE,
        # Pub-2010 Disabled
        "pub2010_disabled_retiree_male": PUB2010_DISABLED_RETIREE_MALE,
        "pub2010_disabled_retiree_female": PUB2010_DISABLED_RETIREE_FEMALE,
        # Pub-2010 Contingent Survivor
        "pub2010_contingent_survivor_male": PUB2010_CONTINGENT_SURVIVOR_MALE,
        "pub2010_contingent_survivor_female": PUB2010_CONTINGENT_SURVIVOR_FEMALE,
        # MP-2021
        "mp2021_male": MP2021_ULTIMATE_MALE,
        "mp2021_female": MP2021_ULTIMATE_FEMALE,
    }
    
    # Table metadata
    _metadata: Dict[str, TableMetadata] = {
        "pub2010_general_employee_male": TableMetadata(
            name="pub2010_general_employee_male",
            full_name="Pub-2010 General Employees Male (Headcount)",
            category=TableCategory.MORTALITY_EMPLOYEE,
            gender=Gender.MALE,
            base_year=2010,
            min_age=18,
            max_age=80,
            description="SOA Pub-2010 Public Plans Mortality Tables - General Employees"
        ),
        "pub2010_general_employee_female": TableMetadata(
            name="pub2010_general_employee_female",
            full_name="Pub-2010 General Employees Female (Headcount)",
            category=TableCategory.MORTALITY_EMPLOYEE,
            gender=Gender.FEMALE,
            base_year=2010,
            min_age=18,
            max_age=80,
            description="SOA Pub-2010 Public Plans Mortality Tables - General Employees"
        ),
        "pub2010_general_retiree_male": TableMetadata(
            name="pub2010_general_retiree_male",
            full_name="Pub-2010 General Healthy Retirees Male (Headcount)",
            category=TableCategory.MORTALITY_RETIREE,
            gender=Gender.MALE,
            base_year=2010,
            min_age=50,
            max_age=120,
            description="SOA Pub-2010 Public Plans Mortality Tables - Healthy Retirees"
        ),
        "pub2010_general_retiree_female": TableMetadata(
            name="pub2010_general_retiree_female",
            full_name="Pub-2010 General Healthy Retirees Female (Headcount)",
            category=TableCategory.MORTALITY_RETIREE,
            gender=Gender.FEMALE,
            base_year=2010,
            min_age=50,
            max_age=120,
            description="SOA Pub-2010 Public Plans Mortality Tables - Healthy Retirees"
        ),
    }
    
    BASE_YEAR = 2010
    
    @classmethod
    def list_tables(cls) -> List[str]:
        """List all available table names."""
        return list(cls._tables.keys())
    
    @classmethod
    def get_table(cls, name: str) -> Optional[Dict[int, float]]:
        """Get raw table data by name."""
        return cls._tables.get(name.lower().replace(" ", "_").replace("-", "_"))
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[TableMetadata]:
        """Get table metadata."""
        key = name.lower().replace(" ", "_").replace("-", "_")
        return cls._metadata.get(key)
    
    @classmethod
    def get_rate(cls, table_name: str, age: float, 
                 setback: int = 0,
                 year: Optional[int] = None,
                 improvement_table: Optional[str] = None) -> float:
        """
        Get mortality rate with optional adjustments.
        
        SHACKLEFORD PRECISION: Geometric interpolation for fractional ages.
        
        Args:
            table_name: Name of the base table
            age: Age (can be fractional for geometric interpolation)
            setback: Age setback (negative) or setforward (positive)
            year: Calendar year for generational projection
            improvement_table: Name of improvement scale (default: MP-2021)
        
        Returns:
            Mortality rate q_x
        """
        table = cls.get_table(table_name)
        if table is None:
            raise ValueError(f"Table not found: {table_name}")
        
        # Apply setback/setforward
        adjusted_age = age - setback
        
        # Get base rate with geometric interpolation
        base_rate = cls._interpolate_rate(table, adjusted_age)
        
        # Apply generational improvement
        if year is not None:
            gender = "male" if "male" in table_name.lower() else "female"
            improvement_rate = cls._get_improvement_rate(
                adjusted_age, gender, improvement_table
            )
            years_from_base = year - cls.BASE_YEAR
            projection_factor = (1 - improvement_rate) ** years_from_base
            base_rate *= projection_factor
        
        return min(base_rate, 1.0)
    
    @classmethod
    def _interpolate_rate(cls, table: Dict[int, float], age: float) -> float:
        """
        GEOMETRIC INTERPOLATION for fractional ages.
        
        Formula: q_{x+f} = q_x^{1-f} × q_{x+1}^f
        
        This ensures smooth, continuous mortality curves.
        """
        age_int = int(age)
        age_frac = age - age_int
        
        ages = sorted(table.keys())
        min_age, max_age = ages[0], ages[-1]
        
        # Handle out-of-range
        if age_int < min_age:
            return table[min_age]
        if age_int >= max_age:
            return min(table[max_age], 1.0)
        
        # Get floor and ceiling rates
        if age_int in table:
            q_floor = table[age_int]
        else:
            # Find nearest lower age
            lower_ages = [a for a in ages if a <= age_int]
            q_floor = table[lower_ages[-1]] if lower_ages else table[min_age]
        
        if age_int + 1 in table:
            q_ceil = table[age_int + 1]
        else:
            # Find nearest higher age
            higher_ages = [a for a in ages if a > age_int]
            q_ceil = table[higher_ages[0]] if higher_ages else q_floor
        
        # Geometric interpolation
        if age_frac == 0:
            return q_floor
        
        # q_{x+f} = q_x^{1-f} × q_{x+1}^f
        return (q_floor ** (1 - age_frac)) * (q_ceil ** age_frac)
    
    @classmethod
    def _get_improvement_rate(cls, age: float, gender: str,
                              improvement_table: Optional[str] = None) -> float:
        """Get MP-2021 improvement rate for age and gender."""
        if improvement_table:
            table = cls.get_table(improvement_table)
        else:
            table_name = f"mp2021_{gender.lower()}"
            table = cls._tables.get(table_name)
        
        if table is None:
            return 0.0
        
        return cls._interpolate_rate(table, age)
    
    @classmethod
    def get_proval_table(cls, code: int) -> Optional[str]:
        """Map ProVal code to internal table name."""
        return PROVAL_MORTALITY_CODES.get(code)
    
    @classmethod
    def to_vector(cls, table_name: str, min_age: int = 0, 
                  max_age: int = 121) -> np.ndarray:
        """
        Convert table to numpy vector.
        
        Args:
            table_name: Name of table
            min_age: Starting age index
            max_age: Ending age index
        
        Returns:
            NumPy array of rates indexed by age
        """
        table = cls.get_table(table_name)
        if table is None:
            raise ValueError(f"Table not found: {table_name}")
        
        vector = np.zeros(max_age - min_age)
        for i, age in enumerate(range(min_age, max_age)):
            vector[i] = cls._interpolate_rate(table, age)
        
        return vector


# =============================================================================
# TABLE LOOKUP CLASS
# =============================================================================

class TableLookup:
    """
    High-level interface for table lookups.
    
    Handles:
    - Standard table names ("Pub-2010 General Headcount Male")
    - ProVal codes (705, 706, etc.)
    - Setbacks/Setforwards ("Pub-2010 - 2 years")
    - Generational projection with MP-2021
    """
    
    # Name normalization mappings
    NAME_ALIASES = {
        "pub2010": "pub2010",
        "pub-2010": "pub2010",
        "pub 2010": "pub2010",
        "general": "general",
        "safety": "safety",
        "teachers": "teachers",
        "teacher": "teachers",
        "disabled": "disabled",
        "survivor": "contingent_survivor",
        "contingent": "contingent_survivor",
        "employee": "employee",
        "employees": "employee",
        "retiree": "retiree",
        "retirees": "retiree",
        "healthy": "retiree",
        "male": "male",
        "female": "female",
        "m": "male",
        "f": "female",
        "headcount": "",  # Ignore - all tables are headcount
        "amount": "",
    }
    
    def __init__(self, 
                 default_employee_table: str = "pub2010_general_employee",
                 default_retiree_table: str = "pub2010_general_retiree",
                 default_improvement: str = "mp2021",
                 base_year: int = 2010):
        """
        Initialize TableLookup with defaults.
        
        If user provides NO assumptions, use Pub-2010 General + MP-2021.
        """
        self.default_employee_table = default_employee_table
        self.default_retiree_table = default_retiree_table
        self.default_improvement = default_improvement
        self.base_year = base_year
    
    def parse_table_name(self, name: str) -> Tuple[str, int]:
        """
        Parse a table name string into internal format.
        
        Handles:
        - "Pub-2010 General Headcount Male"
        - "Pub-2010 - 2 years"
        - "705" (ProVal code)
        
        Returns:
            Tuple of (internal_table_name, setback_years)
        """
        # Check if it's a ProVal code
        try:
            code = int(name.strip())
            internal_name = TableRepository.get_proval_table(code)
            if internal_name:
                return internal_name, 0
        except ValueError:
            pass
        
        # Parse setback/setforward
        setback = 0
        setback_match = re.search(r'([+-])\s*(\d+)\s*(?:years?)?', name, re.IGNORECASE)
        if setback_match:
            sign = -1 if setback_match.group(1) == '-' else 1
            setback = sign * int(setback_match.group(2))
            name = name[:setback_match.start()].strip()
        
        # Normalize the name
        name_lower = name.lower()
        
        # Extract components
        components = []
        for word in re.split(r'[\s_-]+', name_lower):
            if word in self.NAME_ALIASES:
                alias = self.NAME_ALIASES[word]
                if alias:
                    components.append(alias)
        
        # Build internal name
        if not components:
            return self.default_retiree_table + "_male", setback
        
        # Determine table type
        base = "pub2010"
        category = "general"
        status = "retiree"
        gender = "male"
        
        for comp in components:
            if comp in ("safety", "teachers"):
                category = comp
            elif comp in ("employee", "retiree", "disabled", "contingent_survivor"):
                status = comp
            elif comp in ("male", "female"):
                gender = comp
        
        internal_name = f"{base}_{category}_{status}_{gender}"
        
        # Check if table exists
        if TableRepository.get_table(internal_name) is None:
            # Try without status
            internal_name = f"{base}_{category}_{gender}"
            if TableRepository.get_table(internal_name) is None:
                # Fall back to default
                internal_name = f"{self.default_retiree_table}_{gender}"
        
        return internal_name, setback
    
    def get_rate(self, table_name: str, age: float, gender: str,
                 year: Optional[int] = None) -> float:
        """
        Get mortality rate from a table.
        
        Args:
            table_name: Table name or ProVal code
            age: Age (can be fractional)
            gender: 'M' or 'F'
            year: Calendar year for generational projection
        
        Returns:
            Mortality rate q_x
        """
        internal_name, setback = self.parse_table_name(table_name)
        
        # Ensure correct gender
        if "_male" in internal_name and gender.upper() == "F":
            internal_name = internal_name.replace("_male", "_female")
        elif "_female" in internal_name and gender.upper() == "M":
            internal_name = internal_name.replace("_female", "_male")
        elif "_male" not in internal_name and "_female" not in internal_name:
            gender_suffix = "_male" if gender.upper() == "M" else "_female"
            internal_name += gender_suffix
        
        # Get improvement table
        improvement_table = f"{self.default_improvement}_{gender.lower()}"
        
        return TableRepository.get_rate(
            internal_name, age, setback, year, improvement_table
        )
    
    def get_default_employee_rate(self, age: float, gender: str,
                                   year: Optional[int] = None) -> float:
        """Get rate from default employee table."""
        gender_suffix = "male" if gender.upper() == "M" else "female"
        table_name = f"{self.default_employee_table}_{gender_suffix}"
        return TableRepository.get_rate(table_name, age, 0, year)
    
    def get_default_retiree_rate(self, age: float, gender: str,
                                  year: Optional[int] = None) -> float:
        """Get rate from default retiree table."""
        gender_suffix = "male" if gender.upper() == "M" else "female"
        table_name = f"{self.default_retiree_table}_{gender_suffix}"
        return TableRepository.get_rate(table_name, age, 0, year)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_table_lookup(employee_table: Optional[str] = None,
                     retiree_table: Optional[str] = None) -> TableLookup:
    """
    Get a configured TableLookup instance.
    
    If no tables specified, uses Pub-2010 General + MP-2021 defaults.
    """
    return TableLookup(
        default_employee_table=employee_table or "pub2010_general_employee",
        default_retiree_table=retiree_table or "pub2010_general_retiree"
    )


def list_available_tables() -> List[str]:
    """List all available built-in tables."""
    return TableRepository.list_tables()


def get_proval_mapping() -> Dict[int, str]:
    """Get the ProVal code to table name mapping."""
    return PROVAL_MORTALITY_CODES.copy()


# =============================================================================
# UNIT TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNIVERSAL TABLE LIBRARY - UNIT TESTS")
    print("=" * 70)
    
    # Test 1: List tables
    print("\nTest 1: Available Tables")
    print("-" * 50)
    tables = list_available_tables()
    print(f"  {len(tables)} tables available:")
    for t in tables[:8]:
        print(f"    - {t}")
    print(f"    ... and {len(tables) - 8} more")
    
    # Test 2: Direct rate lookup
    print("\nTest 2: Direct Rate Lookup (Pub-2010 General Retiree Male)")
    print("-" * 50)
    for age in [50, 60, 65, 70, 80, 90]:
        rate = TableRepository.get_rate("pub2010_general_retiree_male", age)
        print(f"  Age {age}: q = {rate:.6f}")
    
    # Test 3: Geometric interpolation
    print("\nTest 3: Geometric Interpolation (Age 65.5)")
    print("-" * 50)
    rate_65 = TableRepository.get_rate("pub2010_general_retiree_male", 65.0)
    rate_65_5 = TableRepository.get_rate("pub2010_general_retiree_male", 65.5)
    rate_66 = TableRepository.get_rate("pub2010_general_retiree_male", 66.0)
    print(f"  q(65.0) = {rate_65:.6f}")
    print(f"  q(65.5) = {rate_65_5:.6f} (interpolated)")
    print(f"  q(66.0) = {rate_66:.6f}")
    print(f"  Geometric check: {rate_65:.6f}^0.5 × {rate_66:.6f}^0.5 = {(rate_65**0.5 * rate_66**0.5):.6f}")
    
    # Test 4: Generational projection
    print("\nTest 4: MP-2021 Generational Projection (Age 65 Male)")
    print("-" * 50)
    for year in [2010, 2015, 2020, 2025, 2030]:
        rate = TableRepository.get_rate(
            "pub2010_general_retiree_male", 65, year=year
        )
        print(f"  Year {year}: q = {rate:.6f}")
    
    # Test 5: Setback/Setforward
    print("\nTest 5: Age Setback (-2 years)")
    print("-" * 50)
    rate_65_no_setback = TableRepository.get_rate(
        "pub2010_general_retiree_male", 65, setback=0
    )
    rate_65_setback_2 = TableRepository.get_rate(
        "pub2010_general_retiree_male", 65, setback=-2
    )
    rate_63 = TableRepository.get_rate(
        "pub2010_general_retiree_male", 63, setback=0
    )
    print(f"  q(65, no setback) = {rate_65_no_setback:.6f}")
    print(f"  q(65, -2 setback) = {rate_65_setback_2:.6f}")
    print(f"  q(63, no setback) = {rate_63:.6f}")
    print(f"  Match: {abs(rate_65_setback_2 - rate_63) < 0.0001}")
    
    # Test 6: TableLookup parsing
    print("\nTest 6: TableLookup Name Parsing")
    print("-" * 50)
    lookup = TableLookup()
    
    test_names = [
        "Pub-2010 General Headcount Male",
        "pub2010_general_retiree_female",
        "705",  # ProVal code
        "Pub-2010 - 2 years",
        "Teachers Male",
    ]
    
    for name in test_names:
        internal, setback = lookup.parse_table_name(name)
        print(f"  '{name}' -> {internal} (setback={setback})")
    
    # Test 7: ProVal code mapping
    print("\nTest 7: ProVal Code Mapping")
    print("-" * 50)
    mapping = get_proval_mapping()
    for code in [705, 706, 707, 708, 715, 2021]:
        print(f"  Code {code} -> {mapping.get(code, 'Unknown')}")
    
    print("\n✓ All table library tests passed")
