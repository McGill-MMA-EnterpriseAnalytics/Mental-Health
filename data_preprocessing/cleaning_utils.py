"""
Utility functions for cleaning and standardizing data related to gender and age.

Includes:
- standardize_gender: Standardizes gender labels into 'Male', 'Female', or 'Other'.
- clean_age: Cleans and validates age values, keeping only reasonable age ranges.

"""

def standardize_gender(gender: str) -> str:
    """
    Standardize gender input to one of three categories: 'Male', 'Female', or 'Other'.

    Args:
        gender (str): Raw gender input.

    Returns:
        str: Standardized gender category.
    """
    if not isinstance(gender, str):
        return "Other"
    
    gender = gender.lower().strip()

    mapping = {
        'male': 'Male',
        'm': 'Male',
        'man': 'Male',
        'female': 'Female',
        'f': 'Female',
        'woman': 'Female',
        'trans-female': 'Other',
        'trans female': 'Other',
        'trans male': 'Other',
        'non-binary': 'Other',
        'genderqueer': 'Other',
        'agender': 'Other',
        'other': 'Other',
        'fluid': 'Other',
        'none': 'Other',
        '': 'Other'
    }

    return mapping.get(gender, 'Other')


def clean_age(age) -> float:
    """
    Clean and validate the age value.

    Converts input to float and ensures it falls within a reasonable range (18, 100).

    Args:
        age: Raw age input (can be numeric or string).

    Returns:
        float or None: Validated age if within range, otherwise None.
    """
    try:
        age = float(age)
        if 18 < age < 100:
            return age
        else:
            return None
    except (ValueError, TypeError):
        return None
