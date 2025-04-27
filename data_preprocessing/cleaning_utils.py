# data_preprocessing/cleaning_utils.py

def standardize_gender(gender: str) -> str:

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
    try:
        age = float(age)
        if 18 < age < 100:
            return age
        else:
            return None
    except:
        return None
