import re

MALE_PRONOUNS = ['he', 'him', 'his', 'himself']
FEMALE_PRONOUNS = ['she', 'her', 'hers', 'herself']

def is_gendered(sentence):
    sentence = sentence.lower()
    contains_male = any(re.search(r'\b{}\b'.format(m_pronoun), sentence) for m_pronoun in MALE_PRONOUNS)
    contains_female = any(re.search(r'\b{}\b'.format(f_pronoun), sentence) for f_pronoun in FEMALE_PRONOUNS)
    if contains_male and not contains_female:
        return "male"
    elif contains_female and not contains_male:
        return "female"
    elif contains_male and contains_female:
        return "both"
    return False