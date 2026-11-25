"""Common prospective experiment profiles; exact historical contracts stay explicit."""

RESEARCH_PROFILES = ("exploratory", "qualified")


def research_profile(value="exploratory"):
    if value not in RESEARCH_PROFILES:
        raise ValueError("research_profile must be exploratory or qualified")
    return value
