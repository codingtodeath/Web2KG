"""Empty structure is absence of extractable structure, not proof of a blank page."""
from .page_structure import EmptyStructureError, structure_profile


def isolated_profile():
    return {'structure': [], 'controls': [], 'repeated': [], 'regions': [], 'has_main': False,
            'dom_fallback_nodes': 0, 'region_features': {}, 'isolated': True,
            'evidence': {'sufficient': False, 'control_patterns': 0, 'content_units': 0}}


def admitted(observation):
    # Missing evidence is unknown, never confirmation that a page is blank.
    return observation.get('page_content', {}).get('kind') not in {'blank', 'loading_only'}


def observation_profile(observation):
    try:
        return structure_profile(observation['ax'], observation['dom'])
    except EmptyStructureError:
        return isolated_profile()
