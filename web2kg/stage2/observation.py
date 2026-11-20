"""Small observation adaptations for our Agent; upstream comparisons stay unchanged."""
import importlib
import re
import unicodedata


TEXT_ROLES = {'StaticText', 'LabelText', 'caption'}
NUMERIC_VALUE = re.compile(r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)\s*%?')
NUMERIC_TEXT_POLICY = 'standalone-value-v1'


def is_numeric_text(text):
    """A complete numeric value, optionally signed, percent or currency marked."""
    value = unicodedata.normalize('NFKC', text).strip().replace('\N{MINUS SIGN}', '-')
    if not value:
        return False
    if unicodedata.category(value[0]) == 'Sc':
        value = value[1:].strip()
    elif len(value) > 1 and value[0] in '+-' and unicodedata.category(value[1]) == 'Sc':
        value = value[0] + value[2:].strip()
    elif unicodedata.category(value[-1]) == 'Sc':
        value = value[:-1].strip()
    return NUMERIC_VALUE.fullmatch(value) is not None


def install_numeric_text_guard():
    """Protect numeric text during one task; return a restore hook for its finally."""
    opt = importlib.import_module('webNavigator.obs_opt')
    original_clean = opt.action_remove_unwanted_characters
    original_deduplicate = opt.action_remove_redundant_statictext_node

    def numeric(node):
        return node.role in TEXT_ROLES and is_numeric_text(node.name)

    def clean(node):
        # Preserve units/signs before the upstream cleaner removes %, currency or +.
        if not numeric(node):
            original_clean(node)

    def deduplicate(node):
        if not numeric(node):
            original_deduplicate(node)

    opt.action_remove_unwanted_characters = clean
    opt.action_remove_redundant_statictext_node = deduplicate

    def restore():
        opt.action_remove_unwanted_characters = original_clean
        opt.action_remove_redundant_statictext_node = original_deduplicate

    return restore
