"""URL identity and mechanical candidate validation; no risk keyword policy."""
from urllib.parse import urlsplit


def origin(url):
    parsed = urlsplit(url)
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)


def assess(action, start_url=None):
    if action.get('disabled'):
        return False, 'disabled'
    if not action.get('xpath') and not (action.get('role') and 'name' in action):
        return False, 'missing_locator'
    return True, 'locatable_control'
