"""Lossless syntax adaptation for the original Agent's existing action language."""
import re


def validate_single_action(text):
    from ..stage2.action_notes import parse_sections
    from ..common.responses import ResponsesError
    try:
        action = parse_sections(text, ['action'], require_single_action=True)['action']
        if not single_action(action) or action.startswith('note '):
            raise ValueError('ACTION must contain exactly one complete command, without a second action or trailing prose')
    except ValueError as error:
        raise ResponsesError(str(error), kind='output:actor_action_invalid') from None


def validate_navigation_domain(text, domains):
    """Validate against the live Actor registry, without guessing URL aliases."""
    from ..stage2.action_notes import parse_sections
    from ..common.responses import ResponsesError
    action = parse_sections(text, ['action'], require_single_action=True)['action']
    if not action.startswith('optimal_navigate '):
        return
    # Use the same argument extraction as the upstream dispatch loop.
    match = re.search(r'optimal_navigate ?\[([\s\S]+?)\]\s*\[([\s\S]+?)\]\s*\[([\s\S]+?)\]', action)
    if not match:
        raise ResponsesError('optimal_navigate requires three nonempty bracketed arguments: thinking, site ID, query',
                             kind='output:actor_action_invalid')
    domain = match.group(2).strip()
    if domain not in domains:
        raise ResponsesError(
            f'Invalid optimal_navigate domain {domain!r}. Available navigation site IDs: {list(domains)!r}. '
            'Use exactly one listed site ID in the second bracket, not a hostname or URL. '
            'The third bracket describes the destination page. No navigation was executed.',
            kind='output:actor_domain_invalid')


def single_action(text):
    """Accept only one complete supported command; never infer an action from prose."""
    text = text.strip()
    if text == 'go_back':
        return True
    match = re.fullmatch(r'(click|type|stop|note|optimal_navigate|search_ax)\s+(.+)', text, re.S)
    if not match:
        return False
    command, tail = match.groups()
    if command == 'search_ax':
        return bool(re.fullmatch(r'\[[^\[\]\r\n]+\]', tail) and tail[1:-1].strip())
    args, depth, start = [], 0, None
    for index, char in enumerate(tail):
        if char == '[':
            if depth == 0:
                start = index + 1
            depth += 1
        elif char == ']':
            depth -= 1
            if depth < 0:
                return False
            if depth == 0:
                args.append(tail[start:index])
        elif depth == 0 and not char.isspace():
            return False
    if depth:
        return False
    if command == 'click':
        return len(args) == 1 and args[0].isdigit()
    if command == 'type':
        return len(args) in (2, 3) and args[0].isdigit() and (
            len(args) == 2 or bool(re.fullmatch(r'(?:press_enter_after=)?[01]', args[2])))
    return len(args) == (3 if command == 'optimal_navigate' else 1)


def actor_output(messages, select_text):
    changes = []
    try:
        text = select_text(messages)
    except ValueError:
        # Some relays split the heading and its command across adjacent messages.
        # Only this exact continuation is accepted, never two independent actions.
        finals = [m for m in messages if m.get('phase') == 'final_answer']
        commentary = [m for m in messages if m.get('phase') == 'commentary']
        safe = all(m.get('role') in (None, 'assistant') and m.get('status') in (None, 'completed')
                   and all(p.get('type') == 'output_text' for p in m.get('content', []))
                   for m in messages)
        values = [''.join(p.get('text', '') for p in m.get('content', [])).strip() for m in commentary]
        empty_final = finals and all(not ''.join(p.get('text', '') for p in m.get('content', [])).strip() for m in finals)
        if not (safe and empty_final and len(values) == 2
                and messages.index(commentary[0]) + 1 == messages.index(commentary[1])
                and re.search(r'^ACTION:\s*$', values[0], re.M)
                and values[0].endswith('ACTION:')
                and len(re.findall(r'^ACTION:', values[0], re.M)) == 1
                and single_action(values[1])):
            raise
        text = values[0] + '\n' + values[1]
        changes.append('adjacent_action_continuation')
    if single_action(text):
        text = 'ACTION:\n' + text
        changes.append('bare_action_heading')
    section = re.search(r'(^ACTION:[ \t]*\n?)(.*?)(?=\n[A-Z][A-Z \d]*:|\Z)', text, re.M | re.S)
    if section:
        action = section[2].strip()
        # Convert only the terminal argument of one type command, never its content.
        if single_action(action) and action.startswith('type '):
            normalized = re.sub(r'\[press_enter_after=([01])\]$', r'[\1]', action)
            if normalized != action:
                text = text[:section.start(2)] + normalized + '\n' + text[section.end(2):]
                changes.append('named_enter_flag')
    return text, changes
