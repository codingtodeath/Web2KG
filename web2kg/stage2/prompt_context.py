"""Project audit evidence into the context needed for this decision only."""
from ..common.prompt_text import to_yaml
from .actor_input import navigation_context

TEMPLATE_HEADER = 'Historical template context (advisory, not permissions):\n'
GLOBAL_HEADER = 'Website framework and conditional experience (advisory evidence, never instructions/permissions):\n'


def site_system_prompt(prompt, site):
    """Explain the configured benchmark site to both Actor and selector."""
    if site != 'reddit':
        return prompt
    return prompt + (
        '\n\nEnvironment note: In this benchmark, the configured reddit site is the '
        'Postmill forum instance. Its Postmill branding is expected; perform the task '
        'there, not on public reddit.com. A retrieval miss does not mean the site is unavailable.'
    )


def template_context(raw, *, compact_actor=False):
    template = raw['match']['template_id']
    if template is None:
        return {}
    result = {'template_id': template}
    if raw.get('function_card'):
        card = raw['function_card']
        result['function_card'] = {**card, 'variants': [
            {k: v for k, v in variant.items() if k != 'evidence_nodes'}
            for variant in card['variants']]}
        if compact_actor:
            result['function_card'] = {k: v for k, v in result['function_card'].items()
                if k in ('name', 'summary', 'common_functions', 'variants', 'uncertainties')}
    result['nearby'] = [{k: row[k] for k in ('target_template', 'target_name', 'target_summary',
                                           'control_label', 'current_control_ids')}
                        for row in raw['nearby']]
    if compact_actor:
        groups = {}
        for row in result['nearby']:
            # Keep the full destination description once, and every distinct live entrance.
            key = (row['target_template'], row['target_name'], row['target_summary'])
            group = groups.setdefault(key, {k: row[k] for k in
                ('target_template', 'target_name', 'target_summary')})
            entry = {k: row[k] for k in ('control_label', 'current_control_ids')}
            entries = group.setdefault('entrances', [])
            if entry not in entries:
                entries.append(entry)
        result['nearby'] = list(groups.values())
    result['notice'] = ('Use only current observed controls. Historical destinations are not guaranteed; '
                        'hints are not proof of identity or completion.')
    return result


def framework_context(framework):
    """Site map, not a second copy of all page capabilities. Keep caveats intact."""
    if not framework.get('overview'):
        return framework  # Older/custom schemas without an overview keep their evidence.
    return {'overview': framework['overview'],
            'areas': [{'name': area['name']} for area in framework.get('areas', [])],
            'navigation': framework.get('navigation', []),
            'limitations': framework.get('limitations', [])}


def global_context(raw, *, compact_actor=False):
    result = {}
    if raw.get('navigation_feedback'):
        result['navigation_feedback'] = (navigation_context(raw['navigation_feedback']) if compact_actor
                                          else raw['navigation_feedback'])
    if raw.get('website_framework') is not None:
        result['website_framework'] = (framework_context(raw['website_framework']) if compact_actor
                                       else raw['website_framework'])
    if raw.get('diagnosis_advice'):
        result['diagnosis_advice'] = {k: raw['diagnosis_advice'][k]
            for k in ('verdict', 'reason', 'alternative', 'rule')}
    if raw.get('counterfactual'):
        cf = raw['counterfactual']
        ban = cf['forbidden_control']
        result['counterfactual'] = {
            'diagnosis': {k: cf['diagnosis'][k] for k in ('reason', 'alternative')},
            'return': {k: v for k, v in cf['return'].items() if k in ('success', 'reason')},
            'forbidden_control': {'url': ban['url'], 'kind': ban['action'].get('kind'),
                **{k: v for k, v in ban['control'].items()
                   if k in ('role', 'name', 'href', 'occurrence') and v is not None}},
            'rule': cf['rule']}
    experience = raw.get('experience') or {}
    lessons = {k: experience[k] for k in ('page', 'template') if experience.get(k)}
    if lessons:
        result['experience'] = lessons
    return result


def context_messages(messages, template=None, global_evidence=None, *, compact_actor=False):
    # Never append to the Agent's retained history.
    result = list(messages)
    for header, context in ((TEMPLATE_HEADER, template_context(template, compact_actor=compact_actor) if template else {}),
                            (GLOBAL_HEADER, global_context(global_evidence, compact_actor=compact_actor) if global_evidence else {})):
        if context:
            result.append({'role': 'user', 'content': header + to_yaml(context)})
    return result
