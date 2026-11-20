"""Task-local route confidence, counted per executed Actor decision, not API call."""
import math

POLICY = 'direction-confidence-v1'
DEFAULTS = dict(mode='off', immediate=.30, window=3, low=.60, low_count=2,
                cooldown=3, watchdog=10)


def parse_confidence(value):
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


class DirectionConfidence:
    def __init__(self, config=None, log=None):
        self.config = {**DEFAULTS, **(config or {})}
        c = self.config
        if (set(c) != set(DEFAULTS) or c['mode'] not in ('off', 'shadow', 'active')
                or any(type(c[k]) is not int or c[k] < 1 for k in ('window', 'low_count', 'cooldown', 'watchdog'))
                or c['low_count'] > c['window']
                or any(type(c[k]) not in (int, float) or parse_confidence(c[k]) is None for k in ('immediate', 'low'))
                or c['immediate'] > c['low']):
            raise ValueError('Invalid direction confidence policy')
        self.log = log
        self.completed = 0
        self.last_diagnosis = 0
        self.diagnosed = False
        self.segment = []
        self.proposals = 0

    @property
    def enabled(self):
        return self.config['mode'] != 'off'

    def propose(self, fields):
        """Called only after a valid one-action response, before finalize/execute."""
        self.proposals += 1
        score = parse_confidence(fields.get('direction confidence'))
        c = self.config
        values = (self.segment + [score])[-c['window']:]
        reasons = []
        cooling = self.diagnosed and self.completed - self.last_diagnosis < c['cooldown']
        if not cooling:
            if score is not None and score <= c['immediate']:
                reasons.append('direction_confidence_low')
            if (len(values) == c['window'] and all(v is not None for v in values)
                    and sum(values) / len(values) < c['low']
                    and sum(v < c['low'] for v in values) >= c['low_count']):
                reasons.append('direction_confidence_window')
        signal = dict(policy=POLICY, proposal=self.proposals, completed_actions=self.completed,
                      score=score, raw=fields.get('direction confidence'), window=values,
                      cooldown=cooling, would_trigger=reasons, mode=c['mode'],
                      action=fields.get('action'), observation=fields.get('observation description'))
        if self.log:
            self.log.emit('direction_confidence', **signal)
        fields['confidence audit'] = signal
        return signal

    def commit(self, fields):
        # One upstream Actor turn, including stop or a failed attempted action. Replay
        # edges, searches, discarded proposals and format retries never call this.
        self.completed += 1
        self.segment.append(parse_confidence(fields.get('direction confidence')))
        self.segment = self.segment[-self.config['window']:]
        if self.log:
            self.log.emit('direction_confidence_committed' if self.enabled else 'diagnosis_action_count', actor_action=self.completed,
                          proposal=fields.get('confidence audit', {}).get('proposal'),
                          score=self.segment[-1], action=fields.get('action'))

    def periodic_reasons(self):
        return ['ten_actions_without_diagnosis'] if self.completed - self.last_diagnosis >= self.config['watchdog'] else []

    def mark_diagnosis(self):
        self.last_diagnosis = self.completed
        self.diagnosed = True
        self.segment.clear()
        if self.log:
            self.log.emit('direction_confidence_reset' if self.enabled else 'diagnosis_action_clock_reset', completed_actions=self.completed)
