"""Bounded keyword windows over one immutable AX observation."""
import re

import yaml


SEARCH_SPEC = """- search_ax [keyword]: Search the full current AX text using a literal,
case-insensitive keyword when the needed content is missing. Use this command alone.
It reveals up to 800 lines around one match and merges them into the current AX view.
The first not-yet-visible match is preferred; repeated searches can reach later matches.
There are at most 3 searches and 2500 displayed AX lines per browser decision.
Misses and malformed searches consume a search. Searches do not operate the browser.
After the limit, choose an available browser action (or stop); do not search again.
AX IDs are unchanged. Source line numbers are NOT action IDs. Gaps may exist between
displayed source ranges. A browser action starts a new observation and search budget."""


class AXSearch:
    initial_lines = 800
    window_lines = 800
    max_lines = 2500
    max_searches = 3

    def __init__(self, text):
        self.lines = text.splitlines()
        self.selected = set(range(min(self.initial_lines, len(self.lines))))
        self.searches = 0
        self.last_result = None

    @property
    def available(self):
        return self.searches < self.max_searches and len(self.selected) < self.max_lines

    @property
    def text(self):
        return '\n'.join(self.lines[i] for i in sorted(self.selected))

    def ranges(self):
        ranges = []
        for index in sorted(self.selected):
            line = index + 1
            if ranges and ranges[-1][1] == line - 1:
                ranges[-1][1] = line
            else:
                ranges.append([line, line])
        return ranges

    def status(self):
        return dict(search_available=self.available, searches_used=self.searches,
                    search_limit=self.max_searches, shown_lines=len(self.selected),
                    total_lines=len(self.lines), line_limit=self.max_lines,
                    source_ranges=self.ranges(), last_search=self.last_result)

    def observation_text(self):
        return self.text + '\n\nAX_SEARCH_STATUS (not AX nodes):\n' + yaml.safe_dump(
            self.status(), allow_unicode=True, sort_keys=False)

    def search(self, command):
        if not self.available:
            return dict(status='blocked', reason='search_or_line_limit')
        self.searches += 1
        match = re.fullmatch(r'search_ax\s+\[([^\[\]\r\n]+)\]', command.strip())
        keyword = match[1].strip() if match else ''
        result = dict(status='invalid' if not keyword else 'not_found', keyword=keyword,
                      added_lines=0)
        if keyword:
            hits = [i for i, line in enumerate(self.lines) if keyword.casefold() in line.casefold()]
            result['matches'] = len(hits)
            if hits:
                hit = next((i for i in hits if i not in self.selected), hits[0])
                start = max(0, min(hit - self.window_lines // 2,
                                   len(self.lines) - self.window_lines))
                end = min(len(self.lines), start + self.window_lines)
                unseen = set(range(start, end)) - self.selected
                # Near the cap retain the match and its closest context first.
                added = sorted(unseen, key=lambda i: (abs(i - hit), i))[:self.max_lines - len(self.selected)]
                self.selected.update(added)
                result.update(status='found', match_line=hit + 1, window=[start + 1, end],
                              added_lines=len(added))
        self.last_result = result
        return result
