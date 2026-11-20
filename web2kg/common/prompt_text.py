"""YAML for structured model input; storage and output protocols stay separate."""
import yaml


class PromptDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _string(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:str', value,
                                   style='|' if '\n' in value else None)


PromptDumper.add_representer(str, _string)


def to_yaml(value):
    return yaml.dump(value, Dumper=PromptDumper, allow_unicode=True,
                     sort_keys=False, default_flow_style=False, width=120)
