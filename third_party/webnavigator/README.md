# WebNavigator execution dependency

Pinned upstream: `15c06b96c6f65f8fc24a7ed739f801780d65eb88` from https://github.com/fate-ubw/webNavigator.
Only execution Python modules and unused-provider import shims are retained. No graph, model, credentials, experimental output or generated logs are vendored.

Agent and action parser remain upstream code. Local adaptations are documented in `provenance.json`: browser authentication/HAR/scope, tree-copy performance, disabling an unused scorer download, and the user-authorized Verified JSON `stop` prompt (2026-11-10). The stop command keeps its original syntax; only its final answer follows the evaluator's public JSON schema. The original snapshot hashes remain in `files_sha256`, with the adapted prompt hash separately recorded in `local_prompt_overrides`. Frozen experiments retain their old prompts. Application adapters live in `web2kg/stage2/` and `web2kg/evaluation/`; do not fork the Agent to introduce another action vocabulary.
