# Changelog

## v0.3.4

- Renamed the published tool to **Advanced Tool Use**.
- Added Tool Search across MCP, OpenAPI, and local Open WebUI tools.
- Added Programmatic Tool Calling via `run_tool_script` for multi-step,
  parallel, filtered, and aggregated tool workflows.
- Added per-user access checks for search, listing, and every tool call.
- Added per-server output limits for single `call_tool` responses.
- Added sandbox hardening for code mode:
  - imports disabled entirely
  - safe utilities exposed as sanitised facades instead of real modules
  - blocked hidden/dunder attribute access
  - blocked `str.format()` and `format_map()` traversal
  - validated function annotations
  - added wall-clock timeout, CPU-loop interruption, and tool-call caps
- Kept code mode off by default for safer public installs.
