# Changelog

## v0.3.7

- On whole-script `run_tool_script` timeout, cancel any outstanding bridged
  tool/search calls instead of letting them continue in the background.
- Return the configured `code_timeout` and number of cancelled bridged calls in
  timeout errors.

## v0.3.6

- Changed `list_servers()` to return a compact server inventory instead of
  listing every function on every server.
- Updated discovery guidance so models use `search_tools()` for task-specific
  tool names and schemas, and reserve `list_servers()` for integration
  visibility.

## v0.3.5

- Added `code_tool_call_timeout` to cap each tool/search operation inside
  `run_tool_script`.
- Timed-out tool calls are cancelled and reported as explicit script errors
  instead of leaving the workflow stuck in an executing state.
- Guarded worker completion against late writes after a whole-script timeout.

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
