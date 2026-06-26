# Advanced Tool Use for Open WebUI

Tool Search + Programmatic Tool Calling for Open WebUI.

Search across MCP, OpenAPI, and local tools on demand instead of loading every
tool definition into context. Then call one tool directly or run a sandboxed
Python workflow that loops, calls tools in parallel, filters results, and
returns only the final output to chat.

Attach one tool instead of dozens. It preserves per-user access controls,
reduces context bloat, and keeps intermediate tool results out of the
conversation.

**Security note:** Code mode is off by default. Enable it only in trusted
environments.

Source:
https://github.com/Joshua-Dixon-AI/openwebui/tree/main/Tools/Advanced-Tool-Use

Zero patches to Open WebUI. Works with vanilla installs.

## Why it matters

Benchmark on a real multi-step Databricks workflow, comparing Programmatic Tool
Calling with a standard one-call-per-turn MCP loop:

| Metric | Advanced Tool Use | Standard MCP | Improvement |
|---|---:|---:|---:|
| Turns | 3 | 17 | 82% fewer |
| Input tokens | 7.7K | 826.6K | 107x less |
| Output tokens | 744 | 8.1K | 10.8x less |
| Peak context | 3.4K | 100.7K | 97% smaller |
| Cost | $0.0007 | $0.0177 | 96% cheaper |

*Single workload; gains scale with the number of tool calls and the size of
intermediate results. Your mileage will vary, but the direction is consistent:
more tool calls + bigger intermediate data → bigger win.*

## How it works

- **Tool index** — on first use the tool builds an index of every MCP server,
  OpenAPI tool server, and local tool in your Open WebUI install. It's cached
  with a TTL and rebuilt on demand.
- **Per-user access control** — every search, listing, and call is filtered
  against the *calling* user's access grants (`has_connection_access` for
  servers, tool grants for local tools). Users only ever see and reach what
  they're permitted to use.
- **Hybrid search** — BM25 keyword scoring + semantic similarity using Open
  WebUI's own embedding function (falls back to keyword-only if no embedding
  model is configured).
- **Output limits** — single `call_tool` responses can be capped per server to
  tame verbose tools (e.g. Atlassian).
- **Programmatic Tool Calling** — `run_tool_script` executes model-written
  Python in a restricted sandbox. Only what the script `print()`s returns to the
  model; intermediate data never enters the context window.

## Functions exposed to the model

| Function | Purpose |
|---|---|
| `search_tools(query, limit?)` | Hybrid search over accessible tools; returns names, descriptions, parameter schemas. |
| `list_servers()` | Compact list of accessible tool servers; returns server ids, names, types, and function counts only. |
| `call_tool(server_id, function_name, arguments)` | Execute a single tool, with per-server output limits applied. |
| `run_tool_script(code)` | **Programmatic Tool Calling** — orchestrate many tools in one Python script. |
| `refresh_index()` | Force an index rebuild after adding tools/servers. |

Inside `run_tool_script`, the model has: `await call(...)`, `await search(...)`,
`await gather(...)`, `rows(result)` (normalizes SQL/tabular results), `print(...)`,
safe builtins, and these pre-loaded **sanitised facades** (no `import` needed or
allowed): `json` (dumps/loads), `math`, `re`, `datetime`, `collections`,
`itertools`, `textwrap`, `string`. Use f-strings or concatenation for string
formatting; `str.format()` and `format_map()` are intentionally blocked.

## Install

1. In Open WebUI: **Workspace → Tools → +** and paste the contents of
   `tool_orchestrator.py`. Save.
2. Attach **only this tool** to the model(s) you want. Do **not** attach your
   individual MCP/OpenAPI servers to the model — they just need to exist in
   **Admin → Settings → Tools / Tool Servers**; Advanced Tool Use discovers them.
3. To enable Programmatic Tool Calling, open the tool's **Valves** (gear icon)
   and turn on **`enable_code_mode`**.

### Recommended system prompt

```
You have Advanced Tool Use, not the underlying tools directly. Discovered
functions (e.g. execute_sql_read_only) are NOT callable as top-level tool calls.

To use any capability:
  • Discover tools:     search_tools("...")
  • List integrations:  list_servers()
  • Run ONE call:       call_tool(server_id, function_name, arguments)
  • Run MULTIPLE calls: run_tool_script(code)   ← strongly preferred

Use list_servers only to inspect which integrations are available. It does not
return tool schemas or function names; use search_tools for task-specific
discovery.

Use run_tool_script whenever a task needs 3+ calls, parallel calls, loops, or
filtering/aggregating results — it keeps intermediate data out of context.
Inside a script: await call(server_id, function_name, args_dict),
await gather(*...), rows(result) to flatten SQL results, print(...) to return.
```

## Valves

| Valve | Default | Description |
|---|---|---|
| `search_top_k` | 5 | Results returned per tool search. |
| `index_ttl_seconds` | 300 | Seconds before the tool index is rebuilt. |
| `mcp_discovery_timeout` | 10 | Max seconds to enumerate each MCP server during indexing. |
| `embedding_timeout` | 20 | Max seconds for embeddings before falling back to keyword-only. |
| `semantic_weight` | 0.6 | Hybrid balance: 1.0 = pure semantic, 0.0 = pure keyword. |
| `default_max_output_tokens` | 8000 | Default cap for a single `call_tool` response (0 = unlimited). |
| `output_truncation_strategy` | head_tail | `head_tail` (keep start + end) or `truncate`. |
| `server_token_limits` | "" | Per-server overrides, one per line: `server_id=limit`. |
| `enable_code_mode` | **false** | Enable `run_tool_script` (executes model-written code). |
| `code_timeout` | 60 | Max wall-clock seconds per script (also caps CPU loops). |
| `code_tool_call_timeout` | 30 | Max seconds per tool/search operation inside `run_tool_script` (0 = unlimited). |
| `code_max_calls` | 50 | Max tool calls a single script may make (0 = unlimited). |
| `code_max_output_chars` | 24000 | Max characters of script output returned to the model. |
| `debug_events` | false | Show step-by-step status events in chat (debugging). |

Example per-server output limits:

```
atlassian=4000
databricks=6000
```

## Security

`run_tool_script` **executes model-generated Python**. It runs in a restricted
in-process sandbox:

- AST allow-list — no arbitrary syntax; classes, `with`, lambdas-with-escapes,
  etc. are rejected.
- No dangerous imports (`import` is disabled entirely; safe data utilities are
  pre-loaded as sanitised facades); no `eval`, `exec`, `open`, `getattr`,
  `__import__`, etc.
- Blocked `_`-prefixed attribute access and `str.format()` / `format_map()`
  traversal — kills classic hidden-attribute escapes.
- Runs in a **separate thread** with a wall-clock timeout that also **interrupts
  CPU-bound loops** and cancels outstanding bridged tool calls, so a runaway
  script cannot block the server.
- A **per-user access check on every call**, plus a **tool-call cap** and
  per-call timeout so one slow MCP/API request cannot hang the whole workflow.

It is a strong boundary, **not full process isolation**: it does **not hard-cap
memory** (a giant allocation could OOM the host), and a thread wedged in a
non-Python C call can't be force-killed (the main loop is freed, pending tool
calls are cancelled, and the request returns, but the thread may linger). For
these reasons `enable_code_mode` is
**off by default** — enable it only in environments you trust. If you need to run
this in a hostile multi-user setting, put a true subprocess/container sandbox in
front of it.

## Known limitations

- **Per-user MCP catalogs:** the tool index is shared and enumerated using the
  credentials of whoever triggers a rebuild. If an MCP server exposes a
  *different* tool list per authenticated user, the cached catalog reflects the
  builder's view. Access to *use* a server is always checked per-user; only the
  visible catalog can differ. Most servers expose a stable catalog.
- **Tested single-user.** Access-isolation logic is implemented and reasoned
  through, but hasn't been battle-tested across many users/groups or under load.
- **Result shapes:** `rows()` normalizes common SQL/tabular shapes (Databricks
  statement results, `{"rows": [...]}`, list-of-lists). Exotic shapes may need
  manual parsing in the script.

## License

MIT
