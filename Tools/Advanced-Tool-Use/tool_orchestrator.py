"""
title: Advanced Tool Use
author: Joshua Dixon
version: 0.3.4
license: MIT
description: >
    Tool Search + Programmatic Tool Calling for Open WebUI. Search across MCP,
    OpenAPI, and local tools on demand, then call one tool or run a sandboxed
    Python workflow that orchestrates many tool calls while returning only the
    final result to chat.
"""

import ast
import asyncio
import concurrent.futures
import json
import logging
import math
import re
import sys
import threading
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool index
# ---------------------------------------------------------------------------


class _ToolEntry:
    __slots__ = (
        "name", "description", "parameters", "server_id", "server_name",
        "server_type", "tool_id", "connection", "search_text", "embedding",
    )

    def __init__(self, name, description, parameters, server_id, server_name,
                 server_type, tool_id, connection=None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.server_id = server_id
        self.server_name = server_name
        self.server_type = server_type
        self.tool_id = tool_id
        self.connection = connection
        self.search_text = f"{server_name} {name} {description}"
        self.embedding: Optional[list[float]] = None


class _ToolIndex:
    def __init__(self):
        self.entries: list[_ToolEntry] = []
        self.built_at: float = 0.0
        self._idf: dict[str, float] = {}
        self._tokenized: list[list[str]] = []
        self._avg_dl: float = 1.0

    def is_stale(self, ttl: int) -> bool:
        return (time.time() - self.built_at) > ttl or not self.entries

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def build_bm25(self):
        n = len(self.entries)
        if n == 0:
            self._tokenized, self._avg_dl, self._idf = [], 1.0, {}
            return
        tokenized, df = [], {}
        for entry in self.entries:
            tokens = self._tokenize(entry.search_text)
            tokenized.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}
        self._tokenized = tokenized
        self._avg_dl = (sum(len(t) for t in tokenized) / n) or 1.0

    def bm25_score(self, query: str) -> list[float]:
        k1, b = 1.5, 0.75
        q_tokens = self._tokenize(query)
        scores = []
        for doc_tokens in self._tokenized:
            dl = len(doc_tokens)
            tf_map: dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            score = 0.0
            for qt in q_tokens:
                idf = self._idf.get(qt)
                if idf is None:
                    continue
                tf = tf_map.get(qt, 0)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avg_dl))
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# Sandbox for Programmatic Tool Calling
# ---------------------------------------------------------------------------

# Node types the model's orchestration code is allowed to use. Anything else
# (imports, class defs, with-blocks, yields, etc.) is rejected at compile time.
_ALLOWED_NODES = {
    ast.Module, ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda,
    ast.arguments, ast.arg, ast.Return, ast.Pass, ast.Break, ast.Continue,
    ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr,
    ast.For, ast.AsyncFor, ast.While, ast.If, ast.IfExp,
    ast.Try, ast.ExceptHandler, ast.Raise,
    ast.Await, ast.Call, ast.keyword,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.MatMult,
    ast.USub, ast.UAdd, ast.Invert,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.Dict, ast.Set, ast.List, ast.Tuple, ast.Starred,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Subscript, ast.Slice, ast.Name, ast.Attribute,
    ast.Load, ast.Store, ast.Del,
    ast.Constant, ast.JoinedStr, ast.FormattedValue,
}

# Modules pre-bound as facades inside the sandbox. We do NOT bind real module
# objects — they expose too much object graph (e.g. statistics.sys.modules["os"]).
# Instead we expose only the specific callables/constants each module needs to be
# useful for data-processing scripts. Update the tuples below to add more.
#
# The set of names advertised in validator error messages:
_SAFE_MODULE_NAMES = (
    "json", "math", "re", "datetime", "collections", "itertools", "textwrap", "string",
)


def _build_sandbox_globals() -> dict:
    """Return a dict of name → safe facade for pre-binding in the script sandbox.

    Every facade is a SimpleNamespace (or plain constant/function) that exposes
    only the symbols a data-processing orchestration script reasonably needs.
    No real module object is bound — this closes the `statistics.sys.modules`
    and similar object-graph escapes.
    """
    import json as _json
    import math as _math
    import re as _re
    import datetime as _dt
    import collections as _col
    import itertools as _it
    import textwrap as _tw
    import string as _st
    from types import SimpleNamespace

    g: dict = {}

    # json — the most commonly needed module; expose all safe public API
    g["json"] = SimpleNamespace(
        dumps=_json.dumps,
        loads=_json.loads,
        JSONDecodeError=_json.JSONDecodeError,
    )

    # math — constants + pure numeric functions only
    g["math"] = SimpleNamespace(
        pi=_math.pi, e=_math.e, tau=_math.tau, inf=_math.inf, nan=_math.nan,
        floor=_math.floor, ceil=_math.ceil, round=round,
        sqrt=_math.sqrt, log=_math.log, log2=_math.log2, log10=_math.log10,
        exp=_math.exp, pow=_math.pow,
        sin=_math.sin, cos=_math.cos, tan=_math.tan,
        asin=_math.asin, acos=_math.acos, atan=_math.atan, atan2=_math.atan2,
        degrees=_math.degrees, radians=_math.radians,
        fabs=_math.fabs, factorial=_math.factorial, gcd=_math.gcd,
        isnan=_math.isnan, isinf=_math.isinf, isfinite=_math.isfinite,
        hypot=_math.hypot, trunc=_math.trunc,
    )

    # re — pattern compilation + search/match/sub + constants
    g["re"] = SimpleNamespace(
        compile=_re.compile, search=_re.search, match=_re.match,
        fullmatch=_re.fullmatch, findall=_re.findall, finditer=_re.finditer,
        sub=_re.sub, subn=_re.subn, split=_re.split, escape=_re.escape,
        IGNORECASE=_re.IGNORECASE, MULTILINE=_re.MULTILINE,
        DOTALL=_re.DOTALL, VERBOSE=_re.VERBOSE,
        error=_re.error,
    )

    # datetime — types and constructors; no filesystem/timezone DB access
    g["datetime"] = SimpleNamespace(
        date=_dt.date,
        time=_dt.time,
        datetime=_dt.datetime,
        timedelta=_dt.timedelta,
        timezone=_dt.timezone,
        MINYEAR=_dt.MINYEAR,
        MAXYEAR=_dt.MAXYEAR,
    )

    # collections — data structures only
    g["collections"] = SimpleNamespace(
        Counter=_col.Counter,
        defaultdict=_col.defaultdict,
        OrderedDict=_col.OrderedDict,
        namedtuple=_col.namedtuple,
        deque=_col.deque,
        ChainMap=_col.ChainMap,
    )

    # itertools — all pure combinatorial generators
    g["itertools"] = SimpleNamespace(
        chain=_it.chain,
        islice=_it.islice,
        groupby=_it.groupby,
        product=_it.product,
        permutations=_it.permutations,
        combinations=_it.combinations,
        combinations_with_replacement=_it.combinations_with_replacement,
        accumulate=_it.accumulate,
        compress=_it.compress,
        dropwhile=_it.dropwhile,
        takewhile=_it.takewhile,
        filterfalse=_it.filterfalse,
        starmap=_it.starmap,
        zip_longest=_it.zip_longest,
        repeat=_it.repeat,
        cycle=_it.cycle,
        count=_it.count,
        pairwise=getattr(_it, "pairwise", None),  # 3.10+
    )

    # textwrap — string formatting utilities
    g["textwrap"] = SimpleNamespace(
        wrap=_tw.wrap,
        fill=_tw.fill,
        shorten=_tw.shorten,
        indent=_tw.indent,
        dedent=_tw.dedent,
    )

    # string — constants only (no Template — its safe_substitute can be probed)
    g["string"] = SimpleNamespace(
        ascii_letters=_st.ascii_letters,
        ascii_lowercase=_st.ascii_lowercase,
        ascii_uppercase=_st.ascii_uppercase,
        digits=_st.digits,
        hexdigits=_st.hexdigits,
        octdigits=_st.octdigits,
        punctuation=_st.punctuation,
        printable=_st.printable,
        whitespace=_st.whitespace,
    )

    return g

# Names that must never resolve inside the sandbox (escape hatches).
_BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "import",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "hasattr", "input", "exit", "quit", "help", "breakpoint", "memoryview",
    "super", "object", "type", "classmethod", "staticmethod", "property",
    "__builtins__", "__loader__", "__spec__", "__class__", "compile",
}

_BLOCKED_ATTRS = {
    # str.format/format_map interpret field expressions themselves, so
    # "{0.__closure__[0].cell_contents}".format(call) bypasses AST checks.
    "format", "format_map",
}


class _ScriptValidator(ast.NodeVisitor):
    """Reject any node, name, or attribute access that could escape the sandbox."""

    def generic_visit(self, node):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_") or node.attr in _BLOCKED_ATTRS:
            raise ValueError(f"access to '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in _BLOCKED_NAMES or node.id.startswith("__"):
            raise ValueError(f"use of '{node.id}' is not allowed")
        # Name has no children worth visiting; skip generic node-type check.

    def visit_arg(self, node):
        if node.arg in _BLOCKED_NAMES or node.arg.startswith("__"):
            raise ValueError(f"argument '{node.arg}' is not allowed")
        self.generic_visit(node)

    def visit_Import(self, node):
        raise ValueError(
            "imports are not allowed; these modules are already available as "
            f"variables: {', '.join(_SAFE_MODULE_NAMES)}"
        )

    def visit_ImportFrom(self, node):
        raise ValueError(
            "imports are not allowed; these modules are already available as "
            f"variables: {', '.join(_SAFE_MODULE_NAMES)}"
        )


def _unwrap_mcp(content):
    """Flatten the MCP content envelope so callers get usable data.

    MCP tools return a list of content blocks like
    ``[{"type": "text", "text": "..."}]``. When every block is text, return the
    text (JSON-parsed if it parses), so scripts don't need boilerplate to dig
    through the envelope. Non-text / mixed content is returned unchanged.
    """
    if isinstance(content, list) and content:
        texts = [
            b.get("text")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text") is not None
        ]
        if texts and len(texts) == len(content):
            joined = texts[0] if len(texts) == 1 else "\n".join(texts)
            try:
                return json.loads(joined)
            except (ValueError, TypeError):
                return joined
    return content


_VALUE_KEYS = ("string_value", "int_value", "long_value", "double_value", "boolean_value", "null")


def _scalar(v):
    """Reduce a typed value wrapper (e.g. {'string_value': 'x'}) to its scalar."""
    if isinstance(v, dict):
        for k in _VALUE_KEYS:
            if k in v:
                return v[k]
        if len(v) == 1:
            return next(iter(v.values()))
    return v


def rows(result):
    """Normalize a tool result into a list of rows (each a list of scalars).

    Handles the common tabular shapes returned by SQL-style MCP tools:
      • Databricks Statement Execution — ``result.data_array`` (or top-level
        ``data_array``) with ``{"values": [{"string_value": ...}, ...]}`` rows
      • a generic ``{"rows": [[...], ...]}`` mapping
      • a bare list of rows / scalars
    Returns ``[]`` when no tabular data is found. Exposed inside run_tool_script.
    """
    if isinstance(result, dict):
        data_array = result.get("result", {}).get("data_array")
        if data_array is None:
            data_array = result.get("data_array")
        if isinstance(data_array, list):
            out = []
            for row in data_array:
                if isinstance(row, dict) and "values" in row:
                    out.append([_scalar(x) for x in row.get("values", [])])
                elif isinstance(row, list):
                    out.append([_scalar(x) for x in row])
                else:
                    out.append([_scalar(row)])
            return out
        if isinstance(result.get("rows"), list):
            return result["rows"]
    if isinstance(result, list):
        return [r if isinstance(r, list) else [r] for r in result]
    return []


def _safe_builtins() -> dict:
    names = [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "int", "isinstance", "issubclass",
        "len", "list", "map", "max", "min", "range", "repr", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip", "bytes",
    ]
    import builtins as _b

    safe = {n: getattr(_b, n) for n in names if hasattr(_b, n)}
    # A curated set of exception classes so user code can try/except.
    for exc in ("Exception", "ValueError", "KeyError", "TypeError", "IndexError",
                "ZeroDivisionError", "StopIteration", "RuntimeError",
                "ArithmeticError", "AttributeError", "LookupError"):
        safe[exc] = getattr(_b, exc)
    safe["True"], safe["False"], safe["None"] = True, False, None
    return safe


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


class Tools:
    class Valves(BaseModel):
        # ── Search ──
        search_top_k: int = Field(default=5, description="Results returned per tool search.")
        index_ttl_seconds: int = Field(default=300, description="Seconds before the tool index is rebuilt.")
        mcp_discovery_timeout: float = Field(default=10.0, description="Max seconds to enumerate each MCP server during indexing.")
        embedding_timeout: float = Field(default=20.0, description="Max seconds for embedding generation before falling back to keyword-only.")
        semantic_weight: float = Field(default=0.6, description="Hybrid balance: 1.0 = pure semantic, 0.0 = pure keyword.")

        # ── Output limits (single call_tool path only) ──
        default_max_output_tokens: int = Field(default=8000, description="Default max tokens for a single call_tool response (0 = unlimited).")
        output_truncation_strategy: str = Field(
            default="head_tail",
            json_schema_extra={"input": {"type": "select", "options": [
                {"value": "head_tail", "label": "Head + Tail (keeps beginning and end)"},
                {"value": "truncate", "label": "Truncate (keeps beginning only)"},
            ]}},
            description="How to trim oversized single tool responses.",
        )
        server_token_limits: str = Field(default="", description="Per-server overrides, one per line: server_id=limit")

        # ── Programmatic Tool Calling ──
        enable_code_mode: bool = Field(
            default=False,
            description=(
                "Enable run_tool_script (programmatic tool calling). This EXECUTES "
                "model-written Python in a restricted in-process sandbox. Off by "
                "default — turn on only in environments you trust."
            ),
        )
        code_timeout: float = Field(default=60.0, description="Max wall-clock seconds for an orchestration script (also caps CPU loops).")
        code_max_calls: int = Field(default=50, description="Max tool calls a single script may make (0 = unlimited).")
        code_max_output_chars: int = Field(default=24000, description="Max characters of script output returned to the model.")

        # ── Debug ──
        debug_events: bool = Field(default=False, description="Show step-by-step status events in chat (debugging).")

    def __init__(self):
        self.valves = self.Valves()
        self._index = _ToolIndex()
        self._index_lock = asyncio.Lock()

    # -- small helpers --

    async def _emit(self, emitter, description, done=False):
        if emitter and self.valves.debug_events:
            await emitter({"type": "status", "data": {"description": description, "done": done}})

    @staticmethod
    def _user_model(__user__):
        from open_webui.models.users import UserModel
        if isinstance(__user__, UserModel):
            return __user__
        return UserModel(**(__user__ or {}))

    @staticmethod
    def _content_prefix():
        try:
            from open_webui.config import RAG_EMBEDDING_CONTENT_PREFIX
            return RAG_EMBEDDING_CONTENT_PREFIX or None
        except Exception:
            return None

    @staticmethod
    def _query_prefix():
        try:
            from open_webui.config import RAG_EMBEDDING_QUERY_PREFIX
            return RAG_EMBEDDING_QUERY_PREFIX or None
        except Exception:
            return None

    async def _embed_query(self, request, query) -> Optional[list[float]]:
        try:
            embed_fn = request.app.state.EMBEDDING_FUNCTION
            if not embed_fn:
                return None
            qe = await embed_fn(query, prefix=self._query_prefix())
            if isinstance(qe, list) and qe:
                return qe[0] if isinstance(qe[0], list) else qe
        except Exception:
            return None
        return None

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0


    # -- indexing --

    async def _ensure_index(self, request, user_model, force=False, __self_id__="tool_orchestrator", emitter=None):
        if not force and not self._index.is_stale(self.valves.index_ttl_seconds):
            return
        async with self._index_lock:
            if not force and not self._index.is_stale(self.valves.index_ttl_seconds):
                return

            from open_webui.utils.tools import get_tool_servers, is_string_allowed

            entries: list[_ToolEntry] = []
            connections = request.app.state.config.TOOL_SERVER_CONNECTIONS

            def _flist(cfg):
                if isinstance(cfg, str):
                    return [f.strip() for f in cfg.split(",") if f.strip()]
                return cfg or []

            # OpenAPI
            await self._emit(emitter, "🧭 Indexing: OpenAPI servers...")
            try:
                servers = await get_tool_servers(request)
            except Exception as e:
                log.warning(f"Tool Orchestrator: get_tool_servers failed: {e}")
                servers = []
            for server in servers:
                idx = server.get("idx", 0)
                if idx >= len(connections):
                    continue
                conn = connections[idx]
                sid = server.get("id", "unknown")
                sname = server.get("openapi", {}).get("info", {}).get("title", sid)
                fl = _flist(conn.get("config", {}).get("function_name_filter_list", ""))
                for spec in server.get("specs", []):
                    fn = spec.get("name", "")
                    if fl and not is_string_allowed(fn, fl):
                        continue
                    entries.append(_ToolEntry(fn, spec.get("description", ""), spec.get("parameters", {}),
                                              sid, sname, "openapi", f"server:{sid}", conn))

            # MCP (per-server budget; abandon on timeout, never await hung cleanup)
            mcp_conns = [c for c in connections if c.get("type", "") == "mcp" and c.get("config", {}).get("enable")]

            async def _discover(conn):
                info = conn.get("info", {})
                sid = info.get("id", "unknown")
                sname = info.get("name", sid)
                out, client = [], None
                try:
                    from open_webui.utils.mcp.client import MCPClient
                    from open_webui.utils.tools import build_tool_server_headers
                    headers, _ = await build_tool_server_headers(conn, request, user_model, server_id=sid, metadata={}, extra_params={})
                    client = MCPClient()
                    await client.connect(url=conn.get("url", ""), headers=headers or None)
                    specs = await client.list_tool_specs()
                    fl = _flist(conn.get("config", {}).get("function_name_filter_list", ""))
                    for spec in (specs or []):
                        fn = spec.get("name", "")
                        if fl and not is_string_allowed(fn, fl):
                            continue
                        out.append(_ToolEntry(fn, spec.get("description", ""), spec.get("parameters", {}),
                                              sid, sname, "mcp", f"server:mcp:{sid}", conn))
                except Exception as e:
                    log.warning(f"Tool Orchestrator: MCP '{sid}' index failed: {e}")
                finally:
                    if client is not None:
                        try:
                            await client.disconnect()
                        except Exception:
                            pass
                return out

            for conn in mcp_conns:
                sid = conn.get("info", {}).get("id", "unknown")
                await self._emit(emitter, f"🧭 Indexing: MCP server '{sid}'...")
                task = asyncio.ensure_future(_discover(conn))
                done, _pending = await asyncio.wait({task}, timeout=self.valves.mcp_discovery_timeout)
                if task in done:
                    try:
                        entries.extend(task.result())
                    except Exception as e:
                        log.warning(f"Tool Orchestrator: MCP '{sid}' error: {e}")
                else:
                    task.cancel()
                    await self._emit(emitter, f"⏱️ MCP '{sid}' timed out; skipping")

            # Local tools
            await self._emit(emitter, "🧭 Indexing: local tools...")
            try:
                from open_webui.models.tools import Tools as ToolsDB
                for tool in await ToolsDB.get_tools(defer_content=True):
                    if tool.id == __self_id__:
                        continue
                    for spec in (tool.specs or []):
                        entries.append(_ToolEntry(spec.get("name", ""), spec.get("description", ""),
                                                  spec.get("parameters", {}), f"local:{tool.id}",
                                                  tool.name or tool.id, "local", tool.id, None))
            except Exception as e:
                log.warning(f"Tool Orchestrator: local tools index failed: {e}")

            new_index = _ToolIndex()
            new_index.entries = entries
            new_index.build_bm25()
            new_index.built_at = time.time()

            if entries:
                await self._emit(emitter, f"🧭 Indexing: embedding {len(entries)} functions...")
                try:
                    embed_fn = request.app.state.EMBEDDING_FUNCTION
                    if embed_fn:
                        vecs = await asyncio.wait_for(
                            embed_fn([e.search_text for e in entries], prefix=self._content_prefix()),
                            timeout=self.valves.embedding_timeout,
                        )
                        if isinstance(vecs, list) and len(vecs) == len(entries):
                            for i, e in enumerate(entries):
                                e.embedding = vecs[i]
                except asyncio.TimeoutError:
                    await self._emit(emitter, "⏱️ Embedding timed out; keyword-only")
                except Exception as e:
                    log.warning(f"Tool Orchestrator: embedding failed (BM25 only): {e}")

            self._index = new_index

    async def _accessible_entries(self, request, user_model) -> list[_ToolEntry]:
        from open_webui.utils.access_control import has_connection_access
        from open_webui.models.groups import Groups

        idx = self._index
        user_group_ids = {g.id for g in await Groups.get_groups_by_member_id(user_model.id)}
        conn_cache: dict[int, bool] = {}

        async def _ok(conn):
            k = id(conn)
            if k not in conn_cache:
                conn_cache[k] = await has_connection_access(user_model, conn, user_group_ids)
            return conn_cache[k]

        accessible_local: Optional[set] = None
        try:
            from open_webui.models.tools import Tools as ToolsDB
            from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL
            if not (user_model.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL):
                tools = await ToolsDB.get_tools_by_user_id(user_model.id, "read", defer_content=True)
                accessible_local = {t.id for t in tools}
        except Exception as e:
            log.warning(f"Tool Orchestrator: local access check failed: {e}")
            accessible_local = set()

        allowed = []
        for e in idx.entries:
            if e.server_type == "local":
                if accessible_local is None or e.tool_id in accessible_local:
                    allowed.append(e)
            elif e.connection is not None and await _ok(e.connection):
                allowed.append(e)
        return allowed

    def _hybrid_search(self, query, query_embedding, allowed, top_k):
        idx = self._index
        if not idx.entries or not allowed:
            return []
        allowed_ids = {id(e) for e in allowed}
        bm25 = idx.bm25_score(query)
        max_bm25 = max(bm25) if bm25 else 0.0
        if max_bm25 <= 0:
            max_bm25 = 1.0
        use_sem = query_embedding is not None and all(e.embedding is not None for e in idx.entries)
        w_sem = max(0.0, min(1.0, self.valves.semantic_weight)) if use_sem else 0.0
        w_bm = 1.0 - w_sem
        scored = []
        for i, e in enumerate(idx.entries):
            if id(e) not in allowed_ids:
                continue
            s = w_bm * (bm25[i] / max_bm25)
            if use_sem:
                s += w_sem * self._cosine(query_embedding, e.embedding)
            scored.append((s, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


    # -- single tool invocation (shared by call_tool and the sandbox) --

    async def _invoke(self, request, user_model, server_id, function_name, args, metadata):
        """Resolve and execute one tool call with a per-user access check.
        Returns the raw result; raises PermissionError / ValueError on problems."""
        from open_webui.utils.access_control import has_connection_access

        connections = request.app.state.config.TOOL_SERVER_CONNECTIONS
        extra_params: dict = {}
        try:
            from open_webui.utils.middleware import get_system_oauth_token
            extra_params["__oauth_token__"] = await get_system_oauth_token(request, user_model)
        except Exception:
            pass

        from open_webui.utils.tools import is_string_allowed

        def _blocked_by_filter(conn_cfg) -> bool:
            flt = (conn_cfg or {}).get("function_name_filter_list", "")
            if isinstance(flt, str):
                flt = [f.strip() for f in flt.split(",") if f.strip()]
            flt = flt or []
            return bool(flt) and not is_string_allowed(function_name, flt)

        is_local = server_id == "local" or server_id.startswith("local:")

        if not is_local:
            mcp_conn = next((c for c in connections
                             if c.get("type", "") == "mcp" and c.get("info", {}).get("id") == server_id), None)
            if mcp_conn:
                if not await has_connection_access(user_model, mcp_conn):
                    raise PermissionError(f"Access denied to server '{server_id}'")
                # Enforce the admin's function filter at call time (not just indexing).
                if _blocked_by_filter(mcp_conn.get("config", {})):
                    raise PermissionError(f"Function '{function_name}' is not exposed by server '{server_id}'")
                from open_webui.utils.mcp.client import MCPClient
                from open_webui.utils.tools import build_tool_server_headers
                headers, _ = await build_tool_server_headers(
                    mcp_conn, request, user_model, server_id=server_id,
                    metadata=metadata or {}, extra_params=extra_params)
                client = MCPClient()
                await client.connect(url=mcp_conn.get("url", ""), headers=headers or None)
                try:
                    return _unwrap_mcp(await client.call_tool(function_name, args))
                finally:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

            from open_webui.utils.tools import get_tool_servers, execute_tool_server, build_tool_server_headers
            target = next((s for s in await get_tool_servers(request) if s.get("id") == server_id), None)
            if not target:
                raise ValueError(f"Server '{server_id}' not found")
            idx = target.get("idx", 0)
            if idx >= len(connections):
                raise ValueError(f"Server index out of range for '{server_id}'")
            conn = connections[idx]
            if not await has_connection_access(user_model, conn):
                raise PermissionError(f"Access denied to server '{server_id}'")
            if _blocked_by_filter(conn.get("config", {})):
                raise PermissionError(f"Function '{function_name}' is not exposed by server '{server_id}'")
            headers, cookies = await build_tool_server_headers(
                conn, request, user_model, server_id=server_id,
                metadata=metadata or {}, extra_params=extra_params)
            headers.setdefault("Content-Type", "application/json")
            result, _ = await execute_tool_server(
                url=target["url"], headers=headers, cookies=cookies,
                name=function_name, params=args, server_data=target)
            return result

        # local tool — server_id is "local:<tool_id>" (unambiguous) or bare "local"
        from open_webui.models.tools import Tools as ToolsDB
        from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL
        target_tool_id = server_id.split(":", 1)[1] if ":" in server_id else None
        if user_model.role == "admin" and BYPASS_ADMIN_ACCESS_CONTROL:
            local_tools = await ToolsDB.get_tools(defer_content=False)
        else:
            local_tools = await ToolsDB.get_tools_by_user_id(user_model.id, "read", defer_content=False)

        target_tool = None
        if target_tool_id:
            target_tool = next((t for t in local_tools if t.id == target_tool_id), None)
            if target_tool and not any(s.get("name") == function_name for s in (target_tool.specs or [])):
                raise ValueError(f"Function '{function_name}' not found on tool '{target_tool_id}'")
        else:
            # legacy bare "local": fall back to first accessible tool with the function
            target_tool = next(
                (t for t in local_tools if any(s.get("name") == function_name for s in (t.specs or []))),
                None,
            )
        if not target_tool:
            raise PermissionError(f"Local function '{function_name}' not found or access denied")
        from open_webui.utils.tools import load_tool_module_by_id
        module, _ = await load_tool_module_by_id(target_tool.id, content=target_tool.content)
        if hasattr(module, "valves") and hasattr(module, "Valves"):
            module.valves = module.Valves(**(await ToolsDB.get_tool_valves_by_id(target_tool.id) or {}))
        fn = getattr(module, function_name, None)
        if fn is None:
            raise ValueError(f"Function '{function_name}' missing on tool '{target_tool.id}'")
        import inspect
        sig = inspect.signature(fn)
        kwargs = dict(args)
        if "__user__" in sig.parameters:
            kwargs["__user__"] = user_model.model_dump() if hasattr(user_model, "model_dump") else user_model
        if "__request__" in sig.parameters:
            kwargs["__request__"] = request
        if asyncio.iscoroutinefunction(fn):
            return await fn(**kwargs)
        return fn(**kwargs)

    def _apply_output_policy(self, server_id, content):
        limits = {}
        for line in self.valves.server_token_limits.splitlines():
            line = line.strip()
            if "=" in line:
                sid, _, raw = line.partition("=")
                try:
                    limits[sid.strip()] = int(raw.strip())
                except ValueError:
                    pass
        max_tokens = limits.get(server_id, self.valves.default_max_output_tokens)
        text = json.dumps(content, indent=2, default=str) if isinstance(content, (dict, list)) else str(content)
        original = len(text)
        if max_tokens <= 0:
            return text, original, original
        max_chars = max_tokens * 4
        if original <= max_chars:
            return text, original, original
        omitted = (original - max_chars) // 4
        if self.valves.output_truncation_strategy == "truncate":
            return text[:max_chars] + "\n\n[... output truncated — exceeded limit ...]", original, max_chars
        head = int(max_chars * 0.7)
        tail = max_chars - head
        out = text[:head] + f"\n\n[... ~{omitted} tokens omitted by output policy ...]\n\n" + text[-tail:]
        return out, original, len(out)

    # -- public: discovery --

    async def search_tools(self, query: str, limit: int = None, __request__=None, __user__=None,
                           __event_emitter__=None, __id__="tool_orchestrator") -> str:
        """
        Find tool functions by describing the capability you need. Returns matches
        with their server_id, function_name, description, and JSON parameter schema.
        Call this before call_tool or run_tool_script to discover names/arguments.
        To see EVERY available tool at once, call list_servers instead.

        :param query: Natural-language capability, e.g. "create a Jira ticket" or "query the warehouse".
        :param limit: Optional max number of results to return (defaults to the configured top-K).
        :return: JSON array of matching tools.
        """
        await self._emit(__event_emitter__, f"🔍 Searching tools for: \"{query}\"")
        try:
            user_model = self._user_model(__user__)
            await self._ensure_index(__request__, user_model, __self_id__=__id__, emitter=__event_emitter__)
            allowed = await self._accessible_entries(__request__, user_model)
            qe = await self._embed_query(__request__, query)
            top_k = limit if isinstance(limit, int) and limit > 0 else self.valves.search_top_k
            results = self._hybrid_search(query, qe, allowed, top_k)
            out = [{
                "function_name": e.name, "description": e.description,
                "server_id": e.server_id, "server_name": e.server_name,
                "server_type": e.server_type, "parameters": e.parameters,
            } for e in results]
            if out:
                summary = ", ".join(f"{r['function_name']} ({r['server_name']})" for r in out[:3])
                await self._emit(__event_emitter__, f"✅ Found {len(out)} tool(s): {summary}", done=True)
            else:
                await self._emit(__event_emitter__, "⚠️ No matching tools found.", done=True)
            return json.dumps(out, indent=2)
        except Exception as e:
            msg = f"Error searching tools: {e}"
            log.exception(msg)
            await self._emit(__event_emitter__, f"❌ {msg}", done=True)
            return json.dumps({"error": msg})

    async def call_tool(self, server_id: str, function_name: str, arguments: str,
                        __request__=None, __user__=None, __event_emitter__=None,
                        __metadata__=None, __id__="tool_orchestrator") -> str:
        """
        Execute a single tool function. Use search_tools first to get server_id,
        function_name and the parameter schema. For multi-step or bulk work, prefer
        run_tool_script instead so intermediate results stay out of context.

        :param server_id: Server identifier from search_tools.
        :param function_name: Exact function name from search_tools.
        :param arguments: JSON object string of arguments.
        :return: The tool's response (may be trimmed by the per-server output limit).
        """
        await self._emit(__event_emitter__, f"🔌 Calling {function_name} on {server_id}...")
        try:
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError as e:
                return json.dumps({"error": f"Invalid JSON arguments: {e}"})
            if not isinstance(args, dict):
                return json.dumps({"error": "arguments must be a JSON object"})
            user_model = self._user_model(__user__)
            result = await self._invoke(__request__, user_model, server_id, function_name, args, __metadata__)
            output, original, final = self._apply_output_policy(server_id, result)
            if final < original:
                pct = round(final / original * 100) if original else 100
                await self._emit(__event_emitter__, f"✅ {function_name} done (trimmed to ~{pct}%)", done=True)
            else:
                await self._emit(__event_emitter__, f"✅ {function_name} done", done=True)
            return output
        except Exception as e:
            msg = f"Error calling {function_name}: {e}"
            log.exception(msg)
            await self._emit(__event_emitter__, f"❌ {msg}", done=True)
            return json.dumps({"error": msg})

    async def list_servers(self, __request__=None, __user__=None,
                           __event_emitter__=None, __id__="tool_orchestrator") -> str:
        """
        List every tool server you can access AND every function on each one
        (name + short description). This is the complete catalogue — use it to
        answer "what tools do you have?" in a single call instead of repeatedly
        searching. Use search_tools when you need full parameter schemas.

        :return: JSON array of servers, each with its full list of functions.
        """
        await self._emit(__event_emitter__, "📡 Discovering accessible servers...")
        try:
            user_model = self._user_model(__user__)
            await self._ensure_index(__request__, user_model, __self_id__=__id__, emitter=__event_emitter__)
            allowed = await self._accessible_entries(__request__, user_model)
            servers: dict[str, dict] = {}
            for e in allowed:
                s = servers.setdefault(e.server_id, {
                    "server_id": e.server_id, "server_name": e.server_name,
                    "server_type": e.server_type, "function_count": 0, "functions": [],
                })
                s["function_count"] += 1
                desc = (e.description or "").strip().split("\n")[0][:120]
                s["functions"].append({"function_name": e.name, "description": desc})
            out = list(servers.values())
            total = sum(s["function_count"] for s in out)
            await self._emit(__event_emitter__, f"✅ {len(out)} server(s), {total} function(s) accessible", done=True)
            return json.dumps(out, indent=2)
        except Exception as e:
            msg = f"Error listing servers: {e}"
            log.exception(msg)
            await self._emit(__event_emitter__, f"❌ {msg}", done=True)
            return json.dumps({"error": msg})

    async def refresh_index(self, __request__=None, __user__=None,
                            __event_emitter__=None, __id__="tool_orchestrator") -> str:
        """
        Force a rebuild of the tool index. Use after adding new tools/servers.

        :return: JSON summary of what was indexed (for the calling user).
        """
        await self._emit(__event_emitter__, "🔄 Rebuilding tool index...")
        try:
            user_model = self._user_model(__user__)
            await self._ensure_index(__request__, user_model, force=True, __self_id__=__id__, emitter=__event_emitter__)
            allowed = await self._accessible_entries(__request__, user_model)
            counts: dict[str, int] = {}
            for e in allowed:
                counts[e.server_name] = counts.get(e.server_name, 0) + 1
            await self._emit(__event_emitter__, f"✅ Indexed {len(allowed)} accessible functions", done=True)
            return json.dumps({"status": "ok", "functions_indexed": len(allowed),
                               "servers": len(counts), "breakdown": counts})
        except Exception as e:
            msg = f"Error rebuilding index: {e}"
            log.exception(msg)
            await self._emit(__event_emitter__, f"❌ {msg}", done=True)
            return json.dumps({"error": msg})


    # -- public: programmatic tool calling --

    @staticmethod
    def _compile_script(code: str):
        """Validate against the allow-list and wrap into an async function. Raises ValueError if rejected."""
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"syntax error: {e}")
        for node in tree.body:
            _ScriptValidator().visit(node)
        func = ast.AsyncFunctionDef(
            name="__ptc_main__",
            args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                               kw_defaults=[], kwarg=None, defaults=[]),
            body=tree.body or [ast.Pass()],
            decorator_list=[], returns=None, type_comment=None,
        )
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        return compile(module, "<tool_script>", "exec")

    async def run_tool_script(self, code: str, __request__=None, __user__=None,
                              __event_emitter__=None, __metadata__=None,
                              __id__="tool_orchestrator") -> str:
        """
        PROGRAMMATIC TOOL CALLING. Write an async Python script that orchestrates
        many tools at once. Loops, parallel calls, filtering and aggregation run in
        a sandbox; ONLY what you print() is returned to you. Use this instead of
        many call_tool round-trips when you must combine 3+ calls, process large
        results, or run calls in parallel — it keeps intermediate data out of context.

        Available inside the script:
          • await call(server_id, function_name, args_dict)   -> tool result (parsed)
          • await search(query, top_k=5)                       -> list of matching tools
          • await gather(*coros)                               -> run calls in parallel
          • rows(result)  -> normalize a SQL/tabular tool result into a list of
                             rows (list of scalars). Handles Databricks
                             (result.data_array), {"rows": [...]}, and list shapes.
          • print(...) to return results to yourself
          • json, plus safe builtins (len, sum, sorted, zip, enumerate, range, ...)
          • these modules are pre-loaded as sanitised facades (no import needed/allowed):
            json (dumps/loads), math, re, datetime, collections, itertools,
            textwrap, string
        Notes: MCP results are auto-unwrapped (text/JSON comes back parsed). Use
        `return` to stop early — `exit()` is not available. Use f-strings or
        concatenation for formatting; `str.format()` / `format_map()` are blocked
        because they can traverse hidden attributes outside the AST validator.
        Discover server_id/function_name via search_tools first.

        Example — top 5 schemas by table count in the first catalog (one script):
            cats = await call("databricks", "execute_sql_read_only", {"query": "SHOW CATALOGS"})
            catalogs = [r[0] for r in rows(cats)]
            first = catalogs[0]
            sch = await call("databricks", "execute_sql_read_only",
                             {"query": f"SHOW SCHEMAS IN {first}"})
            names = [r[0] for r in rows(sch)]
            counts = await gather(*[
                call("databricks", "execute_sql_read_only",
                     {"query": f"SELECT count(*) FROM {first}.information_schema.tables "
                               f"WHERE table_schema = '{s}'"})
                for s in names
            ])
            ranked = sorted(
                ((s, int(rows(c)[0][0])) for s, c in zip(names, counts)),
                key=lambda x: x[1], reverse=True,
            )
            print(ranked[:5])

        :param code: The Python orchestration script. Print final results.
        :return: The script's printed output (capped), or an error with partial output.
        """
        if not self.valves.enable_code_mode:
            return json.dumps({"error": "Programmatic tool calling is disabled (enable_code_mode valve)."})

        await self._emit(__event_emitter__, "🧩 Preparing orchestration sandbox...")
        try:
            user_model = self._user_model(__user__)
            await self._ensure_index(__request__, user_model, __self_id__=__id__, emitter=__event_emitter__)

            try:
                compiled = self._compile_script(code)
            except ValueError as e:
                return json.dumps({"error": f"Script rejected: {e}"})

            main_loop = asyncio.get_running_loop()
            buffer: list[str] = []
            call_count = {"n": 0}
            max_calls = self.valves.code_max_calls

            def _print(*args, sep=" ", end="\n", **_kw):
                buffer.append(sep.join(str(a) for a in args) + end)

            # call()/search() run their actual I/O on the MAIN event loop (the loop
            # the DB engine, MCP clients and aiohttp sessions were created on),
            # bridged from the worker thread. This avoids "future attached to a
            # different loop" errors while keeping orchestration off the main loop.
            async def _call(server_id, function_name, args=None, **kwargs):
                merged = dict(args or {})
                merged.update(kwargs)
                call_count["n"] += 1
                if max_calls and call_count["n"] > max_calls:
                    raise RuntimeError(
                        f"tool-call limit ({max_calls}) exceeded; raise code_max_calls if intended"
                    )
                cf = asyncio.run_coroutine_threadsafe(
                    self._invoke(__request__, user_model, server_id, function_name, merged, __metadata__),
                    main_loop,
                )
                return await asyncio.wrap_future(cf)

            async def _search(query, top_k=None):
                cf = asyncio.run_coroutine_threadsafe(
                    self._accessible_entries(__request__, user_model), main_loop
                )
                allowed = await asyncio.wrap_future(cf)
                cfq = asyncio.run_coroutine_threadsafe(self._embed_query(__request__, query), main_loop)
                qe = await asyncio.wrap_future(cfq)
                res = self._hybrid_search(query, qe, allowed, top_k or self.valves.search_top_k)
                return [{"server_id": e.server_id, "function_name": e.name,
                         "description": e.description, "parameters": e.parameters} for e in res]

            sandbox = {
                "__builtins__": _safe_builtins(),
                "call": _call,
                "search": _search,
                "gather": asyncio.gather,
                "rows": rows,
                "print": _print,
            }
            # Bind safe module facades (SimpleNamespaces — never real module
            # objects). update() so the json facade replaces any base binding;
            # the facade names (json, math, re, ...) don't collide with the
            # helpers above.
            sandbox.update(_build_sandbox_globals())
            exec(compiled, sandbox)

            await self._emit(__event_emitter__, "🧩 Running orchestration script...")

            # Run the script in a dedicated daemon thread with its OWN event loop.
            # A wall-clock tracer interrupts CPU-bound loops (e.g. `while True`),
            # and the main loop is never blocked — so a misbehaving script cannot
            # take down the server; other requests keep being served.
            done: concurrent.futures.Future = concurrent.futures.Future()
            timeout = self.valves.code_timeout

            def _worker():
                worker_loop = asyncio.new_event_loop()
                deadline = time.monotonic() + timeout

                def _tracer(frame, event, arg):
                    if time.monotonic() > deadline:
                        raise TimeoutError("script exceeded code_timeout")
                    return _tracer

                try:
                    asyncio.set_event_loop(worker_loop)
                    sys.settrace(_tracer)
                    worker_loop.run_until_complete(sandbox["__ptc_main__"]())
                    done.set_result(True)
                except BaseException as exc:  # propagate everything to the caller
                    done.set_exception(exc)
                finally:
                    sys.settrace(None)
                    try:
                        worker_loop.close()
                    except Exception:
                        pass

            threading.Thread(target=_worker, name="tool-orchestrator-ptc", daemon=True).start()

            try:
                # Grace beyond the CPU deadline so the tracer fires first; if the
                # thread is wedged in a C call we still free the main loop here.
                await asyncio.wait_for(asyncio.wrap_future(done), timeout=timeout + 5)
            except (asyncio.TimeoutError, TimeoutError):
                partial = "".join(buffer)[: self.valves.code_max_output_chars]
                await self._emit(__event_emitter__, "⏱️ Script timed out", done=True)
                return json.dumps({"error": "Script exceeded time limit", "partial_output": partial})
            except PermissionError as e:
                return json.dumps({"error": f"Access denied during script: {e}",
                                   "output": "".join(buffer)[:2000]})
            except Exception as e:
                partial = "".join(buffer)[: self.valves.code_max_output_chars]
                await self._emit(__event_emitter__, f"❌ Script error: {e}", done=True)
                return json.dumps({"error": f"{type(e).__name__}: {e}", "partial_output": partial})

            output = "".join(buffer)
            await self._emit(__event_emitter__, f"✅ Script finished ({call_count['n']} tool call(s))", done=True)
            if not output.strip():
                return "(script finished with no output — use print() to return results)"
            if len(output) > self.valves.code_max_output_chars:
                output = output[: self.valves.code_max_output_chars] + "\n[... output truncated by code_max_output_chars ...]"
            return output

        except Exception as e:
            msg = f"Error running script: {e}"
            log.exception(msg)
            await self._emit(__event_emitter__, f"❌ {msg}", done=True)
            return json.dumps({"error": msg})
