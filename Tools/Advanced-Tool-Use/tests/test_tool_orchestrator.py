import asyncio
import importlib.util
import json
import pathlib
import types
import unittest


TOOL_PATH = pathlib.Path(__file__).resolve().parents[1] / "tool_orchestrator.py"
SPEC = importlib.util.spec_from_file_location("tool_orchestrator", TOOL_PATH)
tool_orchestrator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool_orchestrator)


class BaseStubTools(tool_orchestrator.Tools):
    @staticmethod
    def _user_model(__user__):
        return __user__ or types.SimpleNamespace(id="user-1", role="user")

    async def _ensure_index(self, *args, **kwargs):
        return None


class ToolOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def test_rows_normalizes_common_shapes(self):
        databricks = {
            "result": {
                "data_array": [
                    {"values": [{"string_value": "a"}, {"long_value": 2}]},
                    {"values": [{"double_value": 3.5}, {"boolean_value": True}]},
                ]
            }
        }
        self.assertEqual(tool_orchestrator.rows(databricks), [["a", 2], [3.5, True]])
        self.assertEqual(tool_orchestrator.rows({"rows": [[1, 2]]}), [[1, 2]])
        self.assertEqual(tool_orchestrator.rows(["x", "y"]), [["x"], ["y"]])

    def test_script_validator_rejects_escape_shapes(self):
        with self.assertRaisesRegex(ValueError, "imports are not allowed"):
            tool_orchestrator.Tools._compile_script("import os")
        with self.assertRaisesRegex(ValueError, "access to '__class__' is not allowed"):
            tool_orchestrator.Tools._compile_script("x = ''.__class__")

    async def test_embed_query_timeout_falls_back_to_keyword_only(self):
        tools = BaseStubTools()
        tools.valves.embedding_timeout = 0.01

        async def slow_embedding(*_args, **_kwargs):
            await asyncio.sleep(1)
            return [[1.0]]

        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(EMBEDDING_FUNCTION=slow_embedding)
            )
        )

        self.assertIsNone(await tools._embed_query(request, "warehouse query"))

    async def test_call_tool_timeout_cancels_standalone_invocation(self):
        class SlowTools(BaseStubTools):
            def __init__(self):
                super().__init__()
                self.cancelled = False

            async def _invoke(self, *_args, **_kwargs):
                try:
                    await asyncio.sleep(1)
                finally:
                    self.cancelled = True

        tools = SlowTools()
        tools.valves.tool_call_timeout = 0.01

        result = json.loads(await tools.call_tool("srv", "slow", "{}"))

        self.assertIn("exceeded tool_call_timeout", result["error"])
        self.assertEqual(result["timeout_seconds"], 0.01)
        self.assertTrue(tools.cancelled)

    async def test_run_tool_script_limits_parallel_tool_calls(self):
        class ParallelTools(BaseStubTools):
            def __init__(self):
                super().__init__()
                self.active = 0
                self.max_active = 0

            async def _invoke(self, _request, _user_model, _server_id, _function_name, args, _metadata):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    await asyncio.sleep(0.02)
                    return {"i": args["i"]}
                finally:
                    self.active -= 1

        tools = ParallelTools()
        tools.valves.enable_code_mode = True
        tools.valves.code_max_parallel_calls = 2
        tools.valves.code_timeout = 2

        output = await tools.run_tool_script(
            """
results = await gather(*[
    call("srv", "fn", {"i": i})
    for i in range(6)
])
print(json.dumps(results))
"""
        )

        self.assertEqual([row["i"] for row in json.loads(output)], list(range(6)))
        self.assertLessEqual(tools.max_active, 2)

    async def test_run_tool_script_reports_timeout_diagnostics(self):
        class SlowScriptTools(BaseStubTools):
            async def _invoke(self, *_args, **_kwargs):
                await asyncio.sleep(1)
                return {"ok": True}

        tools = SlowScriptTools()
        tools.valves.enable_code_mode = True
        tools.valves.code_tool_call_timeout = 0.01
        tools.valves.code_timeout = 2

        result = json.loads(await tools.run_tool_script('await call("srv", "slow", {})'))

        self.assertIn("code_tool_call_timeout", result["error"])
        self.assertEqual(result["diagnostics"]["tool_calls_started"], 1)
        self.assertEqual(result["diagnostics"]["timed_out_operations"], ["tool call srv.slow"])


if __name__ == "__main__":
    unittest.main()
