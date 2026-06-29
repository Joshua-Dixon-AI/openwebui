import os
import sys
import time
import unittest
from unittest.mock import MagicMock

# 1. Inject mock modules into sys.modules to prevent ModuleNotFoundError when importing tool_orchestrator
mock_users = MagicMock()
mock_tools = MagicMock()
mock_access = MagicMock()
mock_groups = MagicMock()
mock_mcp = MagicMock()
mock_middleware = MagicMock()
mock_config = MagicMock()

# Setup UserModel mock
class MockUserModel:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", "test-user-id")
        self.role = kwargs.get("role", "user")
mock_users.UserModel = MockUserModel

sys.modules["open_webui"] = MagicMock()
sys.modules["open_webui.models"] = MagicMock()
sys.modules["open_webui.models.users"] = mock_users
sys.modules["open_webui.models.tools"] = mock_tools
sys.modules["open_webui.models.groups"] = mock_groups
sys.modules["open_webui.utils"] = MagicMock()
sys.modules["open_webui.utils.tools"] = mock_tools
sys.modules["open_webui.utils.access_control"] = mock_access
sys.modules["open_webui.utils.middleware"] = mock_middleware
sys.modules["open_webui.utils.mcp"] = MagicMock()
sys.modules["open_webui.utils.mcp.client"] = mock_mcp
sys.modules["open_webui.config"] = mock_config

# 2. Add Advanced-Tool-Use to python path and import Tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Tools/Advanced-Tool-Use")))
from tool_orchestrator import Tools, rows, columns


class TestSandboxValidation(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Tools()

    def test_allowed_syntax(self):
        """Verify that basic loops, conditionals, and safe syntax compile successfully."""
        safe_code = """
total = 0
for i in range(10):
    if i % 2 == 0:
        total += i
print(total)
"""
        # Should compile without raising ValueError
        compiled = self.orchestrator._compile_script(safe_code)
        self.assertIsNotNone(compiled)

    def test_disallowed_syntax_import(self):
        """Verify that import statements are blocked."""
        bad_code = "import os"
        with self.assertRaises(ValueError) as ctx:
            self.orchestrator._compile_script(bad_code)
        self.assertIn("imports are not allowed", str(ctx.exception))

    def test_disallowed_syntax_class(self):
        """Verify that class definitions are blocked."""
        bad_code = """
class MyClass:
    pass
"""
        with self.assertRaises(ValueError) as ctx:
            self.orchestrator._compile_script(bad_code)
        self.assertIn("disallowed syntax: ClassDef", str(ctx.exception))

    def test_disallowed_syntax_with(self):
        """Verify that with-blocks are blocked."""
        bad_code = """
with open("test.txt") as f:
    pass
"""
        with self.assertRaises(ValueError) as ctx:
            self.orchestrator._compile_script(bad_code)
        self.assertIn("disallowed syntax: With", str(ctx.exception))

    def test_blocked_names(self):
        """Verify that blocked global function/variable names are rejected."""
        bad_codes = [
            "eval('1+1')",
            "exec('x = 1')",
            "open('file.txt', 'r')",
            "x = __builtins__",
            "x = __class__",
        ]
        for code in bad_codes:
            with self.subTest(code=code):
                with self.assertRaises(ValueError) as ctx:
                    self.orchestrator._compile_script(code)
                self.assertIn("is not allowed", str(ctx.exception))

    def test_blocked_attributes(self):
        """Verify that private attributes and format exploits are blocked."""
        bad_codes = [
            "x = obj._private_attr",
            "x = 'hello'.format()",
            "x = 'hello'.format_map({})",
        ]
        for code in bad_codes:
            with self.subTest(code=code):
                with self.assertRaises(ValueError) as ctx:
                    self.orchestrator._compile_script(code)
                self.assertIn("is not allowed", str(ctx.exception))


class TestTabularRowsHelper(unittest.TestCase):
    def test_rows_list_of_lists(self):
        """Verify rows() with standard list of lists."""
        data = [[1, 2], [3, 4]]
        self.assertEqual(rows(data), [[1, 2], [3, 4]])

    def test_rows_list_of_scalars(self):
        """Verify rows() with flat list of scalars."""
        data = [1, 2, 3]
        self.assertEqual(rows(data), [[1], [2], [3]])

    def test_rows_databricks_format(self):
        """Verify rows() parses Databricks data_array values format."""
        db_result = {
            "data_array": [
                {"values": [{"string_value": "val1"}, {"int_value": 42}]},
                {"values": [{"string_value": "val2"}, {"int_value": 43}]},
            ]
        }
        self.assertEqual(rows(db_result), [["val1", 42], ["val2", 43]])

    def test_rows_list_of_dicts(self):
        """Verify rows() parses a list of dictionaries consistently."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        self.assertEqual(rows(data), [["Alice", 30], ["Bob", 25]])

    def test_columns_databricks_format(self):
        """Verify columns() extracts headers from Databricks schema."""
        db_result = {
            "schema": {
                "columns": [
                    {"name": "col1", "type": "STRING"},
                    {"name": "col2", "type": "INTEGER"},
                ]
            }
        }
        self.assertEqual(columns(db_result), ["col1", "col2"])

    def test_columns_generic_dict(self):
        """Verify columns() extracts keys from generic dictionary formats."""
        data1 = {"columns": ["c1", "c2"]}
        data2 = {"headers": ["h1", "h2"]}
        self.assertEqual(columns(data1), ["c1", "c2"])
        self.assertEqual(columns(data2), ["h1", "h2"])

    def test_columns_list_of_dicts(self):
        """Verify columns() extracts dict keys from a list of dicts."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        self.assertEqual(columns(data), ["a", "b"])


class TestSandboxExecution(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = Tools()
        self.orchestrator.valves.enable_code_mode = True
        self.orchestrator._index.entries = [MagicMock()]
        self.orchestrator._index.built_at = time.time()
        self.mock_user = MagicMock()

    async def test_simple_execution(self):
        """Verify a simple script runs and captures stdout."""
        code = """
print("hello", "world")
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "hello world")

    async def test_sandbox_math(self):
        """Verify math operations inside the sandbox."""
        code = """
import_math_facade = math
print(math.floor(math.pi))
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "3")

    async def test_infinite_loop_timeout(self):
        """Verify infinite loop is terminated by the timeout tracer."""
        self.orchestrator.valves.code_timeout = 1.0
        code = """
while True:
    pass
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertIn("exceeded code_timeout", result)

    async def test_sandbox_urllib_parse(self):
        """Verify urllib.parse functions work inside the sandbox."""
        code = """
print(urllib.parse.quote("hello world/test"))
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "hello%20world/test")

    async def test_sandbox_base64(self):
        """Verify base64 functions work inside the sandbox."""
        code = """
# Since string inputs inside Sandbox are unicode, we must use bytes (via b"") or encode
encoded = base64.b64encode(b"hello")
print(encoded.decode('utf-8'))
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "aGVsbG8=")

    async def test_sandbox_hashlib(self):
        """Verify hashlib functions work inside the sandbox."""
        code = """
h = hashlib.sha256(b"hello").hexdigest()
print(h)
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")

    async def test_sandbox_random(self):
        """Verify random functions work inside the sandbox."""
        code = """
val = random.choice([10, 20, 30])
print(val in [10, 20, 30])
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "True")

    async def test_sandbox_uuid(self):
        """Verify uuid functions work inside the sandbox."""
        code = """
u = uuid.uuid4()
print(len(str(u)))
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "36")

    async def test_sandbox_columns_helper(self):
        """Verify the columns helper works inside the sandbox."""
        code = """
data = [{"a": 1, "b": 2}]
print(columns(data))
"""
        result = await self.orchestrator.run_tool_script(code, __user__=self.mock_user)
        self.assertEqual(result.strip(), "['a', 'b']")


if __name__ == "__main__":
    unittest.main()
