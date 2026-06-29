import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

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
from tool_orchestrator import Tools


class TestOutputPolicy(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Tools()

    def test_no_truncation_when_under_limit(self):
        """Verify that short output is not truncated."""
        content = "short content"
        output, original, final = self.orchestrator._apply_output_policy("server1", content)
        self.assertEqual(output, content)
        self.assertEqual(original, len(content))
        self.assertEqual(final, len(content))

    def test_head_tail_truncation(self):
        """Verify that long output is truncated keeping head and tail."""
        self.orchestrator.valves.default_max_output_tokens = 10  # 40 chars max
        self.orchestrator.valves.output_truncation_strategy = "head_tail"
        
        content = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLM" # 49 chars
        output, original, final = self.orchestrator._apply_output_policy("server1", content)
        
        self.assertEqual(original, 49)
        self.assertIn("omitted by output policy", output)
        self.assertTrue(output.startswith("abcdefghijklmnopqrst"))
        self.assertTrue(output.endswith("IJKLM"))

    def test_truncate_strategy(self):
        """Verify that long output is truncated keeping only the head."""
        self.orchestrator.valves.default_max_output_tokens = 10  # 40 chars max
        self.orchestrator.valves.output_truncation_strategy = "truncate"
        
        content = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLM" # 49 chars
        output, original, final = self.orchestrator._apply_output_policy("server1", content)
        
        self.assertEqual(original, 49)
        self.assertIn("exceeded limit", output)
        self.assertTrue(output.startswith("abcdefghijklmnopqrst"))
        self.assertFalse(output.endswith("IJKLM"))

    def test_server_override_limit(self):
        """Verify that per-server override limits are applied."""
        self.orchestrator.valves.default_max_output_tokens = 1000
        self.orchestrator.valves.server_token_limits = "databricks=10" # override databricks to 40 chars max
        self.orchestrator.valves.output_truncation_strategy = "truncate"
        
        content = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLM" # 49 chars
        
        # Test default server (should not be truncated because default is 1000 tokens)
        output_def, _, _ = self.orchestrator._apply_output_policy("other-server", content)
        self.assertEqual(output_def, content)
        
        # Test databricks server (should be truncated to 40 chars max)
        output_db, _, _ = self.orchestrator._apply_output_policy("databricks", content)
        self.assertIn("exceeded limit", output_db)


class TestSearchAndIndexing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = Tools()
        self.mock_user = MockUserModel(id="test-user", role="admin")

    @patch("open_webui.utils.tools.get_tool_servers", new_callable=AsyncMock)
    @patch("open_webui.models.tools.Tools.get_tools", new_callable=AsyncMock)
    @patch("open_webui.utils.access_control.has_connection_access", new_callable=AsyncMock)
    @patch("open_webui.models.groups.Groups.get_groups_by_member_id", new_callable=AsyncMock)
    async def test_ensure_index_and_search(self, mock_groups, mock_access, mock_get_tools, mock_get_servers):
        """Verify indexing and search works with mock servers and local tools."""
        # 1. Setup mock data
        mock_get_servers.return_value = [
            {
                "id": "jira-server",
                "idx": 0,
                "url": "http://jira.local",
                "openapi": {"info": {"title": "Jira API"}},
                "specs": [
                    {"name": "create_ticket", "description": "Create a new Jira ticket", "parameters": {}}
                ]
            }
        ]
        
        mock_tool_db = MagicMock()
        mock_tool_db.id = "my_local_tool"
        mock_tool_db.name = "My Local Tool"
        mock_tool_db.specs = [
            {"name": "local_func", "description": "Calculate local variables", "parameters": {}}
        ]
        mock_get_tools.return_value = [mock_tool_db]
        
        # Mock connections setup
        mock_request = MagicMock()
        mock_request.app.state.config.TOOL_SERVER_CONNECTIONS = [
            {"type": "openapi", "config": {}, "info": {"id": "jira-server", "name": "Jira"}}
        ]
        mock_request.app.state.EMBEDDING_FUNCTION = None # fallback to BM25

        mock_groups.return_value = []
        mock_access.return_value = True

        # 2. Trigger index build
        result = await self.orchestrator.refresh_index(
            __request__=mock_request,
            __user__=self.mock_user,
            __id__="tool_orchestrator"
        )
        
        # 3. Check search functionality
        search_result = await self.orchestrator.search_tools(
            query="Jira ticket",
            __request__=mock_request,
            __user__=self.mock_user
        )
        
        self.assertIn("create_ticket", search_result)
        self.assertIn("jira-server", search_result)


if __name__ == "__main__":
    unittest.main()
