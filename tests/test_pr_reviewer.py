import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import urllib.request
import json

# Add Tools directory to path and import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../Tools")))
from pr_reviewer import main


class TestPRReviewer(unittest.TestCase):
    def setUp(self):
        # Clear/store environment variables
        self.env_patcher = patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-api-key"})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("subprocess.check_output")
    @patch("urllib.request.urlopen")
    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_reviewer_success(self, mock_remove, mock_exists, mock_run, mock_urlopen, mock_check_output):
        """Verify reviewer success path: diff fetched, API called, review posted."""
        # 1. Mock diff
        mock_check_output.return_value = "diff --git a/file.py b/file.py\n+new_line"

        # 2. Mock DeepSeek HTTP response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": "Looks good! No issues found."
                    }
                }
            ]
        }).encode("utf-8")
        
        # Make urlopen work as a context manager returning mock_response
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # 3. Mock file exist check for cleanup
        mock_exists.return_value = True

        # Mock open to avoid writing to disk during tests
        with patch("builtins.open", mock_open()) as mock_file:
            main()
            
            # Assert file was written with the expected heading + response
            mock_file.assert_called_once_with("review.md", "w", encoding="utf-8")
            mock_file().write.assert_called_once_with("### 🤖 DeepSeek AI Code Review\n\nLooks good! No issues found.")

        # Verify API request details
        mock_urlopen.assert_called_once()
        req_arg = mock_urlopen.call_args[0][0]
        self.assertIsInstance(req_arg, urllib.request.Request)
        self.assertEqual(req_arg.get_header("Authorization"), "Bearer test-api-key")
        self.assertEqual(req_arg.full_url, "https://api.deepseek.com/v1/chat/completions")

        # Verify gh PR review call
        mock_run.assert_called_once_with(["gh", "pr", "review", "--comment", "-F", "review.md"], check=True)
        # Verify cleanup
        mock_remove.assert_called_once_with("review.md")

    @patch("subprocess.check_output")
    def test_missing_api_key(self, mock_check_output):
        """Verify script exits when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 1)

    @patch("subprocess.check_output")
    @patch("urllib.request.urlopen")
    def test_empty_diff(self, mock_urlopen, mock_check_output):
        """Verify script exits early with no API call if the diff is empty."""
        mock_check_output.return_value = "   \n "
        
        main()
        
        # urlopen should never be called since diff is empty
        mock_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
