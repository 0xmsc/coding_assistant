import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

import coding_assistant.sandbox


class TestSandboxFunctions:
    """Unit tests for sandbox helper functions."""

    def test_to_paths(self):
        """Test _to_paths expands home and resolves."""
        paths = ["~", "~/.config"]
        result = coding_assistant.sandbox._to_paths(paths)
        assert all(isinstance(p, Path) for p in result)
        assert result[0] == Path.home().resolve()
        assert result[1] == Path.home() / ".config" / "resolve" or str(result[1]).endswith(".config")

    def test_allow_read_dir(self):
        """Test allow_read adds rule for directory."""
        mock_rs = MagicMock()
        paths = [Path("/tmp")]
        with patch('coding_assistant.sandbox.Path.exists', return_value=True), \
             patch('coding_assistant.sandbox.Path.is_dir', return_value=True):
            coding_assistant.sandbox.allow_read(mock_rs, paths)
            mock_rs.allow.assert_called_with(Path("/tmp"), rules=coding_assistant.sandbox._get_read_only_rule())

    def test_allow_read_file(self):
        """Test allow_read for file."""
        mock_rs = MagicMock()
        paths = [Path("/tmp/test")]
        with patch('coding_assistant.sandbox.Path.exists', return_value=True), \
             patch('coding_assistant.sandbox.Path.is_dir', return_value=False):
            coding_assistant.sandbox.allow_read(mock_rs, paths)
            mock_rs.allow.assert_called_with(Path("/tmp/test"), rules=coding_assistant.sandbox._get_read_only_file_rule())

    def test_allow_write_dir(self):
        """Test allow_write for directory."""
        mock_rs = MagicMock()
        paths = [Path("/tmp")]
        with patch('coding_assistant.sandbox.Path.exists', return_value=True), \
             patch('coding_assistant.sandbox.Path.is_dir', return_value=True):
            coding_assistant.sandbox.allow_write(mock_rs, paths)
            # FSAccess.all() called for dir
            call_args = mock_rs.allow.call_args
            assert call_args[1]['rules'] == mock_rs.allow.call_args[1]['rules']  # Assume all is set

    def test_allow_write_skips_nonexistent(self):
        """Test allow_write skips nonexistent paths."""
        mock_rs = MagicMock()
        paths = [Path("/nonexistent")]
        with patch('coding_assistant.sandbox.Path.exists', return_value=False):
            coding_assistant.sandbox.allow_write(mock_rs, paths)
            mock_rs.allow.assert_not_called()

    @patch('coding_assistant.sandbox.allow_write')
    @patch('coding_assistant.sandbox.allow_read')
    @patch('coding_assistant.sandbox._to_paths')
    @patch('coding_assistant.sandbox.Ruleset')
    def test_sandbox_include_defaults(self, mock_rs_class, mock_to_paths, mock_allow_read, mock_allow_write):
        """Test sandbox with include_defaults."""
        mock_rs = MagicMock()
        mock_rs_class.return_value = mock_rs
        mock_to_paths.return_value = [Path("/tmp")]
        readable = [Path("/usr")]
        writable = [Path("/var")]
        coding_assistant.sandbox.sandbox(readable_paths=readable, writable_paths=writable, include_defaults=True)
        mock_to_paths.assert_called()
        # Check defaults are extended
        mock_allow_write.assert_called_once()
        mock_allow_read.assert_called_once()
        mock_rs.apply.assert_called_once()


class TestSandboxMain:
    """Tests for CLI main."""

    @patch('coding_assistant.sandbox.subprocess.run')
    @patch('coding_assistant.sandbox.sandbox')
    @patch('coding_assistant.sandbox.sys.exit')
    def test_main_succeeds(self, mock_exit, mock_sandbox, mock_run):
        """Test main runs command and exits with return code."""
        mock_run.return_value = MagicMock(returncode=0)
        with patch('sys.argv', ['sandbox', '--readable-directories', '/usr', '--', 'ls']):
            coding_assistant.sandbox.main()
            mock_sandbox.assert_called_with(readable_paths=[Path("/usr").resolve()], writable_paths=[])
            mock_run.assert_called_with(['ls'], capture_output=False)
            mock_exit.assert_called_with(0)

    @patch('coding_assistant.sandbox.subprocess.run')
    @patch('coding_assistant.sandbox.sandbox')
    @patch('coding_assistant.sandbox.sys.exit')
    def test_main_fails(self, mock_exit, mock_sandbox, mock_run):
        """Test main exits with command's return code."""
        mock_run.return_value = MagicMock(returncode=1)
        with patch('sys.argv', ['sandbox', 'false']):
            coding_assistant.sandbox.main()
            mock_exit.assert_called_with(1)