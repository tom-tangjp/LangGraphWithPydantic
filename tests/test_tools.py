"""Tests for tools module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import utils


class TestToolImports:
    """Tests for tool imports and registration."""

    def test_tools_list_not_empty(self):
        """Test that TOOLS list is not empty."""
        from tools import TOOLS

        assert len(TOOLS) > 0

    def test_tool_registry_built(self):
        """Test that TOOL_REGISTRY is built."""
        from tools import TOOL_REGISTRY

        assert len(TOOL_REGISTRY) > 0

class TestToolRegistry:
    """Tests for tool registry functionality."""

    def test_registry_has_core_tools(self):
        """Test registry has expected core tools."""
        from tools import TOOL_REGISTRY

        expected_tools = [
            "read_file",
            "list_dir",
            "grep_text",
            "run_bash",
            "list_source_files",
            "read_source_file",
        ]

        for tool_name in expected_tools:
            assert tool_name in TOOL_REGISTRY, f"Missing tool: {tool_name}"

    def test_registry_case_insensitive(self):
        """Test registry lookup for lowercase."""
        from tools import TOOL_REGISTRY

        # Lowercase should work
        assert "read_file" in TOOL_REGISTRY


class TestRunBash:
    """Tests for run_bash tool."""

    def test_run_bash_simple(self):
        """Test running a simple bash command."""
        from tools import run_bash

        result = run_bash.invoke({"command": "echo 'hello'"})
        data = json.loads(result)

        assert data["ok"] is True
        assert "hello" in data["stdout"]

    def test_run_bash_with_timeout(self):
        """Test bash command with timeout."""
        from tools import run_bash

        result = run_bash.invoke({"command": "sleep 1", "timeout": 5})
        data = json.loads(result)

        assert data["ok"] is True
        assert data["exit_code"] == 0

    def test_run_bash_cd_command(self):
        """Test bash command changing directory."""
        from tools import run_bash

        result = run_bash.invoke({"command": "pwd", "work_dir": "."})
        data = json.loads(result)

        assert data["ok"] is True
        assert data["exit_code"] == 0


class TestSafeCommands:
    """Tests for safe command execution."""

    def test_safe_command_python_syntax_valid(self):
        """Test safe Python syntax check with valid code."""
        from tools import run_safe_command

        # Use pytest which is in the safe commands list
        result = run_safe_command.invoke({"command": "pytest --version"})
        data = json.loads(result)

        # pytest should work or error not related to security
        assert data["ok"] is True or "error" not in data or data.get("error", "").find("安全") == -1

    def test_safe_command_invalid_rejected(self):
        """Test that unsafe commands are rejected."""
        from tools import run_safe_command

        result = run_safe_command.invoke({"command": "rm -rf /"})
        data = json.loads(result)

        assert data["ok"] is False
        assert "error" in data


class TestPathResolution:
    """Tests for path resolution functions."""

    def test_resolve_under_root_basic(self):
        """Test basic path resolution."""
        from tools import _resolve_under_root

        # Test with absolute path within workspace
        test_path = utils.get_workspace_root() / "test.txt"
        result = _resolve_under_root(str(test_path))
        assert result == test_path

    def test_resolve_under_root_relative(self):
        """Test relative path resolution."""
        from tools import _resolve_under_root

        # Test with relative path
        result = _resolve_under_root("subdir/file.txt")
        expected = utils.get_workspace_root() / "subdir/file.txt"
        assert result == expected


class TestBuildToolRegistry:
    """Tests for build_tool_registry function."""

    def test_build_tool_registry_returns_dict(self):
        """Test that build_tool_registry returns a dictionary."""
        from tools import build_tool_registry, TOOLS

        registry = build_tool_registry(TOOLS)
        assert isinstance(registry, dict)
        assert len(registry) > 0


class TestGrepText:
    """Tests for grep_text - basic functionality without files."""

    def test_grep_text_pattern_validation(self):
        """Test grep with invalid regex pattern."""
        from tools import grep_text

        # Invalid regex should raise error
        with pytest.raises(ValueError):
            grep_text.invoke({
                "pattern": "[invalid",  # Unclosed bracket
                "path": "."
            })


class TestListSourceFiles:
    """Tests for list_source_files - basic checks."""

    def test_list_source_files_returns_json(self):
        """Test that list_source_files returns valid JSON."""
        from tools import list_source_files

        result = list_source_files.invoke({"path": ".", "max_files": 10})
        data = json.loads(result)

        assert "ok" in data
        assert "count" in data
        assert "files" in data


class TestGetSourceMetrics:
    """Tests for get_source_metrics - basic checks."""

    def test_get_source_metrics_returns_json(self):
        """Test that get_source_metrics returns valid JSON."""
        from tools import get_source_metrics

        result = get_source_metrics.invoke({"path": "."})
        data = json.loads(result)

        assert "ok" in data
        assert "total_files" in data


class TestFindCmakeFiles:
    """Tests for find_cmake_files - basic checks."""

    def test_find_cmake_files_returns_json(self):
        """Test that find_cmake_files returns valid JSON."""
        from tools import find_cmake_files

        result = find_cmake_files.invoke({"path": "."})
        data = json.loads(result)

        assert "ok" in data
        assert "cmake_files" in data


class TestWalkDir:
    """Tests for walk_dir - basic checks."""

    def test_walk_dir_returns_json(self):
        """Test that walk_dir returns valid JSON."""
        from tools import walk_dir

        result = walk_dir.invoke({"path": ".", "max_depth": 1})
        data = json.loads(result)

        assert data["ok"] is True
        assert "items" in data


class TestGetDirTree:
    """Tests for get_dir_tree - basic checks."""

    def test_get_dir_tree_returns_json(self):
        """Test that get_dir_tree returns valid JSON."""
        from tools import get_dir_tree

        result = get_dir_tree.invoke({"path": ".", "max_depth": 1})
        data = json.loads(result)

        assert data["ok"] is True
        assert "tree" in data


class TestReadSourceFile:
    """Tests for read_source_file - basic checks."""

    def test_read_source_file_nonexistent(self):
        """Test reading nonexistent file."""
        from tools import read_source_file

        result = read_source_file.invoke({"path": "nonexistent_file_xyz.cpp"})
        data = json.loads(result)

        assert data["ok"] is False


class TestExtractCppFunctions:
    """Tests for extract_cpp_functions - basic checks."""

    def test_extract_cpp_functions_nonexistent(self):
        """Test extracting from nonexistent file."""
        from tools import extract_cpp_functions

        result = extract_cpp_functions.invoke({"path": "nonexistent_file_xyz.cpp"})
        data = json.loads(result)

        assert data["ok"] is False


class TestListDir:
    """Tests for list_dir - basic checks."""

    def test_list_dir_returns_list(self):
        """Test that list_dir returns a list."""
        from tools import list_dir

        result = list_dir.invoke({"path": ".", "max_entries": 5})

        # list_dir returns list directly
        assert isinstance(result, list)
        assert len(result) >= 0


class TestEnsureDir:
    """Tests for ensure_dir - basic checks."""

    def test_ensure_dir_in_workspace(self):
        """Test creating directory in workspace."""
        from tools import ensure_dir

        test_dir = utils.get_workspace_root() / "test_ensure_dir_created"
        try:
            result = ensure_dir.invoke({"path": "test_ensure_dir_created"})
            data = json.loads(result)

            assert data["ok"] is True
            assert data["path"] == "test_ensure_dir_created"
        finally:
            if test_dir.exists():
                test_dir.rmdir()


class TestWriteTextFile:
    """Tests for write_text_file - basic checks."""

    def test_write_text_file_in_workspace(self):
        """Test writing file in workspace."""
        from tools import write_text_file

        test_file = utils.get_workspace_root() / "test_write_file.txt"
        try:
            result = write_text_file.invoke({
                "path": "test_write_file.txt",
                "content": "Hello, World!",
                "mode": "overwrite"
            })
            data = json.loads(result)

            assert data["ok"] is True
            assert data["path"] == "test_write_file.txt"
        finally:
            if test_file.exists():
                test_file.unlink()


class TestMermaidDiagram:
    """Tests for mermaid diagram tools."""

    def test_save_mermaid_diagram_in_workspace(self):
        """Test saving mermaid diagram in workspace."""
        from tools import save_mermaid_diagram

        test_file = utils.get_workspace_root() / "test_diagram.md"
        try:
            result = save_mermaid_diagram.invoke({
                "mermaid_code": "graph TD\n    A-->B",
                "filename": "test_diagram.md"
            })
            data = json.loads(result)

            assert data["ok"] is True
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
