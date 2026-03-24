"""LangChain ChatModel wrapper for Cursor CLI."""

import os
import re
import subprocess
import shutil
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Regex to strip ANSI escape sequences
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


class ChatCursorCLI(BaseChatModel):
    """Chat model that delegates to Cursor CLI via subprocess.

    Requires Cursor CLI to be installed and logged in.
    Uses `cursor agent -p "<prompt>"` in non-interactive mode.
    """

    model: str = "cursor"
    """Model to pass to Cursor CLI via --model flag. If 'cursor', no --model flag is sent."""

    timeout: int = 120
    """Timeout in seconds for the subprocess call."""

    @property
    def _llm_type(self) -> str:
        return "cursor-cli"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Find cursor CLI binary
        cursor_bin = self._find_cursor_cli()
        if not cursor_bin:
            raise RuntimeError(
                "Cursor CLI not found. Please install it: "
                "curl https://cursor.com/install -fsS | bash  "
                "Or ensure /Applications/Cursor.app is installed on macOS."
            )

        # Convert LangChain messages to a single text prompt
        prompt = self._format_messages(messages)

        # Build command
        cmd = [cursor_bin, "agent", "-p", prompt]
        if self.model and self.model != "cursor":
            cmd.extend(["--model", self.model])

        # Use a clean environment to reduce noise in output
        env = os.environ.copy()
        env["NO_COLOR"] = "1"  # Disable colors if supported

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Cursor CLI timed out after {self.timeout}s. "
                "You can increase timeout via CURSOR_CLI_TIMEOUT env var."
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Cursor CLI exited with code {result.returncode}: {stderr}"
            )

        content = self._clean_output(result.stdout)
        if not content:
            raise RuntimeError("Cursor CLI returned empty response.")

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @staticmethod
    def _clean_output(raw: str) -> str:
        """Strip ANSI codes, spinner lines, and other CLI noise from output."""
        # Remove ANSI escape sequences
        text = _ANSI_RE.sub("", raw)
        # Remove common spinner / progress patterns (lines with only symbols)
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines at the start
            if not cleaned and not stripped:
                continue
            # Skip spinner-like lines (e.g. "⠋ Thinking...", "● ...", progress bars)
            if stripped and stripped[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏●○◉◎▓░█▒":
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    @staticmethod
    def _find_cursor_cli() -> Optional[str]:
        """Locate the cursor CLI binary."""
        # Check PATH first
        path_bin = shutil.which("cursor")
        if path_bin:
            return path_bin

        # macOS: check inside Cursor.app bundle
        app_bin = "/Applications/Cursor.app/Contents/Resources/app/bin/cursor"
        if os.path.isfile(app_bin) and os.access(app_bin, os.X_OK):
            return app_bin

        # Linux / custom install locations
        for candidate in [
            os.path.expanduser("~/.local/bin/cursor"),
            "/usr/local/bin/cursor",
        ]:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        return None

    @staticmethod
    def _format_messages(messages: List[BaseMessage]) -> str:
        """Convert LangChain messages into a single text prompt for Cursor CLI."""
        parts = []
        for msg in messages:
            role = msg.type  # "human", "ai", "system"
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if role == "system":
                parts.append(f"[System]\n{text}")
            elif role == "human":
                parts.append(f"[User]\n{text}")
            elif role == "ai":
                parts.append(f"[Assistant]\n{text}")
            else:
                parts.append(f"[{role}]\n{text}")
        return "\n\n".join(parts)
