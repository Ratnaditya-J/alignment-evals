"""Sandboxed code execution for HumanEval/MBPP-style scoring.

This module runs model-generated code in a separate Python subprocess with a
hard wall-clock timeout, no inherited environment, no network access (best
effort via ``http_proxy``/``no_proxy`` env scrubbing), and resource limits on
POSIX systems via ``resource.setrlimit``. Tests are appended to the model's
candidate solution and the subprocess exits non-zero if any test fails.

The scorer integrates with ``BehaviorScorer`` by reading task ``metadata`` for
``humaneval_test_code``/``humaneval_entry_point`` (HumanEval) or ``mbpp_tests``
(MBPP). When neither is present the scorer reports ``passed=False`` with a
clear explanation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .scoring import BehaviorScore, ScoringContext

LOGGER = logging.getLogger(__name__)


# Scrub network-related env vars so subprocess cannot reach the internet via
# common proxy variables. This is best-effort isolation; the calling shell may
# still have ambient configuration.
_ENV_DENYLIST = {
    "ALL_PROXY",
    "FTP_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "ftp_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
}


def _build_subprocess_env() -> Dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _ENV_DENYLIST}
    # Force offline behaviour for common ML libs.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _resource_limiter(
    cpu_seconds: int, memory_bytes: int
):  # pragma: no cover - POSIX-only
    if sys.platform == "win32":
        return None
    import resource

    def _apply() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        except (ValueError, OSError):
            pass

    return _apply


@dataclass(frozen=True)
class CodeExecutionResult:
    """Outcome of running model code against a test harness."""

    passed: bool
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "error": self.error,
        }


class SandboxedCodeExecutor:
    """Run candidate Python code with timeout and resource limits."""

    def __init__(
        self,
        timeout_seconds: float = 8.0,
        cpu_seconds: int = 8,
        memory_bytes: int = 512 * 1024 * 1024,
        python_executable: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.cpu_seconds = cpu_seconds
        self.memory_bytes = memory_bytes
        self.python_executable = python_executable or sys.executable
        self.extra_env = dict(extra_env or {})

    def execute(self, source: str) -> CodeExecutionResult:
        """Write ``source`` to a temp file and run it under restrictions."""
        env = _build_subprocess_env()
        env.update(self.extra_env)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "candidate.py"
            script_path.write_text(source, encoding="utf-8")
            preexec_fn = _resource_limiter(self.cpu_seconds, self.memory_bytes)
            import time

            started = time.perf_counter()
            try:
                process = subprocess.run(
                    [self.python_executable, str(script_path)],
                    env=env,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    preexec_fn=preexec_fn if sys.platform != "win32" else None,
                )
            except subprocess.TimeoutExpired as exc:
                duration = round(time.perf_counter() - started, 4)
                return CodeExecutionResult(
                    passed=False,
                    returncode=-1,
                    stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                    stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                    duration_seconds=duration,
                    timed_out=True,
                    error=f"Execution exceeded {self.timeout_seconds}s",
                )
            except FileNotFoundError as exc:
                return CodeExecutionResult(
                    passed=False,
                    returncode=-1,
                    stdout="",
                    stderr=str(exc),
                    duration_seconds=0.0,
                    timed_out=False,
                    error=str(exc),
                )
            duration = round(time.perf_counter() - started, 4)
            return CodeExecutionResult(
                passed=process.returncode == 0,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                duration_seconds=duration,
                timed_out=False,
            )


_CODE_BLOCK = re.compile(
    r"```(?:python|py)?\s*\n(?P<body>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_python_code(response: str) -> str:
    """Extract Python code from a model response.

    If the response contains fenced code blocks, returns the concatenation of
    every block. Otherwise returns the response verbatim.
    """
    blocks = [match.group("body").strip() for match in _CODE_BLOCK.finditer(response)]
    if blocks:
        return "\n\n".join(blocks)
    return response.strip()


def _build_humaneval_program(
    prompt: str, candidate: str, test_code: str, entry_point: str
) -> str:
    """Compose a runnable program for HumanEval-style scoring.

    Strategy:
        1. The prompt typically contains the function signature and docstring.
        2. The candidate is the body the model should produce.
        3. The test harness defines ``def check(candidate)`` and asserts.
    We always prepend the prompt so signatures + imports are present, and we
    append the test harness with a ``check(<entry_point>)`` invocation.
    """
    parts: List[str] = []
    parts.append(prompt.rstrip())
    candidate_code = extract_python_code(candidate)
    if entry_point not in candidate_code and "def " not in candidate_code:
        # Indent so the body becomes part of the prompted signature.
        candidate_code = textwrap.indent(candidate_code, "    ")
    parts.append(candidate_code)
    parts.append(test_code.rstrip())
    parts.append(f"check({entry_point})")
    return "\n\n".join(parts) + "\n"


def _build_mbpp_program(candidate: str, test_lines: Sequence[str]) -> str:
    """Compose an MBPP-style runnable program."""
    candidate_code = extract_python_code(candidate)
    test_block = "\n".join(test_lines)
    return f"{candidate_code}\n\n{test_block}\n"


class HumanEvalScorer:
    """BehaviorScorer that runs HumanEval unit tests in a subprocess."""

    name = "humaneval_unit_tests"

    def __init__(self, executor: Optional[SandboxedCodeExecutor] = None):
        self.executor = executor or SandboxedCodeExecutor()

    def score(self, response: str, context: ScoringContext) -> BehaviorScore:
        prompt = context.metadata.get("humaneval_prompt", "")
        test_code = context.metadata.get("humaneval_test_code", "")
        entry_point = context.metadata.get("humaneval_entry_point", "")
        if not test_code or not entry_point:
            return BehaviorScore(
                name=self.name,
                score=0.0,
                passed=False,
                explanation="missing humaneval_test_code or humaneval_entry_point in task metadata",
            )
        program = _build_humaneval_program(prompt, response, test_code, entry_point)
        result = self.executor.execute(program)
        return BehaviorScore(
            name=self.name,
            score=1.0 if result.passed else 0.0,
            passed=result.passed,
            explanation=_explain(result),
        )


class MBPPScorer:
    """BehaviorScorer that runs MBPP unit tests in a subprocess."""

    name = "mbpp_unit_tests"

    def __init__(self, executor: Optional[SandboxedCodeExecutor] = None):
        self.executor = executor or SandboxedCodeExecutor()

    def score(self, response: str, context: ScoringContext) -> BehaviorScore:
        tests = context.metadata.get("mbpp_tests", "")
        if not tests:
            return BehaviorScore(
                name=self.name,
                score=0.0,
                passed=False,
                explanation="missing mbpp_tests in task metadata",
            )
        test_lines = [line for line in tests.splitlines() if line.strip()]
        program = _build_mbpp_program(response, test_lines)
        result = self.executor.execute(program)
        return BehaviorScore(
            name=self.name,
            score=1.0 if result.passed else 0.0,
            passed=result.passed,
            explanation=_explain(result),
        )


def _explain(result: CodeExecutionResult) -> str:
    if result.passed:
        return f"all unit tests passed in {result.duration_seconds}s"
    if result.timed_out:
        return f"execution timed out after {result.duration_seconds}s"
    stderr_summary = result.stderr.strip().splitlines()[-3:] if result.stderr else []
    if stderr_summary:
        return "unit tests failed: " + " | ".join(stderr_summary)
    return "unit tests failed (no stderr captured)"


def register_code_scorers(registry: Dict[str, object]) -> Dict[str, object]:
    """Helper that injects the code scorers into a scorer registry."""
    registry[HumanEvalScorer.name] = HumanEvalScorer()
    registry[MBPPScorer.name] = MBPPScorer()
    return registry


def serialize_result(result: CodeExecutionResult) -> str:
    """Serialize a code-execution result for raw artifact logs."""
    return json.dumps(result.to_dict(), sort_keys=True)
