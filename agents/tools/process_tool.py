"""
agents.tools.process_tool — Ejecución segura de comandos externos.

Todas las herramientas que envuelven un binario de sistema (git, docker...)
pasan por aquí en vez de llamar a `subprocess` directamente, para que las
reglas de seguridad (`shell=False`, timeout, captura de stderr) se apliquen
en un único sitio.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agents.exceptions import MissingDependencyError, ToolExecutionError

DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class ProcessResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def require_binary(binary: str, *, cwd: Path | None = None) -> None:
    """
    Lanza MissingDependencyError con un mensaje útil si el binario no está en PATH.

    Si hay un `.venv` en ``cwd`` (o ascendiendo desde él) o el intérprete
    actual vive en un `.venv`, también busca ahí: en este entorno `uv` está en
    `.venv/bin/uv` y no en el PATH del sistema, y sin este fallback el doctor,
    `env sync` y `dependency` fallaban por «uv no instalado» pese a estar
    disponible.
    """
    if shutil.which(binary) is not None:
        return
    if _resolver_binario(binary, cwd) is not None:
        return
    raise MissingDependencyError(f"'{binary}' no está instalado o no está en el PATH. Este agente necesita el binario '{binary}' disponible en el sistema.")


def _resolver_binario(binary: str, cwd: Path | None) -> str | None:
    """Busca `<cwd>/.venv/bin/<binary>` (subiendo hacia la raíz) y, si el
    intérprete actual corre dentro de un venv, `<venv-actual>/bin/<binary>`."""
    candidatos: list[Path] = []
    if cwd is not None:
        actual = cwd.resolve()
        while True:
            candidatos.append(actual / ".venv" / "bin" / binary)
            if actual.parent == actual:
                break
            actual = actual.parent
    sys_prefix = Path(sys.prefix)
    if (sys_prefix / "bin" / binary).is_file():
        candidatos.append(sys_prefix / "bin" / binary)
    for c in candidatos:
        if c.is_file():
            return str(c)
    return None


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    check: bool = False,
) -> ProcessResult:
    """
    Ejecuta `args` (nunca a través de una shell — evita inyección de comandos)
    y devuelve un `ProcessResult`. Si `check=True`, lanza `ToolExecutionError`
    cuando el returncode no es 0.
    """
    if not args:
        raise ValueError("run_command requiere una lista de argumentos no vacía.")

    require_binary(args[0], cwd=cwd)
    # El binario puede no estar en el PATH del sistema pero sí en el .venv del
    # proyecto o en el venv del intérprete actual. Se ejecuta la ruta completa.
    if shutil.which(args[0]) is None:
        en_venv = _resolver_binario(args[0], cwd)
        if en_venv is not None:
            args = [en_venv, *args[1:]]

    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            f"El comando '{' '.join(args)}' superó el timeout de {timeout}s.",
        ) from exc

    result = ProcessResult(
        command=args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )

    if check and not result.ok:
        raise ToolExecutionError(
            f"El comando '{' '.join(args)}' falló (exit {result.returncode}): {result.stderr.strip()}",
            returncode=result.returncode,
            stderr=result.stderr,
        )

    return result
