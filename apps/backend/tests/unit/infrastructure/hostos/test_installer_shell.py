"""Shell integration tests for the online installer and its generated lensctl.sh."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

INSTALLER_NAME = "LensInstaller-Ubuntu-x86_64-online.sh"


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "scripts" / INSTALLER_NAME).is_file():
            return parent
    raise AssertionError("could not locate the repository root")


REPO_ROOT = _repo_root()
INSTALLER = REPO_ROOT / "scripts" / INSTALLER_NAME

PACKAGES_STUB = """\
source "$1"
hostos_exec() {
    case "$1 $2" in
        "packages missing") printf '%s' '__MISSING__' ;;
        "packages hint") printf 'hint:%s\\n' "${*:3}" ;;
        "packages show") printf 'show:%s\\n' "${*:3}" ;;
        "packages install") printf 'install:%s\\n' "${*:3}" ;;
        *) return 2 ;;
    esac
}
__SETUP__
ensure_system_packages
"""


def _bash(source: str, *args: object, env: dict[str, str] | None = None, cwd: Path | None = None):
    run_env = dict(os.environ)
    run_env.pop("INSTALL_DIR", None)
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", "-c", source, "_", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
        cwd=cwd,
    )


def _packages_script(missing: list[str], setup: str = "") -> str:
    return PACKAGES_STUB.replace("__MISSING__", "".join(f"{package}\n" for package in missing)).replace(
        "__SETUP__", setup
    )


def test_installer_sources():
    result = _bash(
        'source "$1"\ndeclare -F online_main >/dev/null\ndeclare -F write_control_script >/dev/null', INSTALLER
    )
    assert result.returncode == 0, result.stderr


def test_port_probe_aborts_on_hostos_usage_error():
    script = """\
source "$1"
hostos_exec() { printf 'hostos boom\\n' >&2; return 2; }
if port_in_use 54321; then
    echo in-use
else
    echo free
fi
echo continued
"""
    result = _bash(script, INSTALLER)
    assert result.returncode != 0
    assert "free" not in result.stdout
    assert "continued" not in result.stdout
    assert "hostos boom" in result.stderr


def test_find_free_port_aborts_on_hostos_usage_error():
    script = """\
source "$1"
hostos_exec() { printf 'hostos boom\\n' >&2; return 2; }
find_free_port 54321
echo reported
"""
    result = _bash(script, INSTALLER)
    assert result.returncode != 0
    assert "reported" not in result.stdout
    assert "hostos boom" in result.stderr


def test_ensure_system_packages_nothing_missing():
    result = _bash(_packages_script([]), INSTALLER)
    assert result.returncode == 0, result.stderr
    assert "all required system packages are present" in result.stdout


def test_ensure_system_packages_no_apt_reports_manual_hint():
    result = _bash(_packages_script(["curl", "rsync"], "ALLOW_SYSTEM_PACKAGES=0"), INSTALLER)
    assert result.returncode != 0
    assert "cannot install missing packages (--no-apt set)" in result.stderr
    assert "Install manually: hint:curl rsync" in result.stderr


def test_ensure_system_packages_silent_installs():
    result = _bash(_packages_script(["curl"], "SILENT=1"), INSTALLER)
    assert result.returncode == 0, result.stderr
    assert "show:curl" in result.stdout
    assert "install:curl" in result.stdout


def test_parse_args_rejects_invalid_port():
    result = _bash('source "$1"\nparse_args --app-port 30a0', INSTALLER)
    assert result.returncode != 0
    assert "--app-port must be a port between 1 and 65535 (got '30a0')" in result.stderr


def test_resolve_ports_silently_rejects_invalid_port():
    result = _bash('source "$1"\nAPP_PORT=0\nresolve_ports_silently', INSTALLER)
    assert result.returncode != 0
    assert "app port must be a port between 1 and 65535 (got '0')" in result.stderr


def test_ask_port_reasks_on_invalid_input(tmp_path: Path):
    answer_file = tmp_path / "answered"
    script = """\
source "$1"
ANSWER_FILE="$2"
ask() {
    if [[ ! -f "$ANSWER_FILE" ]]; then
        : > "$ANSWER_FILE"
        printf '30a0\\n'
    else
        printf '3000\\n'
    fi
}
port_in_use() { return 1; }
ask_port "Port" 3000
"""
    result = _bash(script, INSTALLER, answer_file)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3000"
    assert "not valid" in result.stderr


def _prepare_lensctl(tmp_path: Path, python_body: str) -> tuple[Path, Path]:
    install_dir = tmp_path / "install"
    (install_dir / "hack").mkdir(parents=True)
    (install_dir / "hack" / "storage-env.sh").write_text(
        'export LENS_LOG_DIR="$PWD/lens-log"\nexport LENS_DATA_DIR="$PWD/lens-data"\n',
        encoding="utf-8",
    )
    (install_dir / ".lens-install.env").write_text("DB_MODE=external\nENABLE_HTTPS=0\n", encoding="utf-8")
    marker = tmp_path / "python-calls"
    fake_python = install_dir / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(python_body.replace("__MARKER__", str(marker)), encoding="utf-8")
    fake_python.chmod(0o755)

    result = _bash('INSTALL_DIR="$2"\nsource "$1"\nwrite_control_script', INSTALLER, install_dir)
    assert result.returncode == 0, result.stderr
    return install_dir, marker


def test_generated_lensctl_parses_and_stop_does_not_call_hostos(tmp_path: Path):
    install_dir, marker = _prepare_lensctl(tmp_path, '#!/bin/sh\nprintf called >> "__MARKER__"\nexit 0\n')

    lensctl = install_dir / "lensctl.sh"
    syntax = subprocess.run(["bash", "-n", str(lensctl)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    stop = subprocess.run(["bash", str(lensctl), "stop"], capture_output=True, text=True, check=False)
    assert stop.returncode == 0, stop.stderr
    assert not marker.exists(), "stop must not invoke hostos"


def test_generated_lensctl_aborts_on_hostos_usage_error(tmp_path: Path):
    install_dir, marker = _prepare_lensctl(
        tmp_path,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "__MARKER__"\n'
        'if [ "$1" = "-m" ] && [ "$2" = "llm_d_lens.infrastructure.hostos" ]; then exit 2; fi\n'
        "exit 0\n",
    )

    result = subprocess.run(
        ["bash", str(install_dir / "lensctl.sh"), "start"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "hostos command failed (exit 2)" in result.stderr
    assert not (install_dir / ".prism-pids" / "app.pid").exists()
    calls = marker.read_text(encoding="utf-8") if marker.exists() else ""
    assert "llm_d_lens.infrastructure.hostos" in calls
    assert " -c " not in calls


def test_generated_lensctl_explains_silent_host_check_failure(tmp_path: Path):
    install_dir, _ = _prepare_lensctl(tmp_path, "#!/bin/sh\nexit 1\n")

    result = subprocess.run(
        ["bash", str(install_dir / "lensctl.sh"), "start"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "host check failed" in result.stderr
