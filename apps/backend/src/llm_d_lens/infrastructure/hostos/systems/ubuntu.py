"""Ubuntu host definition (phase 1: the only supported host OS)."""

from __future__ import annotations

from ..definition import OsDefinition, PackageManager, Requirement

APT = PackageManager(
    name="apt",
    refresh=("apt-get", "update", "-qq"),
    install=("apt-get", "install", "-y", "-qq"),
    manual_install=("apt-get", "install", "-y"),
    query=("dpkg-query", "-W", "-f=${Status}"),
    installed_marker="install ok installed",
    needs_elevation=True,
)

UBUNTU = OsDefinition(
    id="ubuntu",
    display_name="Ubuntu",
    family="linux",
    min_version="22.04",
    architectures=("amd64",),
    installer="LensInstaller-Ubuntu-x86_64-online.sh",
    package_manager=APT,
    requirements=(
        Requirement(("curl",), command="curl"),
        Requirement(("tar",), command="tar"),
        Requirement(("git",), command="git"),
        Requirement(("python3",), command="python3"),
        Requirement(("iproute2",), command="ss"),
        Requirement(("openssl",), command="openssl"),
        Requirement(("python3-venv", "python3-pip"), package="python3-venv"),
        Requirement(("python3-pip",), command="pip3"),
        Requirement(("rsync",), command="rsync"),
    ),
    optional_tools={"sshpass": ("sshpass",)},
    capabilities={"cluster-bootstrap": True},
)
