"""Mark the ctypes runtime distribution as a platform wheel."""

from __future__ import annotations

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class _PlatformWheel(_bdist_wheel):
    """Use a platform tag without claiming a CPython extension ABI."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(cmdclass={"bdist_wheel": _PlatformWheel})
