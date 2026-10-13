# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal cross-platform compilation test for BCR presubmit configuration."""

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

# (BCR platform, Bazel CPU name, Clang target triple).
_BCR_PLATFORMS = (
    ("debian11", "k8", "x86_64-unknown-linux-gnu"),
    ("ubuntu2204", "k8", "x86_64-unknown-linux-gnu"),
    ("macos", "darwin_x86_64", "x86_64-apple-darwin24.2.0"),
    ("macos_arm64", "darwin_arm64", "arm64-apple-darwin24.2.0"),
)

# Exercises C++17 (`namespace a::b`, `if constexpr`) and `crc32c@1.1.0`
# `src/crc32c_sse42.cc` (`HAVE_SSE42` + `_mm_crc32_u8`) without host headers.
_TEST_SOURCE = """\
#define HAVE_SSE42 1
namespace crc32c::internal {
#if HAVE_SSE42 && defined(__x86_64__)
static inline unsigned __attribute__((__always_inline__, __target__("crc32")))
_mm_crc32_u8(unsigned c, unsigned char d) {
  return __builtin_ia32_crc32qi(c, d);
}
unsigned ExtendSse42(unsigned c, const unsigned char* p) {
  if constexpr (sizeof(c) == 4) return _mm_crc32_u8(c, *p);
}
#endif
}  // namespace crc32c::internal
"""


def _find_clang() -> str | None:
  """Returns an executable Clang compiler path for this host architecture."""
  candidates: list[pathlib.Path] = []
  if cc := os.environ.get("CC"):
    candidates.extend([pathlib.Path(cc).parent / "clang", pathlib.Path(cc)])
  for name in ("clang++-18", "clang++", "clang-18", "clang"):
    if found := shutil.which(name):
      candidates.append(pathlib.Path(found))
  for path in candidates:
    try:
      if (
          path.is_file()
          and subprocess.run(
              [str(path), "--version"], capture_output=True, check=False
          ).returncode
          == 0
      ):
        return str(path)
    except OSError:
      continue
  return None


def _load_bcr_platform_flags(
    presubmit_yml: pathlib.Path,
) -> dict[str, list[str]]:
  """Returns {platform: build_flags} expanded from `.bcr/presubmit.yml`."""
  matrix: dict[str, list[str]] = {}
  platform_flags: dict[str, list[str]] = {}
  section = subkey = task_platform = ""
  for raw in presubmit_yml.read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].rstrip()
    if not line.strip():
      continue
    indent = len(line) - len(line.lstrip(" "))
    val = line.strip()
    if indent == 0:
      section = val.rstrip(":")
    elif section == "matrix" and indent == 2:
      subkey = val.rstrip(":")
      matrix[subkey] = []
    elif section == "matrix" and val.startswith("- "):
      matrix[subkey].append(val[2:].strip())
    elif section == "tasks" and indent == 2:
      subkey = task_platform = ""
    elif section == "tasks" and indent == 4:
      if val.startswith("platform:"):
        task_platform = val.split(":", 1)[1].strip()
      else:
        subkey = val.rstrip(":")
    elif (
        section == "tasks"
        and subkey == "build_flags"
        and val.startswith("- ")
    ):
      flag = val[2:].strip().strip("'\"")
      match = re.fullmatch(r"\$\{\{\s*(\w+)\s*\}\}", task_platform)
      targets = matrix[match.group(1)] if match else [task_platform]
      for plat in targets:
        platform_flags.setdefault(plat, []).append(flag)
  return platform_flags


def _resolve_flags(flags: list[str], bazel_cpu: str, src: str) -> list[str]:
  """Resolves Bazel C++ flags + `crc32c@1.1.0` base copts into compiler args."""
  resolved = ["-msse4.2"] if bazel_cpu in ("k8", "darwin") else []
  for flag in flags:
    if flag.startswith(("--copt=", "--cxxopt=")):
      resolved.append(flag.split("=", 1)[1])
    elif flag.startswith("--per_file_copt="):
      pattern, copt = flag.split("=", 1)[1].split("@", 1)
      if re.search(pattern, src):
        resolved.extend(copt.split(","))
  return resolved


class BcrPresubmitTest(unittest.TestCase):
  """Verifies BCR compilation succeeds across all 4 targeted platforms."""

  def test_bcr_compiles_across_all_4_platforms(self):
    """Compiles C++17 and crc32c_sse42.cc across all 4 BCR platforms."""
    clang = _find_clang()
    if not clang:
      self.skipTest("No executable Clang compiler available")

    here = pathlib.Path(__file__).parent
    presubmit_yml = here.parent / ".bcr/presubmit.yml"
    platform_flags = _load_bcr_platform_flags(presubmit_yml)

    with tempfile.TemporaryDirectory() as tmpdir:
      src = pathlib.Path(tmpdir) / "external/crc32c~/src/crc32c_sse42.cc"
      src.parent.mkdir(parents=True)
      src.write_text(_TEST_SOURCE, encoding="utf-8")

      for platform_name, bazel_cpu, triple in _BCR_PLATFORMS:
        self.assertIn(platform_name, platform_flags)
        cmd = [
            clang,
            f"--target={triple}",
            "-ffreestanding",
            "-Werror=unused-command-line-argument",
            *_resolve_flags(
                platform_flags[platform_name], bazel_cpu, str(src)
            ),
            "-c",
            str(src),
            "-o",
            str(pathlib.Path(tmpdir) / f"{platform_name}.o"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(
            res.returncode, 0, f"{platform_name} failed:\n{res.stderr}"
        )


if __name__ == "__main__":
  unittest.main()
