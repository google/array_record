#!/usr/bin/env python3
"""Build and serve ArrayRecord documentation.

This script provides an easy way to build and serve the documentation locally.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import webbrowser


def run_command(cmd, cwd=None):
  """Run a command and return the result."""
  try:
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout, result.stderr
  except subprocess.CalledProcessError as e:
    return False, e.stdout, e.stderr


def install_requirements():
  """Install documentation requirements."""
  print("Installing documentation requirements...")
  success, stdout, stderr = run_command("pip install -r requirements.txt")

  if success:
    print("✓ Requirements installed successfully")
  else:
    print(f"✗ Failed to install requirements: {stderr}")
    return False

  return True


def build_docs(clean=False):
  """Build the documentation."""
  if clean:
    print("Cleaning previous build...")
    success, _, stderr = run_command("make clean")
    if not success:
      print(f"Warning: Failed to clean: {stderr}")

  print("Building HTML documentation...")
  success, stdout, stderr = run_command("make html")

  if success:
    print("✓ Documentation built successfully")
    print(f"HTML files are in: {Path.cwd() / '_build' / 'html'}")
  else:
    print(f"✗ Failed to build documentation: {stderr}")
    return False

  return True


def serve_docs(port=8000, open_browser=True):
  """Serve the documentation locally."""
  html_dir = Path.cwd() / "_build" / "html"

  if not html_dir.exists():
    print("Documentation not found. Building first...")
    if not build_docs():
      return False

  print(f"Serving documentation at http://localhost:{port}")
  print("Press Ctrl+C to stop the server")

  if open_browser:
    webbrowser.open(f"http://localhost:{port}")

  try:
    subprocess.run(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=html_dir,
        check=True,
    )
  except KeyboardInterrupt:
    print("\nStopping server...")
  except subprocess.CalledProcessError as e:
    print(f"Failed to start server: {e}")
    return False

  return True


def live_reload():
  """Start live reload server for development."""
  print("Starting live reload server...")
  print("The documentation will automatically rebuild when files change.")
  print("Press Ctrl+C to stop")

  try:
    success, stdout, stderr = run_command("sphinx-autobuild . _build/html")
    if not success:
      print("sphinx-autobuild not found. Installing...")
      run_command("pip install sphinx-autobuild")
      success, stdout, stderr = run_command("sphinx-autobuild . _build/html")

    if not success:
      print(f"Failed to start live reload server: {stderr}")
      return False

  except KeyboardInterrupt:
    print("\nStopping live reload server...")

  return True


def check_links():
  """Check for broken links in the documentation."""
  print("Checking for broken links...")
  success, stdout, stderr = run_command("make linkcheck")

  if success:
    print("✓ Link check completed successfully")
  else:
    print(f"✗ Link check found issues: {stderr}")
    return False

  return True


def main():
  """Main function."""
  parser = argparse.ArgumentParser(
      description="Build and serve ArrayRecord documentation"
  )

  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # Install command
  install_parser = subparsers.add_parser(
      "install", help="Install documentation requirements"
  )

  # Build command
  build_parser = subparsers.add_parser("build", help="Build documentation")
  build_parser.add_argument(
      "--clean", action="store_true", help="Clean previous build"
  )

  # Serve command
  serve_parser = subparsers.add_parser(
      "serve", help="Serve documentation locally"
  )
  serve_parser.add_argument(
      "--port", type=int, default=8000, help="Port to serve on"
  )
  serve_parser.add_argument(
      "--no-browser", action="store_true", help="Don't open browser"
  )

  # Live reload command
  live_parser = subparsers.add_parser("live", help="Start live reload server")

  # Check links command
  check_parser = subparsers.add_parser("check", help="Check for broken links")

  # All command
  all_parser = subparsers.add_parser("all", help="Install, build, and serve")
  all_parser.add_argument(
      "--port", type=int, default=8000, help="Port to serve on"
  )

  args = parser.parse_args()

  if not args.command:
    parser.print_help()
    return

  # Change to docs directory
  docs_dir = Path(__file__).parent
  os.chdir(docs_dir)

  success = True

  if args.command == "install":
    success = install_requirements()

  elif args.command == "build":
    success = build_docs(clean=args.clean)

  elif args.command == "serve":
    success = serve_docs(port=args.port, open_browser=not args.no_browser)

  elif args.command == "live":
    success = live_reload()

  elif args.command == "check":
    success = check_links()

  elif args.command == "all":
    success = (
        install_requirements()
        and build_docs(clean=True)
        and serve_docs(port=args.port)
    )

  if not success:
    sys.exit(1)


if __name__ == "__main__":
  main()
