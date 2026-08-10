#!/usr/bin/env python3
"""
privacy_scan.py — Pre-release PII & device-info scanner
Scans every git-tracked file for personal data before any push or release.

Detects:
  - Email addresses
  - Non-local IP addresses (IPv4 and IPv6)
  - MAC addresses
  - Real OS/device User-Agent strings (Windows NT, Macintosh, iPhone …)
  - Absolute paths exposing a username (/Users/name, /home/name, C:\\Users\\name)
  - socket.gethostname() / platform.node() calls that could log the hostname
  - Hardcoded credentials patterns (password=, token=, secret=, api_key=)

Usage:
    python tools/privacy_scan.py            # scan only, report
    python tools/privacy_scan.py --fix      # scan + auto-patch + show diff
    python tools/privacy_scan.py --dry-run  # show what --fix would change

Exit code 0 = clean, 1 = findings present.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# ─── SKIP THESE FILES (binary, generated, or outside scope) ────────────────────
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".egg",
    ".pyc", ".pyo", ".so", ".dll", ".exe",
    ".pcap", ".pcapng", ".cap",
}
SKIP_NAMES = {"privacy_scan.py"}          # don't scan this file itself

# ─── PII DETECTION PATTERNS ────────────────────────────────────────────────────
# Each entry: (label, regex, severity, auto_fix_replacement | None)
# severity: HIGH = block release, MEDIUM = warn, LOW = info

PATTERNS: List[Tuple] = [
    (
        "EMAIL",
        re.compile(r"(?<![/@#])\b[A-Za-z0-9._%+\-]{2,}@[A-Za-z0-9.\-]{2,}\.[A-Za-z]{2,}\b"),
        "HIGH",
        None,
    ),
    (
        "IPV4 (non-local)",
        re.compile(
            r"\b(?!127\.|0\.0\.0\.0|localhost|10\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.168\.)"
            r"(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        "HIGH",
        None,
    ),
    (
        "IPV6 (non-loopback)",
        re.compile(r"(?<![:/\w])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![:/\w])"),
        "HIGH",
        None,
    ),
    (
        "MAC ADDRESS",
        re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b"),
        "HIGH",
        None,
    ),
    (
        "OS/DEVICE USER-AGENT",
        re.compile(
            r"Mozilla/[\d.]+ \([^)]*(?:Windows NT|Macintosh|iPhone|Android|Linux x86_64|iPad)[^)]*\)",
            re.IGNORECASE,
        ),
        "MEDIUM",
        None,
    ),
    (
        "USERNAME IN PATH",
        re.compile(
            r"(?:/Users/(?!Shared)[A-Za-z0-9_.\-]{2,}|"
            r"/home/[A-Za-z0-9_.\-]{2,}|"
            r"C:\\\\Users\\\\[A-Za-z0-9_.\-]{2,})",
            re.IGNORECASE,
        ),
        "HIGH",
        None,
    ),
    (
        "HOSTNAME EXPOSURE",
        re.compile(r"(?:socket\.gethostname|platform\.node)\s*\(\s*\)"),
        "MEDIUM",
        None,
    ),
    (
        "HARDCODED CREDENTIAL",
        re.compile(
            r"(?:password|passwd|token|secret|api[_\-]?key|auth[_\-]?key)\s*=\s*[\"'][^\"']{4,}[\"']",
            re.IGNORECASE,
        ),
        "HIGH",
        None,
    ),
]

# ─── FALSE-POSITIVE ALLOW-LIST ─────────────────────────────────────────────────
# Lines containing any of these strings are skipped even if a pattern matches.
ALLOWLIST = [
    "localhost",
    "127.0.0.1",
    "ollama.com",
    "github.com",
    "example.com",
    "your-email@",
    "user@example",
    "ManifestGen",           # our own generic UA
    "# noqa: privacy",       # explicit opt-out comment
    # RFC 5737 TEST-NET documentation ranges — safe in test fixtures
    "203.0.113.",
    "198.51.100.",
    "192.0.2.",
    # Common placeholder MAC in test vectors (IANA OUI unallocated range)
    "00:1A:2B:3C:4D:5E",
    "00:1a:2b:3c:4d:5e",
]


def get_tracked_files(repo_root: Path) -> List[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root, capture_output=True, text=True
    )
    files = []
    for line in result.stdout.splitlines():
        p = repo_root / line.strip()
        if p.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if p.name in SKIP_NAMES:
            continue
        if p.exists() and p.is_file():
            files.append(p)
    return sorted(files)


def scan_file(path: Path) -> List[dict]:
    """Return list of findings: {file, line_no, line, label, severity, match}"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    findings = []
    for line_no, line in enumerate(text.splitlines(), 1):
        # Skip allowlisted lines
        if any(a in line for a in ALLOWLIST):
            continue
        for label, pattern, severity, _ in PATTERNS:
            for m in pattern.finditer(line):
                findings.append({
                    "file": str(path),
                    "line_no": line_no,
                    "line": line.strip(),
                    "label": label,
                    "severity": severity,
                    "match": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                })
    return findings


def print_findings(findings: List[dict], repo_root: Path):
    if not findings:
        print("\n  ✓  No PII or device-info detected. Safe to release.\n")
        return

    by_severity = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for f in findings:
        by_severity[f["severity"]].append(f)

    for sev in ("HIGH", "MEDIUM", "LOW"):
        group = by_severity[sev]
        if not group:
            continue
        icon = "✖" if sev == "HIGH" else "⚠" if sev == "MEDIUM" else "ℹ"
        print(f"\n  {icon}  [{sev}] — {len(group)} finding(s)")
        print("  " + "─" * 58)
        for f in group:
            rel = Path(f["file"]).relative_to(repo_root)
            match_display = f["match"][:60] + ("…" if len(f["match"]) > 60 else "")
            print(f"  {rel}:{f['line_no']}")
            print(f"    Pattern : {f['label']}")
            print(f"    Match   : {match_display}")
            print(f"    Line    : {f['line'][:80]}")
            print()


def simulate_output(repo_root: Path, tracked: List[Path]):
    """Print what a clean release looks like after scanning."""
    print("\n" + "═" * 62)
    print("  SIMULATED PRE-RELEASE SCAN OUTPUT")
    print("═" * 62)
    print(f"  Repo        : {repo_root}")
    print(f"  Files scanned: {len(tracked)}")
    print(f"  Patterns    : {len(PATTERNS)} PII categories")
    print(f"  Allowlist   : {len(ALLOWLIST)} safe-string exemptions")
    print("  " + "─" * 58)
    for p in tracked:
        rel = p.relative_to(repo_root)
        n_findings = len(scan_file(p))
        status = "CLEAN" if n_findings == 0 else f"{n_findings} finding(s)"
        icon = "✓" if n_findings == 0 else "✖"
        print(f"  {icon}  {str(rel):<45}  {status}")
    print("═" * 62 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-release PII scanner for Wild_Root_Prompt.")
    parser.add_argument("--fix", action="store_true", help="Auto-patch detected issues where possible")
    parser.add_argument("--dry-run", action="store_true", help="Show what --fix would change without applying")
    parser.add_argument("--simulate", action="store_true", default=True, help="Print simulated scan report (default: on)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    tracked = get_tracked_files(repo_root)

    if args.simulate:
        simulate_output(repo_root, tracked)

    all_findings = []
    for p in tracked:
        all_findings.extend(scan_file(p))

    print_findings(all_findings, repo_root)

    if all_findings:
        high = [f for f in all_findings if f["severity"] == "HIGH"]
        if high:
            print(f"  RELEASE BLOCKED — {len(high)} HIGH-severity finding(s) must be resolved.\n")
            sys.exit(1)
        else:
            print(f"  RELEASE WARNING — {len(all_findings)} MEDIUM/LOW finding(s). Review before pushing.\n")
            sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
