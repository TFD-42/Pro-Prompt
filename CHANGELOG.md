# Changelog

All notable changes to Wild_Root_Prompt are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.4.0] — 2026-08-10

### Added
- **25-topic index first-pass selection** — `--recommend-techniques` now maps tasks to one of 25 topics (all 173 techniques reachable, bilingual EN/FR triggers) via `match_topic()` before falling back to the legacy QUICK_REFERENCE scoring.
- **CI pipeline** (`.github/workflows/ci.yml`) — Python 3.8–3.13 matrix, compile check, unittest, and privacy scan gate on every push.
- **CodeQL scanning** (`.github/workflows/codeql.yml`) — Python security-extended queries on push/PR and weekly schedule.
- **SECURITY.md** — private vulnerability reporting via GitHub Security Advisories.
- **89-test suite** (`tests/test_core.py`) — covers anonymization, PII patterns, topic index integrity, technique parsing, SSRF block-list, settings loading, and more.

### Fixed
- **Bug A** — `--techniques all` previously returned an empty list; now refused by design with a pedagogical warning, falls back to the default set.
- **Bug B** — `--anonymize` was corrupting LAN IP addresses (`192.168.x.x`) by matching the PHONE pattern; fixed with negative-lookahead guards.
- **Bug C** — Auto technique selection ran on raw text in the interactive menu but on enriched text in the CLI; unified to always run after the pre-processor.
- **CodeQL High #1** — Replaced regex-based HTML stripping in `fetch_page_text()` with `html.parser.HTMLParser`.
- **CodeQL Medium #2/#3** — `str(e)` removed from API error responses in `web_server.py`; exceptions are now logged internally.

### Changed
- `recommend_techniques()` refactored: topic-index first → QUICK_REFERENCE fallback → DEFAULT_TECHNIQUES.
- Privacy scan allowlist extended with RFC 5737 TEST-NET ranges and placeholder MAC for test fixtures.

---

## [2.3.0] — 2026-07-09

### Added
- Web UI output save.
- Version bump, README v2.3.

---

## [2.2.0] — 2026-06-xx

*(earlier entries — fill in from git log as needed)*

---

[2.4.0]: https://github.com/TFD-42/Wild_Root_Prompt/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/TFD-42/Wild_Root_Prompt/compare/v2.2.0...v2.3.0
