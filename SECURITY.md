# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x (latest) | ✅ |
| < 2.0 | ❌ |

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Use [GitHub's private vulnerability reporting](https://github.com/TFD-42/Wild_Root_Prompt/security/advisories/new) to submit a report confidentially. You will receive a response within 7 days.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

## Security Design

Wild Root Prompt is a local-first CLI tool. Key properties:

- **No telemetry** — no data is sent to any remote server by default.
- **No credentials stored** — API keys are read from environment variables or a local config file that is excluded from version control.
- **SSRF protection** — the web server only accepts connections from `localhost`; all outbound fetch calls validate the target URL against a block-list before connecting.
- **Prompt injection detection** — user input is scanned for known injection patterns before being forwarded to the model.
- **PII anonymization** — the `--anonymize` flag strips emails, IPs, phone numbers, and other personal data from prompts before they leave the local machine.

See [`ZERO_TRUST_SECURITY.md`](ZERO_TRUST_SECURITY.md) for the full threat model.
