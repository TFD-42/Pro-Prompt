# Contributing to Wild_Root_Prompt

Thanks for taking the time to contribute.

## Before you start

- **For bug reports or feature requests**, open an issue first so we can discuss the scope before you write code.
- **For small fixes** (typos, doc corrections, trivial one-liners), a PR directly is fine.
- **For new prompt engineering techniques**, include a bibliographic reference (paper title + URL). Unverified techniques are not added to the catalogue.

## Setup

```bash
git clone https://github.com/TFD-42/Wild_Root_Prompt.git
cd Wild_Root_Prompt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Before submitting a PR

1. **Run the test suite** — all 89 tests must pass:
   ```bash
   python3 -m unittest discover -s tests -v
   ```
2. **Run the privacy scan** — no PII or credentials may be introduced:
   ```bash
   python3 tools/privacy_scan.py
   ```
3. **Type-check** if you touched `prompt_expert_enhance.py` or `web_server.py`:
   ```bash
   python3 -m py_compile prompt_expert_enhance.py web_server.py
   ```

## Code style

- No comments unless the *why* is non-obvious (a hidden constraint, a workaround for a specific bug).
- No blank lines inside methods.
- No single-letter lambda names — use descriptive names.
- No `try/except` blocks — use `tryCatch()` from the project's utility helpers.

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities privately.
