# Wild_Root_Prompt — Local LLM Prompt Enhancement Tool v2.4

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/TFD-42/Wild_Root_Prompt/actions/workflows/ci.yml/badge.svg)](https://github.com/TFD-42/Wild_Root_Prompt/actions/workflows/ci.yml)
[![CodeQL](https://github.com/TFD-42/Wild_Root_Prompt/actions/workflows/codeql.yml/badge.svg)](https://github.com/TFD-42/Wild_Root_Prompt/actions/workflows/codeql.yml)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-green.svg)](https://ollama.com)
[![LLM](https://img.shields.io/badge/LLM-Local%20AI-blueviolet.svg)](https://en.wikipedia.org/wiki/Large_language_model)
[![Prompt Engineering](https://img.shields.io/badge/Prompt%20Engineering-173%20Techniques-orange.svg)](#prompt-engineering-techniques)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows%20%7C%20Android-informational.svg)](#install)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/Ellana42/wild-root-prompt)
[![GitHub stars](https://img.shields.io/github/stars/TFD-42/Wild_Root_Prompt?style=social)](https://github.com/TFD-42/Wild_Root_Prompt/stargazers)

**Transform any raw prompt into expert-level output using local LLMs via Ollama — no cloud API keys, no data sent to external servers.**

## Platform Overview


![Capture d’écran 2026-08-10 à 18 47 52](https://github.com/user-attachments/assets/49a3aeed-b33e-41e0-bc8e-a1a3afbd2de8)


Wild_Root_Prompt enhances any prompt before it reaches your LLM. A **pre-processor** first restructures and clarifies your raw input, then applies **173 prompt engineering techniques** across 15 categories (Chain-of-Thought, Tree-of-Thought, ReAct, MECE, red teaming, and more) to generate exhaustive, high-quality outputs. Novice users get a **local web UI** (one click); power users get a full **CLI** with parallel dual-model generation, split-screen streaming, and expert synthesis.

**Keywords:** Prompt engineering · Local LLM · Ollama · Open-source AI · Prompt optimization · LLM enhancement · Chain-of-Thought · Tree-of-Thought · ReAct · MECE · Prompt techniques · AI agents · Llama · Qwen · Dolphin · Privacy-first · Offline AI

## Table of Contents
- [Quick Start](#quick-start) — Install & launch in 2 minutes
- [Web UI](#launch--web-ui-novice-friendly) — One-click browser interface
- [CLI](#launch--cli-power-users) — Power user terminal mode  
- [Techniques](#prompt-engineering-techniques) — 173 techniques across 15 categories
- [Standalone App](#build-a-standalone-compiled-app-single-icon-no-terminal) — Compiled single-icon binary
- [Features](#key-features) — Complete feature matrix

---

## Why Wild_Root_Prompt?

**Unlike generic prompt builders**, Wild_Root_Prompt implements **academic prompt engineering theory** (Chain-of-Thought, ReAct, MECE, Constitutional AI, and 15+ other frameworks) across **173 distinct techniques**. Every technique is researched, categorized, and applied *before* your LLM sees the input — so even simple models produce expert-tier outputs.

**Unlike cloud-first tools**, Wild_Root_Prompt runs **100% locally**:
- ✅ No API calls to external servers
- ✅ Your data stays on your machine
- ✅ No subscription fees — MIT licensed
- ✅ Works offline (except optional web enrichment)
- ✅ Powered by free, open-source [Ollama](https://ollama.com)

**Unlike single-model tools**, Wild_Root_Prompt can:
- Run **two LLMs in parallel** with split-screen streaming
- **Synthesize** both outputs into a unified superior document
- **Combine** multiple generations end-to-end via full pipeline
- **Merge** insights from diverse reasoning styles (systematic vs. creative)

> **Try it now** — a live demo runs at [huggingface.co/spaces/Ellana42/wild-root-prompt](https://huggingface.co/spaces/Ellana42/wild-root-prompt) with no install required.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Pre-processor** | Ollama restructures and completes your raw input before the main pipeline (toggleable) |
| **Local web UI** | One-click browser interface at `http://localhost:7860` — novice-friendly, no CLI needed |
| **Interactive CLI** | Numbered menu, no flags to memorize |
| **Two output modes** | **Quick**: single enhanced prompt ready to paste · **Full**: exhaustive 12-section manifest |
| **173 prompt engineering techniques** | Organized in 15 categories with anti-patterns, quick-reference matrix, and **25-topic auto-index** |
| **Smart technique auto-selection** | `--recommend-techniques` matches your task to one of 25 topics (all 173 techniques reachable, bilingual EN/FR) |
| **60 /slash metacommands** | Inline modifiers: persona, format, depth, reasoning, quality, context |
| **Single model generation** | Real-time token streaming in terminal or browser |
| **Parallel dual-model generation** | Split-screen display with two columns, live tokens |
| **Expert synthesis** | Merges two outputs into a unified, superior document with streaming |
| **Full pipeline** | Parallel generation + synthesis in one command |
| **Web enrichment** | Automatic DuckDuckGo search for real-world context (toggleable, SSRF-protected) |
| **Model auto-detection** | Lists locally installed Ollama models with numbered picker at each run |
| **First-run guidance** | If no models installed, offers guided pull with RAM requirements |
| **One-click launchers** | `WildRoot.command` (macOS) · `WildRoot.bat` (Windows) · `WildRoot` (Linux) |
| **Cross-platform install** | `install.sh` (macOS/Linux) · `install.ps1` (Windows) · `install_termux.sh` (Android) |
| **Zero-trust security** | Input sanitization, SSRF block, no PII in logs, random session key |
| **Session memory** | Tracks past runs for cross-session coherence |
| **Persistent settings** | Models, techniques, temperature, output mode, pre-processor saved locally |

---

## Quick Start

### Install

**No terminal at all (double-click):**

| Platform | Just double-click |
|----------|--------|
| **macOS** | `Install WildRoot.command` |
| **Windows** | `Install WildRoot.bat` |

Both detect what's already installed and run the full setup (Ollama, Python 3, virtual environment, dependencies) automatically, ending with a one-click day-to-day launcher (see below). The Windows wrapper runs PowerShell with `-ExecutionPolicy Bypass` scoped to that single launch only — it does not change your system's script-execution policy.

**One-line terminal, by OS:**

macOS / Linux:

```bash
git clone https://github.com/TFD-42/Wild_Root_Prompt.git && cd Wild_Root_Prompt && chmod +x install.sh && ./install.sh
```

Windows (PowerShell):

```powershell
git clone https://github.com/TFD-42/Wild_Root_Prompt.git; cd Wild_Root_Prompt; powershell -ExecutionPolicy Bypass -File install.ps1
```

Android / Termux:

```bash
git clone https://github.com/TFD-42/Wild_Root_Prompt.git && cd Wild_Root_Prompt && chmod +x install_termux.sh && ./install_termux.sh
```

The installer handles everything: Ollama, Python 3, virtual environment, and dependencies. It also creates a one-click launcher.

### Launch — Web UI (novice-friendly)

| Platform | Action |
|----------|--------|
| **macOS** | Double-click `WildRoot.command` in Finder |
| **Windows** | Double-click `WildRoot.bat` |
| **Linux** | Run `./WildRoot` in terminal |
| **Any** | `python3 prompt_expert_enhance.py --web` |

Opens `http://localhost:7860` in your browser automatically. On macOS, if
Chrome, Firefox, Brave, Edge, Arc, or Opera is installed, Wild_Root_Prompt opens
that instead of Safari — see **Troubleshooting** below for why.

### Launch — CLI (power users)

```bash
source .venv/bin/activate    # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows

python3 prompt_expert_enhance.py
```

Launches the interactive numbered menu. No arguments needed.

### Build a standalone compiled app (single icon, no terminal)

After running the installer once (so dependencies exist in `.venv`), you can
compile Wild_Root_Prompt into one native app with its own icon:

```bash
source .venv/bin/activate           # Linux/macOS
# .\.venv\Scripts\Activate.ps1      # Windows
pip install -r requirements-build.txt
python3 build_app.py
```

Produces:

| Platform | Output |
|----------|--------|
| **macOS** | `dist/Wild_Root_Prompt.app` — drag to /Applications, double-click to launch |
| **Windows** | `dist/Wild_Root_Prompt.exe` — double-click to launch |
| **Linux** | `dist/Wild_Root_Prompt` — single binary; run it, or wire up a `.desktop` file for a menu icon |

The compiled app starts Ollama if needed and opens the web UI in your
browser — it needs no Python install or virtual environment at runtime.
Its data (settings, memory, cache, outputs) lives in a standard per-OS user
data folder (e.g. `~/Library/Application Support/Wild_Root_Prompt` on macOS)
rather than next to the app bundle, since app bundles are read-only.

### CLI Mode

Pass arguments directly for scripting and automation:

```bash
# Single model generation
python3 prompt_expert_enhance.py generate "Design a REST API" --model llama3:latest

# Parallel dual-model generation
python3 prompt_expert_enhance.py parallel "Design a REST API" --model-a llama3 --model-b qwen2.5:7b

# Full pipeline (parallel + synthesis)
python3 prompt_expert_enhance.py full "Design a REST API" --techniques "1-30"

# List all 173 techniques grouped by category
python3 prompt_expert_enhance.py generate x --list-techniques

# Start from a predefined template instead of writing a task from scratch
python3 prompt_expert_enhance.py templates list
python3 prompt_expert_enhance.py generate "distributed systems" --template learning_plan
```

See [`examples/`](examples/) for runnable demos of these commands plus newer
features (technique bundles, draft mode, offline mode, PII redaction, result
caching, deep research, alternate backends) and the REST API.

---

## Interactive Menu

```
==============================================================
   WILD_ROOT_PROMPT  —  Expert Prompt Enhancement Tool  v2.2
==============================================================
   Model A     : llama3:latest
   Model B     : qwen2.5:7b
   Synthesis   : qwen2.5:7b
   Temperature : 0.3
   Techniques  : 15 active / 173 available
   Internet    : ON   Web enrichment : ON   Streaming : ON
--------------------------------------------------------------

  1.  Single generation         (1 model, streaming)
  2.  Parallel generation       (2 models, split screen)
  3.  Full pipeline             (parallel + synthesis)
  4.  Synthesize 2 files

  5.  Configure models
  6.  Configure techniques
  7.  Browse available techniques
  8.  Advanced settings          (temperature, timeout, url, web, stream)

  9.  View memory
  10. Clear memory

  0.  Quit
```

### Model Picker

When selecting a model, Wild_Root_Prompt lists all locally installed models:

```
  -- Generation model --
  Locally installed models:
      1. llama3:latest                   4.7GB  2026-05-20  <-- current
      2. qwen2.5:7b                      4.4GB  2026-05-18
      3. dolphin3:latest                 4.6GB  2026-05-07
      4. [Enter a name manually / pull a new model]

  Choice [llama3:latest] >
```

Type a number to select, a model name to pull, or Enter to keep the current one.

---

## Prompt Engineering Techniques

Wild_Root_Prompt ships with **173 techniques** across **15 categories** in `prompt_expert_methodology.json`, plus **8 anti-patterns** and a **quick-reference matrix** for task-based technique selection.

**Research-backed methods included:**
- **Chain-of-Thought (CoT)** — Reasoning through intermediate steps
- **Tree-of-Thought (ToT)** — Exploring multiple reasoning branches
- **ReAct** — Reasoning + Acting iteratively  
- **MECE** — Mutually Exclusive, Collectively Exhaustive decomposition
- **Constitutional AI** — Self-correcting with predefined principles
- **Red Teaming** — Adversarial stress-testing  
- **Few-Shot Learning** — In-context example priming
- **Automatic Prompt Optimization** — Self-refining techniques

### Categories

| # | Category | Techniques | Examples |
|---|----------|-----------|----------|
| 1 | **Framing** | 6 | Zero-shot, few-shot, many-shot, negative-shot, contrastive prompting |
| 2 | **Directed reasoning** | 10 | Chain-of-Thought (CoT), Tree-of-Thought (ToT), Graph-of-Thought (GoT), ReAct, Program-of-Thought (PoT), Skeleton-of-Thought, least-to-most |
| 3 | **Depth forcing** | 11 | Output length specification, recursive deepening, exhaustive enumeration, anti-lazy preamble |
| 4 | **Constraint-based** | 21 | Format forcing, vocabulary constraint, register constraint, perspective constraint, inverse prompting, rubber duck, constraint stacking |
| 5 | **Multi-perspective** | 9 | Multi-viewpoint analysis, counter-arguments, audience layering, cross-disciplinary |
| 6 | **Meta / recursive** | 15 | Self-critique, self-refine, self-ask, meta-prompting, constitutional prompting, recursive summarization |
| 7 | **Structural** | 13 | Instruction decomposition, strong delimiters, priority stacking, prompt chaining, conditional prompting |
| 8 | **Persona & role** | 9 | Expert persona, multi-persona debate, naive persona, devil's advocate, future historian |
| 9 | **Emergent** | 14 | Emotional priming, anchoring, semantic pressure, cognitive load offloading, counterfactual, steelmanning, pre-mortem |
| 10 | **Cognitive decomposition** | 9 | MECE, first principles, five whys, abstraction ladder, dual process, Socratic decomposition, ontology extraction |
| 11 | **Adversarial** | 9 | Red teaming, stress testing, bias hunting, assumption mapping |
| 12 | **Hybrid multi-pass** | 8 | Generate-then-filter, breadth-first/depth-first, adversarial refinement loop, perspective rotation, zoom protocol |
| 13 | **Evidence & justification** | 16 | Citation thresholds, uncertainty quantification, historical grounding, tiered evidence |
| 14 | **Creative & narrative** | 8 | Analogy generation, narrative embedding, timeline construction, forced self-interruption |
| 15 | **Rarely explored** | 15 | Formal logic coherence, invariant detection, test generation, tacit knowledge elicitation, weak signal detection, second-order effects, heuristic generation |

### Default Set (15 techniques)

Step-by-step reasoning, forced reframing, anti-lazy preamble, recursive deepening, counter-arguments, example-driven expansion, outline-then-expand, definition-first, first-principles, no-word-limit, Tree-of-Thought, constraint stacking, constitutional prompting, MECE decomposition, assumption mapping.

### Selecting Techniques

```bash
--techniques "1,5,8,10,25"           # Specific IDs
--techniques "1-30"                   # Range
--techniques "1-173"                  # All 173 (explicit full range)
--techniques "bundle:code-generation" # Named task bundle
--techniques "bundle:3"               # Bundle by number
--recommend-techniques                # Auto-select based on your task (recommended)
```

**Auto-selection (`--recommend-techniques`)** matches your task text against a 25-topic index covering all 173 techniques, then loads the 6 most coherent techniques for that topic. Topics include: `debugging-troubleshooting`, `creative-exploration`, `security-audit`, `teach-explain`, `structured-output-api`, `critical-decision`, and 19 more — in English and French.

In the interactive menu, use option `6` to configure or option `7` to browse (grouped by category with anti-patterns and quick reference).

---

## Web Enrichment

When internet is available, Wild_Root_Prompt automatically searches DuckDuckGo for the task description, fetches top results, and injects relevant context into the prompt. This runs before generation and adds real-world grounding without any API keys.

Toggle via the advanced settings menu (option `8`) or `--no-web` flag.

---

## Output Structure

Generated manifests follow a **12-section structure**:

1. Title & Executive Summary
2. Final Objective & Success Definition
3. Execution Context & Prerequisites
4. Ambiguity Zones to Resolve
5. Step Decomposition (Detailed Pipeline)
6. Control Loops & Scoring
7. Persistent Artifacts to Maintain
8. Constraints & Guardrails
9. Error Handling Strategy
10. Final Deliverable & Output Format
11. Reproducibility Checklist
12. Notes for the Target Agent

---

## Project Structure

```
Wild_Root_Prompt/
  prompt_expert_enhance.py        # Main application — CLI + pre-processor + topic index
  web_server.py                   # Local web UI server (Flask, SSE streaming)
  prompt_expert_methodology.json  # 173 techniques, 15 categories, 25-topic index, bundles
  prompt_templates.json           # 10 starter task templates
  requirements.txt                # Full deps (requests, flask, cryptography)
  requirements-light.txt          # Minimal headless CLI deps (requests only)
  requirements-build.txt          # Build deps (pyinstaller)
  install.sh                      # Installer — macOS / Linux
  install.ps1                     # Installer — Windows
  install_termux.sh               # Installer — Android / Termux
  WildRoot.command                # macOS double-click launcher (web UI)
  WildRoot.bat                    # Windows launcher (created by install.ps1)
  WildRoot                        # Linux/macOS terminal launcher (created by install.sh)
  build_app.py                    # Compile to standalone single-icon app (PyInstaller)
  tests/
    test_core.py                  # 89 unit tests — stdlib only, no network, <0.1s
  tools/
    privacy_scan.py               # PII scanner — CI gate, run before every push
    batch_test.py                 # Batch evaluation helper
  .github/workflows/
    ci.yml                        # CI: Python 3.8-3.13, tests + PII scan on every push/PR
    release.yml                   # Release: compile macOS/Linux/Windows on tag push
  .gitignore
  README.md
  LICENSE
  memory/                         # Session history (gitignored)
  outputs/                        # Generated outputs (gitignored)
```

## Troubleshooting

**macOS says "Apple could not verify 'Wild_Root_Prompt.app' is free of malware" and blocks it from opening.**
This is macOS Gatekeeper — it flags any app downloaded from outside the App Store that isn't signed with a paid ($99/year) Apple Developer ID and notarized by Apple. Wild_Root_Prompt is free and open-source, so it isn't notarized; the app itself is safe (the code is public in this repo — build it yourself with `build_app.py` if you want to verify). Two ways to open it anyway:
- **Finder**: right-click (or Control-click) `Wild_Root_Prompt.app` → **Open** → confirm **Open** in the dialog. This only needs to be done once.
- **Terminal**: `xattr -cr /path/to/Wild_Root_Prompt.app` (or `xattr -cr /Applications/Wild_Root_Prompt.app` if you moved it there), then double-click normally.

**Safari shows "Safari ne parvient pas à ouvrir la page" / a WebKitErrorDomain:305 error, or the window opens blank.**
This happens when Safari's **HTTPS-Only Mode** is set to apply to *all* websites (Safari → Settings → Advanced). In that mode Safari hard-blocks any plain `http://` navigation — including `localhost` and `127.0.0.1` — with no in-page bypass, since Wild_Root_Prompt's local server intentionally has no TLS certificate (it never leaves your machine). Wild_Root_Prompt already prefers Chrome/Firefox/Brave/Edge/Arc/Opera over Safari on macOS when one is installed, since none of them impose this restriction on loopback addresses. If Safari is your only browser, either:
- Safari → Settings → Advanced → turn off "Use HTTPS-Only for all websites", or
- Install any other browser — Wild_Root_Prompt will use it automatically next launch.

## Requirements

- **[Ollama](https://ollama.com)** — installed automatically by the installer (or an OpenAI-compatible backend — see `examples/alternate_backend.md`)
- **Python 3.8+** — installed automatically by the installer
- **requests** — installed via `pip install -r requirements.txt`
- At least one Ollama model pulled (the launcher handles this interactively)

**Headless / CI / server use (no web UI)**: `prompt_expert_enhance.py` never
imports `flask`/`web_server.py` unless you actually run the `web` subcommand,
so the CLI (`generate`/`parallel`/`full`/`synthesis`/`memory`) works with just
`requests` installed. For that minimal footprint:

```bash
pip install -r requirements-light.txt
```

`flask` (web UI) and `cryptography` (memory encryption) stay fully optional —
each feature detects its own missing dependency and either falls back
gracefully (memory encryption) or exits with a clear one-line message
(the `web` command) instead of crashing.

## Configuration

All settings persist in `settings.json` (gitignored, local to each user):

| Setting | Default | Description |
|---------|---------|-------------|
| `model_a` | `llama3:latest` | Primary generation model |
| `model_b` | `qwen2.5:7b` | Secondary model for parallel runs |
| `synthesis_model` | `qwen2.5:7b` | Model used for expert synthesis |
| `temperature` | `0.3` | LLM temperature (0.0–1.0) |
| `timeout` | `600` | Seconds per Ollama call |
| `techniques` | `[1,5,8,10,12,14,18,25,40,47,108,121,125,147,153]` | Active technique IDs (from 173 available) |
| `use_web` | `true` | Enable web enrichment (DuckDuckGo, SSRF-protected) |
| `stream` | `true` | Enable real-time streaming |
| `output_mode` | `"full"` | `"quick"` = enhanced prompt · `"full"` = 12-section manifest |
| `use_pre_processor` | `true` | Enable pre-processor step (Ollama restructures raw input) |
| `pre_processor_model` | `""` | Model for pre-processor (`""` = use Model A) |

## How It Works

1. **You describe a task** — in natural language, as simple or complex as you want
2. **Wild_Root_Prompt builds an expert prompt** — injecting selected techniques, web context, and session memory
3. **Local LLM generates a manifest** — a structured 12-section document describing the task with methodological precision
4. **Optionally, two models generate in parallel** — and a synthesis pass merges them into a superior unified document
5. **The output is a reproducible instruction set** — ready to be executed by any LLM agent (Claude, GPT, Gemini, Llama, Mistral, Qwen)

## Use Cases

- **Prompt engineering** — Generate expert-level prompts for any LLM task
- **Task specification** — Create detailed, unambiguous task descriptions for AI agents
- **Knowledge extraction** — Force exhaustive exploration of any topic
- **Comparative analysis** — Run two models in parallel and synthesize the best of both
- **Reproducible AI workflows** — Manifests can be reused across models and platforms
- **Learning prompt engineering** — Browse 173 techniques with descriptions and categories

## Roadmap & Known Limitations

**Current limitations:**
- Web enrichment requires an internet connection (all other features are fully offline).
- `--techniques all` is intentionally refused — stacking all 173 techniques degrades output quality. Use `--recommend-techniques`, a named bundle, or an explicit comma-separated list.
- The compiled standalone app embeds the methodology JSON at build time; adding new techniques requires rebuilding the binary.

**Planned:**
- PyPI packaging (`pip install wildroot`) for easier distribution.
- Second-pass LLM technique selection (opt-in): after the first-pass topic index selects a core set, an optional second pass would surface candidates from the `secondary` pool.
- CHANGELOG retroactive entries for v2.0–v2.2.

## License

[MIT](LICENSE)

## Related Resources

- **[Ollama](https://ollama.com)** — Open-source large language models
- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)** — Wei et al. (2022)
- **[Tree-of-Thoughts](https://arxiv.org/abs/2305.10601)** — Yao et al. (2023)
- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)** — Yao et al. (2022)
- **[MECE Principle](https://en.wikipedia.org/wiki/MECE_principle)** — Structured decomposition
- **[Constitutional AI](https://arxiv.org/abs/2212.04092)** — Self-aligning language models
- **[Prompt Engineering Guide](https://www.promptingguide.ai/)** — Community resource for LLM prompting

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-enhancement`)
3. Run the test suite and privacy scan before committing:
   ```bash
   python3 -m unittest discover -s tests -v
   python3 tools/privacy_scan.py
   ```
4. Push to your fork and open a pull request — the CI workflow will run automatically on Python 3.8–3.13

**Keep these files out of commits** (already in `.gitignore`):
- `settings.json` — Local user settings
- `outputs/` — Generated outputs
- `memory/sessions.json` — Session history
- `.env` — Environment variables

## Acknowledgments

Wild_Root_Prompt builds on decades of prompt engineering research from academic institutions and AI labs worldwide. Core theoretical foundations:
- Stanford, MIT, CMU, UC Berkeley research on language models and reasoning
- OpenAI, Anthropic, DeepSeek, and open-source communities
- Original technique papers and methodologies cited in `prompt_expert_methodology.json`

## License

[MIT](LICENSE) — Use freely in personal and commercial projects.

---

**Made with ❤️ for the local AI community. No cloud dependencies. No data collection. Just prompt excellence.**
