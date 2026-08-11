#!/usr/bin/env python3
"""
Wild_Root_Prompt – Expert Prompt Enhancement Tool powered by Ollama.

Version: 2.3.0

Two output modes:
    quick  — single enhanced prompt, ready to paste into any LLM
    full   — exhaustive 12-section instruction manifest (expert use)

Usage:
    python prompt_expert_enhance.py                    (interactive menu)
    python prompt_expert_enhance.py generate <task> [options]
    python prompt_expert_enhance.py parallel <task> [options]
    python prompt_expert_enhance.py synthesis <task> <file_a> <file_b> [options]
    python prompt_expert_enhance.py full <task> [options]
    python prompt_expert_enhance.py memory view
    python prompt_expert_enhance.py memory clear

Metacommands — prefix your task with /slash modifiers:
    /expert /humain /enfant /sceptique /mentor /cynique /serieux /humour
    /tableau /json /markdown /points /checklist
    /resume /concis /detaille /limite:N
    /etapes /exemple /analogie /pourcontre /debat /reverse /iterer /ameliorer
    /precision /hypotheses /sources /risques /decision /verification /audit
    /raisonnement /minimal /silence /urgent /questionner /historique /futuriste
    /niveau:debutant|intermediaire|expert   /confiance   /comparatif

Commands:
    generate       Single model manifest generation (multi‑topic optional).
    parallel       Parallel generation from two models (MODEL_A, MODEL_B).
    synthesis      Run the meta‑architect synthesis on two existing manifests.
    full           Full pipeline: parallel + synthesis.
    memory         Manage session memory (view / clear).

For detailed help on a command: manifest <command> --help
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from html.parser import HTMLParser
from urllib.parse import quote_plus, urlparse

import requests

# ----------------------------------------------------------------------
# Configuration – can be overridden via environment or command line
# ----------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
BASE_DIR = Path(__file__).resolve().parent

# Cloud deploy: set HF_TOKEN env var to switch to HF Inference API backend.
# All Ollama checks are skipped automatically when this is set.
_HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
_CLOUD_MODE: bool = bool(_HF_TOKEN) or bool(os.environ.get("RENDER"))
_HF_INFERENCE_URL = "https://api-inference.huggingface.co/v1/chat/completions"
_HF_DEFAULT_MODEL = os.environ.get("HF_DEFAULT_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def _is_frozen() -> bool:
    """True when running inside a PyInstaller-compiled app, not from source."""
    return bool(getattr(sys, "frozen", False))


def _bundled_resource_dir() -> Path:
    """Read-only data (methodology/templates JSON) bundled inside a compiled
    app, or the repo root when running from source — unchanged either way
    for normal dev/CLI use."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", BASE_DIR))
    return BASE_DIR


def _user_data_dir() -> Path:
    """Writable directory for memory/outputs/cache/settings. A compiled
    app's own bundle is read-only (and on macOS, code-signed), so writes
    there would fail or invalidate the signature — redirect to a proper
    per-OS user data directory instead. Running from source keeps writing
    next to the script, exactly as before."""
    if not _is_frozen():
        return BASE_DIR
    if sys.platform == "darwin":
        data_dir = Path.home() / "Library" / "Application Support" / "Wild_Root_Prompt"
    elif sys.platform == "win32":
        data_dir = Path(os.environ.get("APPDATA", str(Path.home()))) / "Wild_Root_Prompt"
    else:
        data_dir = Path.home() / ".wild_root_prompt"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


_RESOURCE_DIR = _bundled_resource_dir()
_DATA_DIR = _user_data_dir()

MEMORY_DIR = _DATA_DIR / "memory"
OUTPUT_DIR = _DATA_DIR / "outputs"
CACHE_DIR = _DATA_DIR / "cache"
MEMORY_FILE = MEMORY_DIR / "sessions.json"
METHODOLOGY_FILE = _RESOURCE_DIR / "prompt_expert_methodology.json"
DEFAULT_MODEL_A = "llama3:latest"
DEFAULT_MODEL_B = "qwen2.5:7b"
DEFAULT_SYNTH_MODEL = "qwen2.5:7b"
DEFAULT_TEMPERATURE = 0.3

# Generic, non-identifying HTTP User-Agent for web enrichment requests.
# Change this string if you want to identify the tool differently,
# but never include real OS, device, or browser version info here.
HTTP_USER_AGENT = "Mozilla/5.0 (compatible; ManifestGen/2.1; +https://github.com/TFD-42/Wild_Root_Prompt)"
DEFAULT_TIMEOUT = 600  # seconds for a single Ollama call
SYNTH_TIMEOUT = 1200   # longer timeout for synthesis

# Default techniques applied when none are specified (configurable via CLI or settings)
DEFAULT_TECHNIQUES = [1, 5, 8, 10, 12, 14, 18, 25, 40, 47, 108, 121, 125, 147, 153]

# Pre-processor constants
PRE_PROCESSOR_TIMEOUT = 30
PRE_PROCESSOR_MAX_TOKENS = 600          # floor/default, used for short inputs
PRE_PROCESSOR_MAX_TOKENS_CEILING = 2400  # cap so very long inputs can't cause runaway generation time
PRE_PROCESSOR_PROMPT = """You are a prompt reconstruction specialist. Your job is to take a raw, imperfect user input and return a single clean, complete, well-structured prompt that is STRICTLY STRONGER than the raw input — never a mere cleanup pass.

⚠️ LANGUAGE LOCK — THIS IS NON-NEGOTIABLE:
The input language has been detected as: {DETECTED_LANG}
You MUST write the reconstructed prompt in {DETECTED_LANG} ONLY.
DO NOT translate. DO NOT switch languages. If you are unsure, keep every word in {DETECTED_LANG}.

ABSOLUTE RULES:
- Output ONLY the reconstructed prompt. Nothing else.
- No preamble. No explanation. No labels. No headers. No meta-commentary.
- Do NOT answer the question. Do NOT execute the task. Reconstruct the prompt ONLY.
- Preserve the user's intent exactly — do not change the domain or goal.
- LANGUAGE: output MUST be in {DETECTED_LANG}. Never translate. Never switch.
- Preserve role/persona directives ("Tu es...", "Act as...") exactly.

MANDATORY QUALITY BAR — THE OUTPUT MUST BE STRONGER THAN THE INPUT:
- The reconstructed prompt must never be a trivial rewording, a passthrough, or equal-or-weaker in specificity, clarity, or actionability than the raw input. If you cannot make it stronger, you have not tried hard enough — revise before outputting.
- Every one of the "WHAT TO FIX" gaps below that applies must actually be filled in, not just flagged. Silence about a gap is not acceptable if the gap is fixable from context.
- Add explicit success criteria, scope boundaries, or a deliverable format whenever the raw input leaves them implicit — even for long/detailed inputs.
- Before finalizing, silently verify: "Is this reconstruction measurably more specific and actionable than what the user typed?" If the answer is no, strengthen it further.

HARD, CHECKABLE REQUIREMENTS (do not skip these — they are verified mechanically):
- MINIMUM GROWTH: the reconstructed prompt's word count must be at least 3x the raw input's word count, with a floor of 80 words. A reconstruction close in length to the raw input is a FAILED reconstruction.
- MINIMUM STEP LIST: if the raw input asks for a task, deliverable, script, pipeline, system, or anything to be built/created/implemented/designed/written, the reconstructed prompt MUST include an explicit numbered list of at least 3 concrete steps or requirements the executing agent must follow (e.g. inputs/data handling, each core method or component to implement, expected output/validation). Do not merely mention the topic — decompose it.
- Each listed step must be concrete and specific to the actual domain/methods named in the raw input (not generic placeholders like "step 1: understand the task").

WHAT TO FIX:
- Typos, missing words, grammatical fragments → correct silently
- Vague references ("ce truc", "ça", "it") → make explicit from context
- Missing output format → infer and state it clearly
- Missing audience/scope → infer and state it
- Missing success criteria or constraints → add explicit ones inferred from context
- If input < 20 words: expand to a complete actionable prompt
- If input 20-100 words: clean, structure, and sharpen — add any missing specificity even if it grows the prompt
- If input > 100 words: restructure and tighten every ambiguous or underspecified clause; do not pad, but do not leave gaps unfixed for the sake of preserving length

WHAT TO PRESERVE ALWAYS:
- Tone (formal, casual, authoritative, creative)
- Technical level and domain
- Any explicit constraints the user stated
- /slash metacommands at the start
- Language: {DETECTED_LANG}
- Code blocks, code fences (```...```), and concrete examples the user provided — reproduce them verbatim, character-for-character, inside the same fence markers. Never paraphrase, summarize, "clean up", or truncate code/example content. You may add surrounding structure (e.g. "Example:" / "Reference code:" labels) but the fenced content itself is untouchable.
{KEYWORDS_BLOCK}{USER_LEVEL_BLOCK}
Raw user input:
<<<INPUT
{RAW_INPUT}
INPUT>>>

Reconstructed prompt (in {DETECTED_LANG}), strictly stronger than the raw input above:"""

# Create directories at import time
MEMORY_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Load methodologies from JSON
TECHNIQUES_DB: Dict[int, Dict[str, str]] = {}
CATEGORIES_DB: List[Dict] = []
ANTI_PATTERNS: List[Dict] = []
QUICK_REFERENCE: Dict[str, List[int]] = {}
TOPIC_INDEX: List[Dict] = []

def load_methodologies():
    global TECHNIQUES_DB, CATEGORIES_DB, ANTI_PATTERNS, QUICK_REFERENCE, TOPIC_INDEX
    if not METHODOLOGY_FILE.exists():
        return
    try:
        raw = METHODOLOGY_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        if "categories" in data:
            CATEGORIES_DB = data["categories"]
            for cat in CATEGORIES_DB:
                for tech in cat.get("techniques", []):
                    TECHNIQUES_DB[tech["id"]] = {
                        "title": tech["title"],
                        "description": tech["description"],
                        "category": cat["name"],
                    }
            ANTI_PATTERNS = data.get("anti_patterns", [])
            QUICK_REFERENCE = data.get("quick_reference", {}).get("mappings", {})
            TOPIC_INDEX = data.get("topic_index", [])
        else:
            for line in raw.strip().split("\n"):
                if line and line[0].isdigit():
                    match = re.match(r"(\d+)\.\s+\*\*(.+?)\*\*\s+–\s+(.+)", line)
                    if match:
                        num = int(match.group(1))
                        TECHNIQUES_DB[num] = {"title": match.group(2), "description": match.group(3)}
    except Exception as e:
        logger.warning(f"Could not load methodologies: {e}")

load_methodologies()

PROMPT_TEMPLATES_FILE = _RESOURCE_DIR / "prompt_templates.json"
PROMPT_TEMPLATES: Dict[str, Dict] = {}  # id -> {title, category, task, suggested_bundle}


def load_prompt_templates() -> None:
    global PROMPT_TEMPLATES
    if not PROMPT_TEMPLATES_FILE.exists():
        return
    try:
        data = json.loads(PROMPT_TEMPLATES_FILE.read_text(encoding="utf-8"))
        PROMPT_TEMPLATES = {t["id"]: t for t in data.get("templates", []) if t.get("id")}
    except Exception as e:
        logger.warning(f"Could not load prompt templates: {e}")


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("manifest_gen")

load_prompt_templates()


# ----------------------------------------------------------------------
# Ollama bootstrap — detect, install, list models, pull
# ----------------------------------------------------------------------
OLLAMA_API_BASE = OLLAMA_URL.rsplit("/api/", 1)[0]  # http://localhost:11434

# ----------------------------------------------------------------------
# Backend abstraction — Ollama (native) or any OpenAI-compatible server
# (LM Studio, GPT4All's server mode, text-generation-webui's OpenAI
# extension all speak the same /v1/completions or /v1/chat/completions
# wire format, so one adapter covers all three).
# ----------------------------------------------------------------------
_BACKEND_TYPE = "ollama"  # "ollama" | "openai_compatible"
_BACKEND_API_BASE = OLLAMA_API_BASE


def set_backend_type(backend_type: str) -> None:
    global _BACKEND_TYPE
    _BACKEND_TYPE = backend_type if backend_type in ("ollama", "openai_compatible") else "ollama"


def _derive_api_base(url: str) -> str:
    """Strip the endpoint-specific suffix from a full API URL to get the
    server's base — e.g. 'http://host:port/api/generate' -> 'http://host:port',
    'http://host:port/v1/chat/completions' -> 'http://host:port'."""
    for suffix in ("/api/generate", "/v1/chat/completions", "/v1/completions"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else url


def set_backend_api_base(url: str) -> None:
    global _BACKEND_API_BASE
    _BACKEND_API_BASE = _derive_api_base(url) if url else OLLAMA_API_BASE


def _is_chat_endpoint(url: str) -> bool:
    return "/chat/completions" in url


def _openai_compatible_payload(model: str, prompt: str, temperature: float, num_predict: int, url: str, stream: bool) -> dict:
    max_tokens = num_predict if num_predict and num_predict > 0 else 1024
    if _is_chat_endpoint(url):
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
    return {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def _openai_compatible_extract_text(choice: dict, is_chat: bool) -> str:
    if is_chat:
        return choice.get("message", {}).get("content", "") or choice.get("delta", {}).get("content", "")
    return choice.get("text", "")


def detect_os() -> str:
    s = platform.system().lower()
    if s == "darwin":
        return "mac"
    if s == "windows":
        return "windows"
    return "linux"


def is_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def is_ollama_running() -> bool:
    if _CLOUD_MODE:
        return True
    try:
        requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def install_ollama_interactive() -> bool:
    os_type = detect_os()
    print()
    print("  [!] Ollama is not installed on this machine.")
    print(f"  Detected OS: {os_type}")
    print()
    if os_type == "mac":
        print("  Installation options:")
        print("    1. Automatic curl install (recommended)")
        print("    2. Cancel")
        c = input("  Choice > ").strip()
        if c != "1":
            return False
        print("\n  Installing ...")
        try:
            subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                check=True,
            )
            print("  Ollama installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Installation failed: {e}")
            return False
    elif os_type == "linux":
        print("  Installation options:")
        print("    1. Automatic curl install (recommended)")
        print("    2. Cancel")
        c = input("  Choice > ").strip()
        if c != "1":
            return False
        print("\n  Installing ...")
        try:
            subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                check=True,
            )
            print("  Ollama installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Installation failed: {e}")
            return False
    elif os_type == "windows":
        print("  Windows automatic installation:")
        print("    1. Download and run installer (winget)")
        print("    2. Cancel")
        c = input("  Choice > ").strip()
        if c != "1":
            return False
        print("\n  Installing via winget ...")
        try:
            subprocess.run(
                ["winget", "install", "Ollama.Ollama", "--accept-source-agreements", "--accept-package-agreements"],
                check=True,
            )
            print("  Ollama installed successfully.")
            print("  [!] Terminal restart may be required.")
            return True
        except FileNotFoundError:
            print("  [!] winget not available. Trying PowerShell fallback ...")
            try:
                subprocess.run(
                    ["powershell", "-Command",
                     "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile '$env:TEMP\\OllamaSetup.exe'; Start-Process '$env:TEMP\\OllamaSetup.exe' -Wait"],
                    check=True,
                )
                print("  Installer launched. Wait for installation to complete.")
                return True
            except Exception as e2:
                print(f"  [ERROR] {e2}")
                print("  Install manually: https://ollama.com/download")
                return False
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] {e}")
            return False
    return False


def start_ollama_serve():
    if is_ollama_running():
        return
    print("  Starting Ollama in background ...")
    os_type = detect_os()
    try:
        if os_type == "windows":
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        for _ in range(15):
            time.sleep(1)
            if is_ollama_running():
                print("  Ollama is ready.")
                return
        print("  [!] Ollama started but not ready yet. Continuing anyway.")
    except Exception as e:
        print(f"  [!] Could not start Ollama: {e}")


_HF_AVAILABLE_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-3-27b-it",
    "microsoft/Phi-4-reasoning",
]


def list_local_models() -> List[Dict[str, str]]:
    if _CLOUD_MODE:
        return [{"name": m, "size": "cloud", "modified": ""} for m in _HF_AVAILABLE_MODELS]
    base = _BACKEND_API_BASE or OLLAMA_API_BASE
    if _BACKEND_TYPE == "openai_compatible":
        try:
            resp = requests.get(f"{base}/v1/models", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return [
                {"name": m.get("id", ""), "size": "?", "modified": ""}
                for m in data.get("data", []) if m.get("id")
            ]
        except Exception:
            return []
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size_bytes = m.get("size", 0)
            size_gb = size_bytes / (1024 ** 3) if size_bytes else 0
            modified = m.get("modified_at", "")[:10]
            models.append({
                "name": name,
                "size": f"{size_gb:.1f}GB" if size_gb else "?",
                "modified": modified,
            })
        return models
    except Exception:
        return []


def pull_model_interactive(model_name: str) -> bool:
    print(f"\n  Downloading model '{model_name}' ...")
    print("  (This may take several minutes depending on size)")
    print()
    try:
        proc = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                sys.stdout.write(f"\r  {line[:70]:<70}")
                sys.stdout.flush()
        proc.wait()
        print()
        if proc.returncode == 0:
            print(f"  Model '{model_name}' downloaded successfully.")
            return True
        else:
            print(f"  [ERROR] ollama pull returned code {proc.returncode}")
            return False
    except FileNotFoundError:
        print("  [ERROR] 'ollama' command not found.")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def pick_model_interactive(label: str, current: str) -> str:
    models = list_local_models()
    print()
    if models:
        print(f"  -- {label} --")
        print(f"  Locally installed models:")
        for i, m in enumerate(models, 1):
            marker = " <-- current" if m["name"] == current else ""
            print(f"    {i:3d}. {m['name']:<30s}  {m['size']:>6s}  {m['modified']}{marker}")
        print(f"    {len(models)+1:3d}. [Enter a name manually / pull a new model]")
        print()
        raw = input(f"  Choice [{current}] > ").strip()
        if not raw:
            return current
        try:
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]["name"]
        except ValueError:
            pass
        if raw.isdigit() and int(raw) == len(models) + 1:
            raw = ""
    else:
        print(f"  -- {label} --")
        print("  No models installed locally.")
        print()
        raw = ""

    if not raw:
        name = input(f"  Model name (e.g. llama3:latest, qwen2.5:7b) [{current}] > ").strip()
        if not name:
            return current
        raw = name

    local_names = [m["name"] for m in models]
    if raw not in local_names:
        print(f"\n  Model '{raw}' is not installed locally.")
        do_pull = input("  Download now? (yes/no) [yes] > ").strip().lower()
        if do_pull in ("", "yes", "y", "oui", "o", "1"):
            if pull_model_interactive(raw):
                return raw
            else:
                print(f"  Using previous model: {current}")
                return current
        else:
            print(f"  [!] Model '{raw}' will be used as-is (may fail if not available).")
    return raw


def ensure_ollama_ready():
    if _CLOUD_MODE:
        return
    if not is_ollama_installed():
        if not install_ollama_interactive():
            print("\n  [!] Ollama is not available. Generation will fail.")
            print("  Install manually: https://ollama.com/download")
            return
    if not is_ollama_running():
        start_ollama_serve()


def ensure_models_available(settings: dict):
    """First-run: if no models are installed, guide the user to pull one."""
    models = list_local_models()
    if models:
        return
    print()
    print("  " + "─" * 58)
    print("  No local models found. You need at least one to generate.")
    print()
    print("  Recommended models (pick based on available RAM):")
    print("    1.  llama3.2:3b    ~2 GB   — fast, lightweight, good quality")
    print("    2.  llama3:8b      ~4.7 GB — solid all-rounder  (recommended)")
    print("    3.  qwen2.5:7b     ~4.4 GB — strong reasoning")
    print("    4.  mistral:7b     ~4.1 GB — instruction-tuned")
    print("    5.  Skip for now   (generation will fail until a model is pulled)")
    print("  " + "─" * 58)
    choice = input("  Choice [2] > ").strip() or "2"
    model_map = {
        "1": "llama3.2:3b",
        "2": "llama3:8b",
        "3": "qwen2.5:7b",
        "4": "mistral:7b",
    }
    if choice in model_map:
        chosen = model_map[choice]
        if pull_model_interactive(chosen):
            settings["model_a"] = chosen
            settings["model_b"] = chosen
            settings["synthesis_model"] = chosen
            save_settings(settings)
            print(f"\n  Default model set to: {chosen}")


# ----------------------------------------------------------------------
# Meta‑prompts (unchanged from original, kept for brevity)
# ----------------------------------------------------------------------
FULL_MANIFEST_STRUCTURE = """You produce exactly the following 12 sections, in this order:

### § 1. TITLE & EXECUTIVE SUMMARY
### § 2. FINAL OBJECTIVE & SUCCESS DEFINITION
### § 3. EXECUTION CONTEXT & PREREQUISITES
### § 4. AMBIGUITY ZONES TO RESOLVE
### § 5. STEP DECOMPOSITION (DETAILED PIPELINE)
### § 6. CONTROL LOOPS & SCORING
### § 7. PERSISTENT ARTIFACTS TO MAINTAIN
### § 8. CONSTRAINTS & GUARDRAILS
### § 9. ERROR HANDLING STRATEGY
### § 10. FINAL DELIVERABLE & OUTPUT FORMAT
### § 11. REPRODUCIBILITY CHECKLIST
### § 12. NOTES FOR THE TARGET AGENT"""

DRAFT_MANIFEST_STRUCTURE = """DRAFT MODE — you produce EXACTLY the following 2 sections ONLY. Do not write sections 3-12; do not summarize them; do not mention they exist.

### § 1. TITLE & EXECUTIVE SUMMARY
### § 2. FINAL OBJECTIVE & SUCCESS DEFINITION

DRAFT MODE OVERRIDE: ignore the "no length limit, write as much as necessary" rule above for this run. Keep each of the 2 sections to a short paragraph (2-5 sentences) — this is a fast preview, not the exhaustive manifest."""

META_PROMPT = """
# META-PROMPT: REPRODUCIBLE INSTRUCTION MANIFEST GENERATOR

> Usage: Paste this meta-prompt into any LLM agent, then provide your task in the USER_INPUT section. It will produce a complete, expert-level instruction manifest that another LLM agent can execute without prior context.

---

## ROLE

You are a senior prompt architect specializing in producing reproducible instruction manifests. Your mission is never to write code or execute the task. Your mission is to produce a manifest document that describes the task with such methodological precision that another LLM agent (Claude, GPT, Gemini, Llama, Mistral, Qwen) can reproduce it entirely using its own implementation methods.

You write like an RFC author, product spec writer, or technical article author — not like a developer. You describe the WHAT, the WHY, the IN WHAT ORDER, and the CONCEPTUAL-HOW — never the SYNTACTIC-HOW.

---

## USER INPUT

<<<USER_INPUT
{USER_INPUT}
USER_INPUT>>>

{TOPIC_FOCUS}

{MEMORY_CONTEXT}

{WEB_CONTEXT}

---

## ABSOLUTE OUTPUT RULES

1. Zero lines of code. No code blocks. No raw shell commands. No API syntax.
2. Total reproducibility. A different LLM agent must be able to reproduce the result.
3. Implementation-agnostic.
4. Manifest / article / spec style. Numbered sections, clear headings.
5. Expert level. Anticipate edge cases, pitfalls, and ambiguities.
6. Self-diagnose ambiguity if the input is ambiguous.
7. No length limit: write as much as necessary to exhaust the subject.

---

## MANDATORY STRUCTURE OF THE GENERATED MANIFEST

{SECTION_STRUCTURE}

---

## STRICT PROHIBITIONS

- No code blocks, regardless of justification.
- No shell commands, SQL queries, or raw API calls.
- No TODOs or "to be completed".
- No mention of yourself as the generator.
- No decorative emojis. Accepted symbols: empty square for checklists, arrow for transitions, § for sections.

---

Only produce the final manifest. No preamble. No conclusion. The manifest starts directly with "## § 1. TITLE & EXECUTIVE SUMMARY".
"""

SYNTH_PROMPT = """
# ROLE
You are a META-ARCHITECT of prompts, senior expert in LLM/AI/AGI instruction engineering, tasked with producing a FINAL SYNTHESIS with no length limit.

# MISSION
You have two instruction manifests generated by two distinct models (MODEL A and MODEL B) on the same user task. You must:

1. COMPARATIVE ANALYSIS: review both manifests, identify convergence points, divergences, omissions, and respective strengths and weaknesses.
2. MASTER PLAN: produce a structured, exhaustive, prioritized global plan.
3. UNIFIED SYNTHESIS: merge and enrich both manifests into a master document superior to either one, keeping the best of each, filling gaps, and adding blind spots that neither one saw.
4. FINAL EXPERT PROMPT-INSTRUCTION: at the end, produce a professional-quality expert prompt-instruction, ready to be injected as-is into another LLM agent (Claude, GPT, Gemini, etc.) to execute the task at the highest level.

# RULES
- No length limit: write as much as necessary, be exhaustive.
- No decorative unicode emojis.
- No executable code blocks.
- Style: technical, dense, professional, AGI prompt engineer expert level.
- Mandatory structure (numbered sections below).

# MANDATORY OUTPUT STRUCTURE

## PART I - COMPARATIVE ANALYSIS
### I.1 Convergences between the two manifests
### I.2 Notable divergences
### I.3 Detected omissions and blind spots
### I.4 Qualitative score by section (table)

## PART II - UNIFIED MASTER PLAN
### II.1 Vision and strategic objectives
### II.2 Major phases
### II.3 Dependencies and sequencing
### II.4 Critical risks and mitigation

## PART III - ENRICHED SYNTHETIC MANIFEST
Covers the 12 standard sections (§ 1 to § 12) in a unified, enriched version superior to both inputs.

## PART IV - FINAL EXPERT PROMPT-INSTRUCTION
A turnkey, self-contained prompt requiring no external context, ready to be pasted into any LLM agent to solve the task. Senior prompt engineer level. Include: role, context, constraints, methodology, success criteria, self-verification loops, deliverable format.

## PART V - META NOTES
Strategic advice for the target agent, known pitfalls, possible optimizations.

# INPUTS

## ORIGINAL USER TASK
<<<TASK
{USER_INPUT}
TASK>>>

## MODEL A MANIFEST ({MODEL_A})
<<<MANIFEST_A
{MANIFEST_A}
MANIFEST_A>>>

## MODEL B MANIFEST ({MODEL_B})
<<<MANIFEST_B
{MANIFEST_B}
MANIFEST_B>>>

{MEMORY_CONTEXT}

# START THE SYNTHESIS NOW
Begin directly with "## PART I - COMPARATIVE ANALYSIS". No preamble.
"""

QUICK_PROMPT = """
You are an expert prompt engineer with deep knowledge of LLM behavior.

Your task: take the user's raw input and rewrite it as a single, polished, production-ready prompt — optimized for maximum clarity, specificity, and LLM effectiveness.

Apply these prompt engineering techniques as you rewrite:
{TECHNIQUE_BLOCK}

{METACOMMANDS}

{TOPIC_FOCUS}

{MEMORY_CONTEXT}

{WEB_CONTEXT}

Raw user input:
<<<INPUT
{USER_INPUT}
INPUT>>>

Rules:
- Output ONLY the enhanced prompt. Nothing else.
- No preamble, no explanation, no meta-commentary, no section headers.
- The output must be ready to paste directly into any LLM (Claude, GPT, Gemini, Llama, Mistral).
- If the input contains /slash metacommands, they must be reflected in the style and tone of the enhanced prompt, not repeated literally.
- Be specific, concrete, and assume nothing the original left vague.

Start the enhanced prompt now:
"""

# ----------------------------------------------------------------------
# Persistent memory helpers (thread-safe with lock)
# ----------------------------------------------------------------------
_memory_lock = threading.Lock()

try:
    from cryptography.fernet import Fernet
    _FERNET_AVAILABLE = True
except ImportError:
    _FERNET_AVAILABLE = False

MEMORY_KEY_FILE = MEMORY_DIR / ".memory.key"
_MEMORY_ENC_MARKER = b"PPENC1:"
_MEMORY_ENCRYPTION_ENABLED = False


def set_memory_encryption_enabled(enabled: bool) -> None:
    """Enable/disable at-rest encryption for memory/sessions.json. Silently
    stays in plaintext (with a warning) if 'cryptography' isn't installed —
    this is an optional dependency, not a hard requirement."""
    global _MEMORY_ENCRYPTION_ENABLED
    if enabled and not _FERNET_AVAILABLE:
        logger.warning(
            "Memory encryption requested but the 'cryptography' package is not "
            "installed — staying in plaintext. Install with: pip install cryptography"
        )
        _MEMORY_ENCRYPTION_ENABLED = False
        return
    _MEMORY_ENCRYPTION_ENABLED = enabled


def _get_or_create_memory_key() -> bytes:
    if MEMORY_KEY_FILE.exists():
        return MEMORY_KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    MEMORY_KEY_FILE.write_bytes(key)
    try:
        os.chmod(MEMORY_KEY_FILE, 0o600)
    except Exception:
        pass  # best-effort; not all filesystems support POSIX permissions
    return key


def load_memory() -> dict:
    with _memory_lock:
        if not MEMORY_FILE.exists():
            return {"sessions": []}
        try:
            raw = MEMORY_FILE.read_bytes()
        except Exception:
            return {"sessions": []}
        if raw.startswith(_MEMORY_ENC_MARKER):
            if not _FERNET_AVAILABLE:
                logger.warning("Memory file is encrypted but 'cryptography' is not installed — cannot read it.")
                return {"sessions": []}
            try:
                key = _get_or_create_memory_key()
                decrypted = Fernet(key).decrypt(raw[len(_MEMORY_ENC_MARKER):])
                return json.loads(decrypted.decode("utf-8"))
            except Exception:
                logger.warning("Could not decrypt memory file (wrong/missing key) — resetting.")
                return {"sessions": []}
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.warning("Corrupted memory file; resetting.")
            return {"sessions": []}


def save_memory(mem: dict) -> None:
    with _memory_lock:
        payload = json.dumps(mem, ensure_ascii=False, indent=2).encode("utf-8")
        if _MEMORY_ENCRYPTION_ENABLED and _FERNET_AVAILABLE:
            key = _get_or_create_memory_key()
            token = Fernet(key).encrypt(payload)
            MEMORY_FILE.write_bytes(_MEMORY_ENC_MARKER + token)
        else:
            MEMORY_FILE.write_bytes(payload)


def append_session(entry: dict) -> None:
    mem = load_memory()
    mem.setdefault("sessions", []).append(entry)
    # keep only last 50 sessions
    mem["sessions"] = mem["sessions"][-50:]
    save_memory(mem)


def build_memory_context(max_items: int = 3) -> str:
    mem = load_memory()
    sessions = mem.get("sessions", [])[-max_items:]
    if not sessions:
        return ""
    lines = ["## MEMORY CONTEXT (previous sessions, for coherence)"]
    for s in sessions:
        lines.append(f"- [{s.get('timestamp','?')}] Task: {s.get('user_input','')[:200]}")
        if s.get("topics"):
            lines.append(f"  Topics covered: {', '.join(s['topics'])}")
    return "\n".join(lines)


def clear_memory():
    save_memory({"sessions": []})
    logger.debug("Memory cleared.")


def view_memory() -> str:
    mem = load_memory()
    sessions = mem.get("sessions", [])
    if not sessions:
        return "No sessions in memory."
    out = [f"Total sessions: {len(sessions)}", ""]
    for i, s in enumerate(reversed(sessions), 1):
        out.append(f"### Session {i} - {s.get('timestamp','?')}")
        out.append(f"Task: {s.get('user_input','')[:300]}")
        if s.get("topics"):
            out.append(f"Topics: {', '.join(s['topics'])}")
        if s.get("files"):
            out.append(f"Files: {', '.join(s['files'])}")
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Internet detection & web research
# ----------------------------------------------------------------------
_internet_available: Optional[bool] = None
_internet_checked_at: float = 0.0
# A negative result is re-checked periodically instead of being cached for
# the whole process lifetime — a one-off blip (VPN reconnecting, DNS not
# ready yet at startup) used to permanently disable web search/enrichment
# until the app was restarted.
_INTERNET_RECHECK_SECONDS = 60.0


def check_internet(timeout: float = 3.0) -> bool:
    global _internet_available, _internet_checked_at
    now = time.monotonic()
    if _internet_available is not None and (
        _internet_available or (now - _internet_checked_at) < _INTERNET_RECHECK_SECONDS
    ):
        return _internet_available
    for url in ("https://duckduckgo.com/", "https://www.google.com/"):
        try:
            requests.head(url, timeout=timeout, allow_redirects=True)
            _internet_available = True
            _internet_checked_at = now
            return True
        except Exception:
            continue
    _internet_available = False
    _internet_checked_at = now
    return False


WEB_CACHE_DIR = CACHE_DIR / "web"
WEB_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h


class _TextExtractor(HTMLParser):
    """HTML-to-text extractor that skips <script> and <style> content."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = False
        self._parts: List[str] = []
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)
    def get_text(self) -> str:
        return " ".join(self._parts)


def _web_cache_path(kind: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return WEB_CACHE_DIR / f"{kind}_{h}.json"


def _web_cache_get(kind: str, key: str):
    f = _web_cache_path(kind, key)
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["cached_at"])
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()
        if age > WEB_CACHE_TTL_SECONDS:
            return None
        return data.get("value")
    except Exception:
        return None


def _web_cache_set(kind: str, key: str, value) -> None:
    try:
        WEB_CACHE_DIR.mkdir(exist_ok=True, parents=True)
        _web_cache_path(kind, key).write_text(json.dumps({
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "value": value,
        }, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write web cache: {e}")


def web_search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    cache_key = f"{query}|{max_results}"
    cached = _web_cache_get("search", cache_key)
    if cached is not None:
        logger.debug("Web search cache hit.")
        return cached
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": HTTP_USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        results = []
        for m in re.finditer(
            r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>',
            resp.text,
        ):
            href = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if href and title:
                results.append({"title": title, "url": href})
            if len(results) >= max_results:
                break
        snippets = re.findall(
            r'<a class="result__snippet"[^>]*>(.*?)</a>', resp.text
        )
        for i, snip in enumerate(snippets):
            if i < len(results):
                results[i]["snippet"] = re.sub(r"<[^>]+>", "", snip).strip()
        if results:
            _web_cache_set("search", cache_key, results)
        return results
    except Exception as e:
        logger.debug(f"DuckDuckGo search failed: {e}")
        return []


_SSRF_BLOCK = re.compile(
    r"^https?://"
    r"(?:localhost|127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|::1|0\.0\.0\.0)",
    re.IGNORECASE,
)


def fetch_page_text(url: str, max_chars: int = 3000) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return ""
    if _SSRF_BLOCK.match(url):
        return ""
    cache_key = f"{url}|{max_chars}"
    cached = _web_cache_get("page", cache_key)
    if cached is not None:
        logger.debug("Page fetch cache hit.")
        return cached
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": HTTP_USER_AGENT},
            timeout=10,
            allow_redirects=True,
        )
        # Block redirects to internal addresses
        if _SSRF_BLOCK.match(resp.url):
            return ""
        resp.raise_for_status()
        extractor = _TextExtractor()
        extractor.feed(resp.text)
        text = re.sub(r"\s+", " ", extractor.get_text()).strip()
        result = text[:max_chars]
        if result:
            _web_cache_set("page", cache_key, result)
        return result
    except Exception:
        return ""


_SUMMARY_PROMPT = """Summarize the following webpage content in at most 100 words, keeping only information relevant to: {QUERY}

Output ONLY the summary. No preamble, no labels, no meta-commentary.

<<<CONTENT
{CONTENT}
CONTENT>>>

Summary:"""


def summarize_page_text(text: str, query: str, model: str, ollama_url: str = OLLAMA_URL, max_chars: int = 800) -> str:
    """Compress fetched page text via a small Ollama call so the WEB CONTEXT
    budget carries more actual signal than a blind truncation would. Falls
    back to plain truncation on any failure (never raises)."""
    if not text:
        return text
    if not model:
        return text[:max_chars]
    try:
        prompt = _SUMMARY_PROMPT.replace("{QUERY}", query[:150]).replace("{CONTENT}", text[:4000])
        summary = query_ollama(model, prompt, temperature=0.1, num_predict=200, timeout=30, ollama_url=ollama_url)
        summary = summary.strip()
        return summary[:max_chars] if summary else text[:max_chars]
    except Exception as e:
        logger.debug(f"Page summarization failed, falling back to truncation: {e}")
        return text[:max_chars]


def _fetch_web_context_now(task: str, topic: str = "", max_results: int = 3, max_pages: int = 1,
                            summarize: bool = False, summarizer_model: str = "", summarizer_url: str = OLLAMA_URL) -> str:
    if not check_internet():
        return ""
    # Truncate to 150 chars — enough context for search, never sends full user task
    query = (f"{topic} {task}" if topic else task).strip()[:150]
    logger.debug("Web research running...")
    results = web_search_duckduckgo(query, max_results=max_results)
    if not results:
        logger.debug("No web results found.")
        return ""
    lines = ["## WEB CONTEXT (automatic search, for enrichment)"]
    pages_fetched = 0
    for r in results:
        lines.append(f"- [{r['title']}]({r.get('url', '')})")
        if r.get("snippet"):
            lines.append(f"  Summary: {r['snippet']}")
        if pages_fetched < max_pages and r.get("url"):
            if summarize and summarizer_model:
                # Fetch more raw text than we'd inject directly, then compress it —
                # more signal survives than a blind truncation at the same final budget.
                page_text = fetch_page_text(r["url"], max_chars=4000)
                if page_text:
                    page_text = summarize_page_text(page_text, query, summarizer_model, summarizer_url)
            else:
                page_text = fetch_page_text(r["url"])
                if page_text:
                    page_text = page_text[:800]
            if page_text:
                lines.append(f"  Excerpt: {page_text}")
                pages_fetched += 1
    logger.debug(f"Web context: {len(results)} results, {pages_fetched} pages fetched.")
    return "\n".join(lines)


# Pre-processing (an Ollama call) and web enrichment are independent until
# both feed into build_full_prompt — preprocessing only clarifies wording, it
# never changes domain/intent (see PRE_PROCESSOR_PROMPT's "Preserve the user's
# intent exactly"), so a search kicked off on the RAW task text remains valid
# once preprocessing finishes. prefetch_web_context() lets callers start the
# search concurrently with the preprocessor call instead of strictly after it.
_web_context_lock = threading.Lock()
_pending_web_context: Dict[str, "concurrent.futures.Future"] = {}
_WEB_CONTEXT_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def prefetch_web_context(task: str, topic: str = "", max_results: int = 3, max_pages: int = 1,
                          summarize: bool = False, summarizer_model: str = "", summarizer_url: str = OLLAMA_URL) -> None:
    """Kick off the web search/fetch in the background, keyed by topic. A
    later build_web_context() call for the same topic will pick up this
    result instead of re-fetching. Safe to call even if use_web is off
    downstream — an unclaimed future is simply never awaited."""
    with _web_context_lock:
        if topic in _pending_web_context:
            return
        _pending_web_context[topic] = _WEB_CONTEXT_EXECUTOR.submit(
            _fetch_web_context_now, task, topic, max_results, max_pages, summarize, summarizer_model, summarizer_url
        )


def build_web_context(task: str, topic: str = "", max_results: int = 3, max_pages: int = 1,
                       summarize: bool = False, summarizer_model: str = "", summarizer_url: str = OLLAMA_URL) -> str:
    with _web_context_lock:
        future = _pending_web_context.pop(topic, None)
    if future is not None:
        try:
            return future.result()
        except Exception:
            logger.warning("Prefetched web context failed; continuing without it.")
            return ""
    return _fetch_web_context_now(task, topic, max_results, max_pages, summarize, summarizer_model, summarizer_url)


def run_deep_research(query: str, max_results: int = 8) -> str:
    """Search, show numbered candidate results, let the user pick which to
    fetch, then build a WEB CONTEXT block from only those sources. Requires
    a TTY — the selection step is interactive by design. Returns "" if the
    user cancels/selects nothing, so callers can fall back to no override
    (letting the normal automatic enrichment run instead)."""
    print()
    print("  ─" * 31)
    print("  DEEP RESEARCH — searching...")
    print("  ─" * 31)
    if not check_internet():
        print("  No internet connection available.")
        return ""
    results = web_search_duckduckgo(query, max_results=max_results)
    if not results:
        print("  No results found.")
        return ""
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']}")
        print(f"     {r.get('url', '')}")
        if r.get("snippet"):
            print(f"     {r['snippet'][:150]}")
        print()
    print("  Select sources to include (e.g. 1,3,5 or 'all', ENTER to skip):")
    try:
        raw = input("  > ").strip()
    except (KeyboardInterrupt, EOFError):
        return ""
    if not raw:
        return ""
    if raw.lower() == "all":
        chosen_indices = list(range(1, len(results) + 1))
    else:
        chosen_indices = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(results):
                chosen_indices.append(int(part))
    if not chosen_indices:
        print("  No valid selection — skipping deep research context.")
        return ""

    lines = ["## WEB CONTEXT (deep research — manually selected sources)"]
    for idx in chosen_indices:
        r = results[idx - 1]
        lines.append(f"- [{r['title']}]({r.get('url', '')})")
        if r.get("snippet"):
            lines.append(f"  Summary: {r['snippet']}")
        page_text = fetch_page_text(r["url"], max_chars=3000)
        if page_text:
            lines.append(f"  Excerpt: {page_text[:1200]}")
    print(f"  Included {len(chosen_indices)} source(s) in the deep research context.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Result cache — identical (model, prompt, params) requests are served from
# disk instead of re-calling Ollama. Persists across separate CLI runs.
# ----------------------------------------------------------------------
_RESULT_CACHE_ENABLED = True


def set_result_cache_enabled(enabled: bool) -> None:
    global _RESULT_CACHE_ENABLED
    _RESULT_CACHE_ENABLED = enabled


def _cache_key(model: str, prompt: str, temperature: float, num_predict: int, num_ctx: int) -> str:
    raw = f"{model}|{temperature}|{num_predict}|{num_ctx}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    f = CACHE_DIR / f"{key}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data.get("response")
    except Exception:
        return None


def _cache_set(key: str, model: str, response: str) -> None:
    f = CACHE_DIR / f"{key}.json"
    try:
        f.write_text(json.dumps({
            "model": model,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "response": response,
        }, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to write result cache: {e}")


# ----------------------------------------------------------------------
# Ollama interaction — batch and streaming
# ----------------------------------------------------------------------
def query_ollama(
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = -1,
    timeout: int = DEFAULT_TIMEOUT,
    ollama_url: str = OLLAMA_URL,
    num_ctx: int = 8192,
) -> str:
    cache_key = _cache_key(model, prompt, temperature, num_predict, num_ctx)
    if _RESULT_CACHE_ENABLED:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug(f"Result cache hit for model {model}.")
            return cached

    if _BACKEND_TYPE == "openai_compatible":
        is_chat = _is_chat_endpoint(ollama_url)
        payload = _openai_compatible_payload(model, prompt, temperature, num_predict, ollama_url, stream=False)
        headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
        try:
            resp = requests.post(ollama_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            result = _openai_compatible_extract_text(data["choices"][0], is_chat) or "[Empty response]"
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Could not connect to the OpenAI-compatible backend at {ollama_url}. Is it running?")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Timeout while waiting for model {model}.")
        except Exception as e:
            raise RuntimeError(f"Backend call to {model} failed: {e}")
        if _RESULT_CACHE_ENABLED:
            _cache_set(cache_key, model, result)
        return result

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            # Long structured outputs (§-numbered sections) can get the model
            # stuck repeating a whole prior block verbatim, especially with
            # num_predict=-1 (no length cap) — the default repeat_last_n=64
            # is too short a lookback to even notice a repeated multi-line
            # block, so widen it alongside a slightly stronger penalty.
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
        },
    }
    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("response", "[Empty response]")
        if _RESULT_CACHE_ENABLED:
            _cache_set(cache_key, model, result)
        return result
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to Ollama at {ollama_url}. Is it running?")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Timeout while waiting for model {model}.")
    except Exception as e:
        raise RuntimeError(f"Ollama call to {model} failed: {e}")


def query_ollama_stream(
    model: str,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = -1,
    timeout: int = DEFAULT_TIMEOUT,
    ollama_url: str = OLLAMA_URL,
    callback=None,
    num_ctx: int = 8192,
):
    cache_key = _cache_key(model, prompt, temperature, num_predict, num_ctx)
    if _RESULT_CACHE_ENABLED:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug(f"Result cache hit for model {model}.")
            if callback:
                callback(cached)
            return cached

    if _BACKEND_TYPE == "openai_compatible":
        is_chat = _is_chat_endpoint(ollama_url)
        payload = _openai_compatible_payload(model, prompt, temperature, num_predict, ollama_url, stream=True)
        headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}
        full_text = []
        try:
            with requests.post(ollama_url, json=payload, headers=headers, timeout=timeout, stream=True) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    token = _openai_compatible_extract_text(choices[0], is_chat)
                    if token:
                        full_text.append(token)
                        if callback:
                            callback(token)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Could not connect to the OpenAI-compatible backend at {ollama_url}. Is it running?")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Timeout while waiting for model {model}.")
        except Exception as e:
            raise RuntimeError(f"Backend stream to {model} failed: {e}")
        result = "".join(full_text)
        if _RESULT_CACHE_ENABLED and result:
            _cache_set(cache_key, model, result)
        return result

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
        },
    }
    full_text = []
    try:
        with requests.post(ollama_url, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                token = chunk.get("response", "")
                if token:
                    full_text.append(token)
                    if callback:
                        callback(token)
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Could not connect to Ollama at {ollama_url}. Is it running?")
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Timeout while waiting for model {model}.")
    except Exception as e:
        raise RuntimeError(f"Ollama stream to {model} failed: {e}")
    result = "".join(full_text)
    if _RESULT_CACHE_ENABLED and result:
        _cache_set(cache_key, model, result)
    return result


# ----------------------------------------------------------------------
# Dual-pane streaming for parallel generation
# ----------------------------------------------------------------------
class DualPaneDisplay:
    def __init__(self, label_a: str, label_b: str):
        self._lock = threading.Lock()
        self.label_a = label_a
        self.label_b = label_b
        self.buf_a: List[str] = []
        self.buf_b: List[str] = []
        self.lines_a: List[str] = [""]
        self.lines_b: List[str] = [""]
        term_size = shutil.get_terminal_size((120, 40))
        self.width = term_size.columns
        self.height = term_size.lines - 4
        self.col_w = (self.width - 3) // 2
        self._last_drawn = 0

    def _tokenize_lines(self, buf: List[str]) -> List[str]:
        text = "".join(buf)
        raw_lines = text.split("\n")
        wrapped = []
        for rl in raw_lines:
            while len(rl) > self.col_w:
                wrapped.append(rl[: self.col_w])
                rl = rl[self.col_w :]
            wrapped.append(rl)
        return wrapped

    def feed_a(self, token: str):
        with self._lock:
            self.buf_a.append(token)
            self.lines_a = self._tokenize_lines(self.buf_a)
        self._draw()

    def feed_b(self, token: str):
        with self._lock:
            self.buf_b.append(token)
            self.lines_b = self._tokenize_lines(self.buf_b)
        self._draw()

    def _draw(self):
        with self._lock:
            visible = self.height - 2
            la = self.lines_a[-visible:] if len(self.lines_a) > visible else self.lines_a
            lb = self.lines_b[-visible:] if len(self.lines_b) > visible else self.lines_b
            max_rows = max(len(la), len(lb), 1)

            header_a = f" {self.label_a} "
            header_b = f" {self.label_b} "
            ha = header_a.center(self.col_w, "-")
            hb = header_b.center(self.col_w, "-")

            output = [f"\033[2J\033[H{ha} | {hb}"]
            for i in range(max_rows):
                left = la[i] if i < len(la) else ""
                right = lb[i] if i < len(lb) else ""
                left = left[: self.col_w].ljust(self.col_w)
                right = right[: self.col_w].ljust(self.col_w)
                output.append(f"{left} | {right}")
            sep = "-" * self.col_w + "-+-" + "-" * self.col_w
            output.append(sep)
            status = f"  A: {len(self.lines_a)} lines | B: {len(self.lines_b)} lines"
            output.append(status)

            sys.stdout.write("\n".join(output))
            sys.stdout.flush()

    def get_texts(self) -> Tuple[str, str]:
        return "".join(self.buf_a), "".join(self.buf_b)


def get_technique_boost(technique_ids: List[int]) -> str:
    """Generate prompt enhancement block from selected techniques, grouped by category."""
    if not technique_ids:
        return ""
    by_cat: Dict[str, List[str]] = {}
    for tid in sorted(technique_ids):
        if tid in TECHNIQUES_DB:
            tech = TECHNIQUES_DB[tid]
            cat = tech.get("category", "Other")
            by_cat.setdefault(cat, []).append(f"- {tech['title']}: {tech['description']}")
    if not by_cat:
        return ""
    blocks = []
    for cat_name, items in by_cat.items():
        blocks.append(f"### {cat_name}")
        blocks.extend(items)
    blocks_str = "\n".join(blocks)
    if not blocks_str:
        return ""
    return (
        "## PROMPT ENGINEERING TECHNIQUES TO APPLY\n"
        "Integrate the following methods into your generation for superior quality:\n"
        + blocks_str
    )


def build_full_prompt(
    user_input: str,
    topic: str = "",
    use_memory: bool = True,
    techniques: Optional[List[int]] = None,
    use_web: bool = True,
    mode: str = "full",
    max_web_pages: int = 1,
    draft: bool = False,
    summarize_web_pages: bool = False,
    web_summary_model: str = "",
    ollama_url: str = OLLAMA_URL,
    web_context_override: Optional[str] = None,
) -> str:
    # Extract /slash metacommands from the raw input
    clean_input, meta_block = parse_metacommands(user_input)

    topic_block = ""
    if topic.strip():
        if mode == "quick":
            topic_block = f"## FOCUS\nConcentrate the enhanced prompt on this specific angle: '{topic.strip()}'."
        else:
            topic_block = (
                "## FOCUS TOPIC\n"
                f"Focus this entire manifest on the following topic: '{topic.strip()}'. "
                "Treat this topic as the dominant angle, going to maximum depth."
            )

    memory_block = build_memory_context() if use_memory else ""
    if web_context_override is not None:
        # Deep research mode: the user hand-picked these sources — use them
        # verbatim instead of running (or re-running) an automatic search.
        web_block = web_context_override
    else:
        web_block = build_web_context(
            clean_input, topic, max_pages=max_web_pages,
            summarize=summarize_web_pages, summarizer_model=web_summary_model, summarizer_url=ollama_url,
        ) if use_web else ""
    techniques_block = get_technique_boost(techniques or DEFAULT_TECHNIQUES)

    if mode == "quick":
        return (QUICK_PROMPT
                .replace("{USER_INPUT}", clean_input.strip())
                .replace("{TOPIC_FOCUS}", topic_block)
                .replace("{MEMORY_CONTEXT}", memory_block)
                .replace("{WEB_CONTEXT}", web_block)
                .replace("{TECHNIQUE_BLOCK}", techniques_block)
                .replace("{METACOMMANDS}", meta_block))

    # mode == "full" — original 12-section manifest
    combined_prefix = techniques_block
    if meta_block:
        combined_prefix = techniques_block + "\n" + meta_block

    return (META_PROMPT
            .replace("{USER_INPUT}", clean_input.strip())
            .replace("{TOPIC_FOCUS}", topic_block)
            .replace("{MEMORY_CONTEXT}", memory_block)
            .replace("{WEB_CONTEXT}", web_block)
            .replace("{SECTION_STRUCTURE}", DRAFT_MANIFEST_STRUCTURE if draft else FULL_MANIFEST_STRUCTURE)
            .replace("---\n\n## USER INPUT",
                     f"{combined_prefix}\n\n---\n\n## USER INPUT"))


# ----------------------------------------------------------------------
# Metacommand system — /slash modifiers applied at prompt build time
# ----------------------------------------------------------------------

METACOMMANDS: Dict[str, str] = {
    # ── Persona ──────────────────────────────────────────────────────
    "/expert":       "You are a certified senior expert with 20+ years in this domain. Write with authority and precision. Avoid generalities and hedging.",
    "/humain":       "Use a warm, conversational tone. Write as a knowledgeable friend, not a textbook.",
    "/enfant":       "Explain as if to a curious 10-year-old. Short sentences, simple words, concrete analogies only.",
    "/philosophe":   "Approach from a philosophical angle. Question assumptions, explore underlying principles, reference relevant frameworks.",
    "/sceptique":    "Adopt a critical, skeptical stance. Challenge every claim. Identify weaknesses, counterarguments, and what remains uncertain.",
    "/mentor":       "Take the role of a supportive mentor. Acknowledge difficulty, give practical guidance, encourage clear next steps.",
    "/cynique":      "Use dry, sardonic wit. See through optimism. Be realistic without being unhelpful.",
    "/serieux":      "Strictly formal and professional. No humor, no colloquialisms. Dense, structured, precise.",
    "/humour":       "Inject genuine humor throughout: puns, observations, light irony. Make the content memorable.",
    "/emphatique":   "Acknowledge the difficulty or importance of the topic first. Show understanding, then provide substance.",
    "/psychologue":  "Analyze cognitive and behavioral dimensions. Identify patterns and biases — do not invent motivations.",
    "/avocat":       "Build the strongest possible logical defense of the position. State clearly that this is advocacy, not objective truth.",
    "/journaliste":  "Write with journalistic rigor. Use real sources or clearly labeled hypothetical references only. No invented citations.",
    # ── Format ───────────────────────────────────────────────────────
    "/tableau":      "Structure the entire response as one or more tables. Rows and columns must be logically organized.",
    "/json":         "Output only valid JSON. No text before or after. Include an explicit schema in a leading comment.",
    "/markdown":     "Use rich Markdown: headers, bold, italic, code blocks, lists. Optimize for readability.",
    "/code":         "Illustrate with concrete, runnable code examples where applicable. Prefer examples over abstract description.",
    "/points":       "Structure as a numbered or bulleted list. Each item must be self-contained and substantive.",
    "/checklist":    "Output a markdown checklist. Each item is actionable. Group by category if more than 7 items.",
    # ── Depth / Length ───────────────────────────────────────────────
    "/resume":       "Summarize in maximum 5 bullet points. Each captures one essential idea. No filler.",
    "/concis":       "Maximum 50 words. Every word must carry weight. Cut everything else.",
    "/detaille":     "Write between 500 and 1000 words. Develop every point fully. No shortcuts.",
    "/urgent":       "Skip all preamble. Deliver the actionable answer immediately, first line.",
    "/silence":      "Output only the direct answer. No preamble, no explanation, no conclusion.",
    "/minimal":      "Bare minimum: the core answer and one sentence of essential context. Nothing else.",
    # ── Reasoning ────────────────────────────────────────────────────
    "/raisonnement": "Walk through your reasoning step by step before giving the final answer. Show the thinking path, not just the conclusion.",
    "/etapes":       "Break down into clearly numbered sequential steps. Each step: what to do, why, expected output.",
    "/exemple":      "Begin with a concrete real-world example before any theory or abstraction.",
    "/analogie":     "Explain using two analogies from everyday life. Then discuss the limits of each.",
    "/pourcontre":   "Structure as PROS / CONS with equal depth on each side. Conclude with a synthesized recommendation.",
    "/debat":        "Present as a debate between two named experts with opposing views. Summarize consensus at the end.",
    "/reverse":      "Argue the opposite of the expected answer first. Then reconcile by explaining when each position holds.",
    "/iterer":       "Produce 3 distinct versions: V1 (minimal), V2 (standard), V3 (exhaustive).",
    "/ameliorer":    "Identify 3 weaknesses in how this is typically approached. Then rewrite addressing each weakness explicitly.",
    "/reword":       "Restate the question in your own words before answering to confirm understanding.",
    "/critique":     "List every flaw, gap, assumption, and risk in this topic or approach. Be thorough and unsympathetic.",
    "/audit":        "Systematic audit: check for errors, inconsistencies, omissions, and logic or security issues.",
    "/comparatif":   "Compare at least 3 distinct approaches on the same criteria. Use a table for the comparison matrix.",
    # ── Context ──────────────────────────────────────────────────────
    "/historique":   "Ground the answer in historical context. Trace evolution from origin to present state.",
    "/futuriste":    "Project 10 to 50 years forward. What changes? What are the implications and risks?",
    "/questionner":  "Do not answer directly. Ask 3 to 5 clarifying questions that would lead to a better answer.",
    "/neuf":         "Treat this as a completely independent request. Ignore all previous context in this session.",
    "/priorite":     "Focus primarily on the most recent instruction. Other context is secondary.",
    # ── Quality ──────────────────────────────────────────────────────
    "/precision":    "Signal your confidence level for each claim (high / medium / low). Distinguish facts from interpretations.",
    "/hypotheses":   "List all assumptions you are making before answering. For each, state its validity and impact if wrong.",
    "/sources":      "Support each factual claim with a real source or, if unavailable, a clearly labeled hypothetical reference.",
    "/risques":      "Identify and categorize risks: technical, legal, operational, ethical. Rate each by likelihood and impact.",
    "/decision":     "End with a clear, argued recommendation. State what you would do and why.",
    "/verification": "Before outputting, verify internal consistency. Flag any contradiction or uncertainty explicitly.",
    "/confiance":    "For each factual claim, assign a confidence percentage (0–100%) and justify it briefly.",
    "/questions":    "End with the 3 best follow-up questions a curious expert would ask next.",
    "/sansbuzz":     "Avoid all buzzwords, marketing language, and jargon. Use plain, direct language only.",
}


def parse_metacommands(text: str) -> Tuple[str, str]:
    """
    Extract /slash tokens from the task text.
    Returns (cleaned_task, instruction_block).
    The instruction_block is empty string if no metacommands were found.
    """
    tokens = text.split()
    instructions: List[str] = []
    clean: List[str] = []

    for token in tokens:
        if not token.startswith("/"):
            clean.append(token)
            continue

        low = token.lower()

        # Parametric: /limite:N
        if low.startswith("/limite:"):
            try:
                n = int(low.split(":", 1)[1])
                instructions.append(f"Maximum response length: {n} words. Do not exceed this limit.")
            except ValueError:
                clean.append(token)
            continue

        # Parametric: /niveau:X
        if low.startswith("/niveau:"):
            level_raw = low.split(":", 1)[1]
            level_map = {
                "debutant": "beginner", "débutant": "beginner", "beginner": "beginner",
                "intermediaire": "intermediate", "intermédiaire": "intermediate", "intermediate": "intermediate",
                "expert": "expert", "avance": "advanced", "avancé": "advanced",
            }
            level = level_map.get(level_raw, level_raw)
            instructions.append(f"Calibrate depth, vocabulary, and assumed knowledge for a {level}-level audience.")
            continue

        # Parametric: /confiance (with or without value)
        if low.startswith("/confiance"):
            instructions.append("For each factual claim, assign a confidence percentage (0–100%) and justify it briefly.")
            continue

        if low in METACOMMANDS:
            instructions.append(METACOMMANDS[low])
        else:
            # Unknown /command — pass through as literal text
            clean.append(token)

    cleaned = " ".join(clean).strip()

    if not instructions:
        return cleaned, ""

    block = (
        "\n## ACTIVE METACOMMANDS — APPLY STRICTLY\n"
        "The following behavioral directives override default style, format, and depth:\n\n"
        + "\n".join(f"- {instr}" for instr in instructions)
        + "\n"
    )
    return cleaned, block


def split_topics(topics_raw: str) -> List[str]:
    """Parse a user‑supplied topics string into a list of non‑empty topics."""
    if not topics_raw or not topics_raw.strip():
        return []
    # split on newline, comma, or semicolon; allow dashes/bullets as decorations
    parts = re.split(r"[\n;,]+", topics_raw)
    return [p.strip(" -*\t") for p in parts if p.strip(" -*\t")]


# ----------------------------------------------------------------------
# Core generation logic
# ----------------------------------------------------------------------
def generate_multi_topics(
    model: str,
    user_input: str,
    topics_raw: str,
    temperature: float,
    use_memory: bool,
    ollama_url: str,
    timeout: int,
    techniques: Optional[List[int]] = None,
    use_web: bool = True,
    stream_callback=None,
    mode: str = "full",
    max_web_pages: int = 1,
    draft: bool = False,
    summarize_web_pages: bool = False,
    web_context_override: Optional[str] = None,
) -> str:
    if not user_input.strip():
        raise ValueError("Task description must not be empty.")
    topics = split_topics(topics_raw)
    if not topics:
        topics = [""]
    chunks = []
    for idx, topic in enumerate(topics, 1):
        label = topic if topic else "prompt"
        logger.debug(f"Generating {idx}/{len(topics)} for model {model}: '{label}'")
        prompt = build_full_prompt(
            user_input, topic, use_memory, techniques, use_web, mode, max_web_pages=max_web_pages, draft=draft,
            summarize_web_pages=summarize_web_pages, web_summary_model=model if summarize_web_pages else "",
            ollama_url=ollama_url, web_context_override=web_context_override,
        )
        if stream_callback:
            text = query_ollama_stream(model, prompt, temperature, num_predict=-1, timeout=timeout, ollama_url=ollama_url, callback=stream_callback)
        else:
            text = query_ollama(model, prompt, temperature, num_predict=-1, timeout=timeout, ollama_url=ollama_url)
        chunks.append(
            f"\n\n{'='*70}\nOUTPUT {idx} / {len(topics)} - TOPIC: {label}\n{'='*70}\n\n{text}"
        )
    full = "".join(chunks)
    append_session({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "model": model,
        "topics": [t for t in topics if t],
        "mode": "multi_topics",
        "techniques": techniques or DEFAULT_TECHNIQUES,
        "web_enriched": use_web and check_internet(),
    })
    return full


def generate_parallel_both(
    user_input: str,
    topics_raw: str,
    temperature: float,
    use_memory: bool,
    ollama_url: str,
    timeout: int,
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
    techniques: Optional[List[int]] = None,
    use_web: bool = True,
    live_display: bool = False,
    mode: str = "full",
    max_web_pages: int = 1,
    draft: bool = False,
    summarize_web_pages: bool = False,
    web_context_override: Optional[str] = None,
) -> Tuple[str, str, Path, Path]:
    if not user_input.strip():
        raise ValueError("Task description must not be empty.")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    if live_display and sys.stdout.isatty():
        pane = DualPaneDisplay(model_a, model_b)

        def worker_a():
            return generate_multi_topics(
                model_a, user_input, topics_raw, temperature, use_memory,
                ollama_url, timeout, techniques, use_web, stream_callback=pane.feed_a, mode=mode,
                max_web_pages=max_web_pages, draft=draft, summarize_web_pages=summarize_web_pages,
                web_context_override=web_context_override,
            )

        def worker_b():
            return generate_multi_topics(
                model_b, user_input, topics_raw, temperature, use_memory,
                ollama_url, timeout, techniques, use_web, stream_callback=pane.feed_b, mode=mode,
                max_web_pages=max_web_pages, draft=draft, summarize_web_pages=summarize_web_pages,
                web_context_override=web_context_override,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(worker_a)
            fb = pool.submit(worker_b)
            out_a = fa.result()
            out_b = fb.result()
        print("\n")
    else:
        def worker(model):
            return generate_multi_topics(model, user_input, topics_raw, temperature, use_memory,
                                         ollama_url, timeout, techniques, use_web, mode=mode,
                                         max_web_pages=max_web_pages, draft=draft,
                                         summarize_web_pages=summarize_web_pages,
                                         web_context_override=web_context_override)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(worker, model_a)
            fb = pool.submit(worker, model_b)
            out_a = fa.result()
            out_b = fb.result()

    file_a = OUTPUT_DIR / f"{session_id}_{model_a.replace(':','_')}.md"
    file_b = OUTPUT_DIR / f"{session_id}_{model_b.replace(':','_')}.md"

    kind = "Output" if mode == "quick" else "Manifest"
    header_a = f"# {kind} — {model_a}\n\nTask: {user_input}\nTopics: {topics_raw or '(global)'}\nGenerated: {session_id}\n\n---\n"
    header_b = f"# {kind} — {model_b}\n\nTask: {user_input}\nTopics: {topics_raw or '(global)'}\nGenerated: {session_id}\n\n---\n"

    file_a.write_text(header_a + out_a, encoding="utf-8")
    file_b.write_text(header_b + out_b, encoding="utf-8")
    logger.debug(f"Parallel output written to {file_a.name} and {file_b.name}")

    append_session({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "session_id": session_id,
        "topics": split_topics(topics_raw),
        "mode": "parallel_both",
        "files": [str(file_a.name), str(file_b.name)],
        "techniques": techniques or DEFAULT_TECHNIQUES,
        "web_enriched": use_web and check_internet(),
    })
    return out_a, out_b, file_a, file_b


def run_synthesis(
    user_input: str,
    manifest_a: str,
    manifest_b: str,
    temperature: float,
    use_memory: bool,
    ollama_url: str,
    timeout: int,
    synthesis_model: str = DEFAULT_SYNTH_MODEL,
    techniques: Optional[List[int]] = None,
    stream: bool = False,
    model_a_name: str = "",
    model_b_name: str = "",
    stream_callback=None,
) -> Tuple[str, Path]:
    """
    Generate a meta‑synthesis from two manifest texts.
    Returns (synthesis_text, output_path).
    """
    if not manifest_a.strip() or not manifest_b.strip():
        raise ValueError("Both manifests must contain text.")
    if manifest_a.startswith("[ERREUR") or manifest_b.startswith("[ERREUR"):
        raise ValueError("One of the manifests contains an error; cannot synthesise.")

    techniques_block = get_technique_boost(techniques or DEFAULT_TECHNIQUES)

    memory_block = build_memory_context() if use_memory else ""
    prompt = (SYNTH_PROMPT
              .replace("{USER_INPUT}", user_input.strip() or "(not provided)")
              .replace("{MODEL_A}", model_a_name or DEFAULT_MODEL_A)
              .replace("{MODEL_B}", model_b_name or DEFAULT_MODEL_B)
              .replace("{MANIFEST_A}", manifest_a.strip())
              .replace("{MANIFEST_B}", manifest_b.strip())
              .replace("{MEMORY_CONTEXT}", memory_block)
              .replace("# INPUTS", f"{techniques_block}\n\n# INPUTS"))

    ctx_size = max(16384, len(prompt) // 3)

    logger.debug("Starting expert synthesis ...")
    if stream_callback is not None:
        synthesis = query_ollama_stream(
            synthesis_model, prompt, temperature,
            num_predict=-1, timeout=timeout, ollama_url=ollama_url,
            callback=stream_callback, num_ctx=ctx_size,
        )
    elif stream and sys.stdout.isatty():
        print(f"\n{'='*62}")
        print(f"  SYNTHESIS IN PROGRESS — {synthesis_model}")
        print(f"{'='*62}\n")
        def stream_print(token):
            sys.stdout.write(token)
            sys.stdout.flush()
        synthesis = query_ollama_stream(
            synthesis_model, prompt, temperature,
            num_predict=-1, timeout=timeout, ollama_url=ollama_url,
            callback=stream_print, num_ctx=ctx_size,
        )
        print("\n")
    else:
        synthesis = query_ollama(synthesis_model, prompt, temperature,
                                 num_predict=-1, timeout=timeout, ollama_url=ollama_url,
                                 num_ctx=ctx_size)

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_synth_" + uuid.uuid4().hex[:6]
    out_file = OUTPUT_DIR / f"{session_id}_synthesis.md"
    header = (f"# Expert Synthesis\n\nTask: {user_input}\n"
              f"Synthesis model: {synthesis_model}\nGenerated: {session_id}\n\n---\n")
    out_file.write_text(header + synthesis, encoding="utf-8")
    logger.debug(f"Synthesis saved to outputs/{out_file.name}")

    append_session({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": user_input,
        "session_id": session_id,
        "mode": "synthesis",
        "files": [str(out_file.name)],
        "techniques": techniques or DEFAULT_TECHNIQUES,
    })
    return synthesis, out_file


def run_full_pipeline(
    user_input: str,
    topics_raw: str,
    temperature: float,
    use_memory: bool,
    ollama_url: str,
    timeout: int,
    model_a: str,
    model_b: str,
    synthesis_model: str,
    techniques: Optional[List[int]] = None,
    use_web: bool = True,
    live_display: bool = False,
    stream: bool = False,
    mode: str = "full",
    max_web_pages: int = 1,
    draft: bool = False,
    summarize_web_pages: bool = False,
    web_context_override: Optional[str] = None,
):
    out_a, out_b, fa, fb = generate_parallel_both(
        user_input, topics_raw, temperature, use_memory, ollama_url, timeout,
        model_a, model_b, techniques, use_web, live_display, mode=mode,
        max_web_pages=max_web_pages, draft=draft, summarize_web_pages=summarize_web_pages,
        web_context_override=web_context_override,
    )
    synthesis, fs = run_synthesis(
        user_input, out_a, out_b, temperature, use_memory, ollama_url, timeout,
        synthesis_model, techniques, stream=stream,
        model_a_name=model_a, model_b_name=model_b,
    )
    return out_a, out_b, fa, fb, synthesis, fs


# ----------------------------------------------------------------------
# CLI definition
# ----------------------------------------------------------------------
def _resolve_technique_bundle(key: str) -> Optional[List[int]]:
    """Resolve a QUICK_REFERENCE task-type bundle by 1-based index or
    case-insensitive name match. Returns None if no bundle matches."""
    if not QUICK_REFERENCE:
        return None
    names = list(QUICK_REFERENCE.keys())
    if key.isdigit():
        idx = int(key) - 1
        if 0 <= idx < len(names):
            return list(QUICK_REFERENCE[names[idx]])
        return None
    key_lower = key.lower()
    for name in names:
        if name.lower() == key_lower:
            return list(QUICK_REFERENCE[name])
    return None


def _random_technique_subset(count_raw: str) -> List[int]:
    """Pick a random subset of TECHNIQUES_DB, for creative exploration.
    'random' alone uses len(DEFAULT_TECHNIQUES) as the subset size; 'random:N'
    picks exactly N (clamped to the available pool size)."""
    pool = list(TECHNIQUES_DB.keys())
    if not pool:
        return DEFAULT_TECHNIQUES
    count = len(DEFAULT_TECHNIQUES)
    if ":" in count_raw:
        try:
            count = int(count_raw.split(":", 1)[1].strip())
        except ValueError:
            logger.warning(f"Invalid random technique count in '{count_raw}' — using default size.")
    count = max(1, min(count, len(pool)))
    return sorted(random.sample(pool, count))


def parse_techniques(techniques_raw: str) -> List[int]:
    """Parse comma-separated technique IDs, ranges, a 'bundle:<name-or-index>'
    reference into one of the task-type bundles from prompt_expert_methodology.json
    (see QUICK_REFERENCE — e.g. 'bundle:3' or 'bundle:Generation de code robuste'),
    or 'random' / 'random:N' for a random subset (creative exploration mode)."""
    if not techniques_raw or not techniques_raw.strip():
        return DEFAULT_TECHNIQUES
    techniques_raw = techniques_raw.strip()
    if techniques_raw.lower().startswith("bundle:"):
        key = techniques_raw.split(":", 1)[1].strip()
        bundle = _resolve_technique_bundle(key)
        if bundle is None:
            logger.warning(f"Unknown technique bundle: '{key}' — falling back to defaults.")
            return DEFAULT_TECHNIQUES
        return sorted(set(bundle))
    if techniques_raw.lower() == "random" or techniques_raw.lower().startswith("random:"):
        return _random_technique_subset(techniques_raw.lower())
    if techniques_raw.lower() == "all":
        logger.warning("'all' would stack every technique on one prompt, which degrades output — "
                       "techniques are meant to be selected per task (see --recommend-techniques, "
                       "'bundle:<name-or-#>', or an explicit range). Using the default set.")
        return sorted(DEFAULT_TECHNIQUES)
    result = []
    for part in techniques_raw.split(","):
        part = part.strip()
        if "-" in part:
            # Range format: 1-5
            try:
                start, end = part.split("-")
                result.extend(range(int(start.strip()), int(end.strip()) + 1))
            except ValueError:
                logger.warning(f"Invalid range format: {part}")
        else:
            try:
                result.append(int(part))
            except ValueError:
                logger.warning(f"Invalid technique ID: {part}")
    ids = sorted(set(result))
    if not ids:
        logger.warning(f"No valid technique IDs in '{techniques_raw}' — using the default set.")
        return sorted(DEFAULT_TECHNIQUES)
    return ids


def common_args(parser: argparse.ArgumentParser):
    """Add arguments common to all generation commands."""
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM temperature (default: 0.3)")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama API endpoint")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per Ollama call in seconds")
    parser.add_argument("--no-memory", action="store_true", default=False, help="Disable session memory context")
    parser.add_argument("--techniques", default="",
                        help="Technique IDs: 1,5,8 or range 1-20. Also accepts 'bundle:<name-or-#>' "
                             "for a task-type bundle, or 'random'/'random:N' for a random subset. "
                             "Techniques work best selected per task — prefer --recommend-techniques or a bundle.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full",
                        help="quick = single enhanced prompt; full = 12-section manifest (default: full)")
    parser.add_argument("--list-techniques", action="store_true", help="List all 173 prompt engineering techniques")
    parser.add_argument("--no-preprocess", action="store_true", default=False,
                        help="Skip pre-processor step (use raw input directly)")
    parser.add_argument("--pre-processor-model", default="",
                        help="Distinct model for the pre-processor step (e.g. a lighter/faster model). "
                             "Falls back to the settings value, then the main model, if unset.")
    parser.add_argument("--max-web-pages", type=int, default=1,
                        help="Number of search-result pages to fetch full text from during web enrichment (default: 1)")
    parser.add_argument("--offline", action="store_true", default=False,
                        help="Fully disable web enrichment — no internet check, no search, no page fetch")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress all log output (equivalent to the interactive UI's log silencing)")
    parser.add_argument("--draft", action="store_true", default=False,
                        help="Generate only §1-2 of the 12-section manifest, for fast iteration (mode=full only)")
    parser.add_argument("--recommend-techniques", action="store_true", default=False,
                        help="Auto-select techniques from the task's content instead of --techniques/defaults")
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Bypass the result cache — always call Ollama, even for an identical prior request")
    parser.add_argument("--fast-preprocess", action="store_true", default=False,
                        help="Regex-only pre-processing (whitespace/punctuation/capitalization) — no Ollama call, instant, less thorough")
    parser.add_argument("--anonymize", action="store_true", default=False,
                        help="Redact emails/phones/IPs/MACs/SSNs/card numbers/self-identified names from the task before it reaches any model")
    parser.add_argument("--summarize-web-pages", action="store_true", default=False,
                        help="Summarize fetched web pages via an extra Ollama call (using the generation model) instead of truncating raw text")
    parser.add_argument("--deep-research", action="store_true", default=False,
                        help="Search, show candidate sources, and let you manually pick which to include (requires an interactive terminal)")
    parser.add_argument("--backend", choices=["ollama", "openai_compatible"], default="",
                        help="Backend type: 'ollama' (default) or 'openai_compatible' for LM Studio / GPT4All server mode / "
                             "text-generation-webui's OpenAI extension. Point --ollama-url at that server's endpoint "
                             "(e.g. http://localhost:1234/v1/completions).")
    parser.add_argument("--template", default="",
                        help="Use a predefined task template by id (see 'templates list'). If the positional task is "
                             "also given, it fills the template's [TOPIC] placeholder; otherwise it's left as-is.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wild_Root_Prompt: enhance any prompt using 173 prompt-engineering techniques via Ollama."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate manifest with a single model")
    gen_parser.add_argument("task", nargs="?", default="", help="Task description (omit if using --template)")
    gen_parser.add_argument("--topic", action="append", dest="topics", help="Topic(s) to focus on (repeatable)")
    gen_parser.add_argument("--model", default=DEFAULT_MODEL_A, help="Ollama model name")
    gen_parser.add_argument("--output", help="Output file (if omitted, auto‑generated)")
    common_args(gen_parser)

    # parallel
    par_parser = subparsers.add_parser("parallel", help="Generate manifests from two models in parallel")
    par_parser.add_argument("task", nargs="?", default="", help="Task description (omit if using --template)")
    par_parser.add_argument("--topic", action="append", dest="topics", help="Topic(s) (repeatable)")
    par_parser.add_argument("--model-a", default=DEFAULT_MODEL_A, help="First model")
    par_parser.add_argument("--model-b", default=DEFAULT_MODEL_B, help="Second model")
    common_args(par_parser)

    # synthesis
    synth_parser = subparsers.add_parser("synthesis", help="Run expert synthesis on two existing manifests")
    synth_parser.add_argument("task", help="Original task description")
    synth_parser.add_argument("file_a", type=Path, help="Path to manifest A (text file)")
    synth_parser.add_argument("file_b", type=Path, help="Path to manifest B (text file)")
    synth_parser.add_argument("--synthesis-model", default=DEFAULT_SYNTH_MODEL, help="Model for synthesis")
    synth_parser.add_argument("--output", help="Output file")
    common_args(synth_parser)

    # full pipeline
    full_parser = subparsers.add_parser("full", help="Full pipeline: parallel generation + synthesis")
    full_parser.add_argument("task", nargs="?", default="", help="Task description (omit if using --template)")
    full_parser.add_argument("--topic", action="append", dest="topics", help="Topic(s) (repeatable)")
    full_parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    full_parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    full_parser.add_argument("--synthesis-model", default=DEFAULT_SYNTH_MODEL)
    common_args(full_parser)

    # templates
    tmpl_parser = subparsers.add_parser("templates", help="List/show predefined prompt templates")
    tmpl_sub = tmpl_parser.add_subparsers(dest="tmpl_cmd", required=True)
    tmpl_sub.add_parser("list", help="List all available templates")
    tmpl_show_parser = tmpl_sub.add_parser("show", help="Show a template's full task text")
    tmpl_show_parser.add_argument("id", help="Template id (see 'templates list')")

    # memory
    mem_parser = subparsers.add_parser("memory", help="Manage session memory")
    mem_sub = mem_parser.add_subparsers(dest="mem_cmd", required=True)
    mem_sub.add_parser("view", help="Display memory contents")
    mem_sub.add_parser("clear", help="Erase all memory")

    # web UI
    web_parser = subparsers.add_parser("web", help="Launch local web interface (novice-friendly)")
    web_parser.add_argument("--port", type=int, default=7860, help="Port for the web UI (default: 7860)")
    web_parser.add_argument("--no-browser", action="store_true", default=False,
                            help="Don't open browser automatically")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "quiet", False):
        logging.disable(logging.CRITICAL)
    if getattr(args, "no_cache", False):
        set_result_cache_enabled(False)

    # Handle --list-techniques flag if present
    if hasattr(args, "list_techniques") and args.list_techniques:
        print("Available Prompt Engineering Techniques:")
        print("=" * 80)
        if CATEGORIES_DB:
            for cat in CATEGORIES_DB:
                print(f"\n[{cat['id'].upper()}] {cat['name']}")
                print(f"{cat.get('description', '')}")
                print("-" * 60)
                for tech in cat.get("techniques", []):
                    print(f"{tech['id']:3d}. {tech['title']}")
                    print(f"      {tech['description']}\n")
        else:
            for tid in sorted(TECHNIQUES_DB.keys()):
                tech = TECHNIQUES_DB[tid]
                print(f"{tid:3d}. {tech['title']}")
                print(f"      {tech['description']}\n")
        return

    if args.command == "templates":
        if args.tmpl_cmd == "list":
            if not PROMPT_TEMPLATES:
                print("No templates found (prompt_templates.json missing or empty).")
                return
            by_category: Dict[str, List[Dict]] = {}
            for t in PROMPT_TEMPLATES.values():
                by_category.setdefault(t.get("category", "Other"), []).append(t)
            for cat, tmpls in by_category.items():
                print(f"\n[{cat}]")
                for t in tmpls:
                    print(f"  {t['id']:<22s} {t['title']}")
            print("\nUse: python3 prompt_expert_enhance.py templates show <id>")
            print("Or:  python3 prompt_expert_enhance.py generate \"my topic\" --template <id>")
        elif args.tmpl_cmd == "show":
            t = PROMPT_TEMPLATES.get(args.id)
            if not t:
                print(f"[ERROR] Unknown template '{args.id}'. Run: python3 prompt_expert_enhance.py templates list")
                sys.exit(1)
            print(f"{t['title']} ({t['id']})")
            print(f"Category: {t.get('category', 'Other')}")
            print(f"\n{t['task']}")
        return

    use_memory = not args.no_memory if hasattr(args, "no_memory") else True
    techniques = parse_techniques(args.techniques) if hasattr(args, "techniques") else DEFAULT_TECHNIQUES
    mode = getattr(args, "mode", "full")
    use_web = not getattr(args, "offline", False)
    settings = load_settings()
    set_memory_encryption_enabled(settings.get("encrypt_memory", False))
    backend_type = getattr(args, "backend", "") or settings.get("backend_type", "ollama")
    set_backend_type(backend_type)
    set_backend_api_base(getattr(args, "ollama_url", "") or settings.get("ollama_url", OLLAMA_URL))

    # Resolve --template into args.task before anything else touches it
    if getattr(args, "template", ""):
        tmpl = PROMPT_TEMPLATES.get(args.template)
        if not tmpl:
            print(f"[ERROR] Unknown template '{args.template}'. Run: python3 prompt_expert_enhance.py templates list")
            sys.exit(1)
        topic_fill = (getattr(args, "task", "") or "").strip()
        template_task = tmpl["task"]
        if topic_fill:
            template_task = template_task.replace("[TOPIC]", topic_fill)
        else:
            print(f"[templates] Using '{tmpl['title']}' — remember to replace [TOPIC] in the output.")
        args.task = template_task

    if args.command in ("generate", "parallel", "full") and not getattr(args, "task", ""):
        print("[ERROR] No task provided — pass one directly or use --template <id>.")
        sys.exit(1)

    # CLI pre-processor
    cli_task = sanitize_input(getattr(args, "task", ""), "text")
    if cli_task and getattr(args, "anonymize", False):
        cli_task, pii_labels = anonymize_pii(cli_task)
        if pii_labels:
            print(f"[privacy] Redacted before sending to any model: {', '.join(sorted(set(pii_labels)))}")
        args.task = cli_task
    if cli_task:
        injection_labels = detect_prompt_injection(cli_task)
        if injection_labels:
            print(f"[warning] Input contains phrasing that resembles a prompt-injection/jailbreak attempt ({', '.join(injection_labels)}). "
                  "Proceeding — this is advisory only, not a block.")
    deep_research_context = None
    if cli_task and getattr(args, "deep_research", False):
        deep_research_context = run_deep_research(cli_task) or None
    if cli_task and not getattr(args, "no_preprocess", False) and settings.get("use_pre_processor", True):
        # Skip the prefetch when summarization is on: the prefetch mechanism
        # doesn't carry summarizer params, so a prefetched hit would bypass
        # summarization. build_full_prompt's direct build_web_context() call
        # handles it correctly instead — one sequential fetch instead of a
        # parallel one, but summarization already adds its own latency.
        if use_web and not getattr(args, "summarize_web_pages", False) and deep_research_context is None:
            # Start the web search now, on the raw task, concurrently with the
            # preprocessor's Ollama call below — build_full_prompt's later
            # build_web_context() call for this same topic will reuse it instead
            # of fetching again sequentially.
            first_topic = (args.topics[0] if getattr(args, "topics", None) else "")
            prefetch_web_context(cli_task, first_topic, max_pages=getattr(args, "max_web_pages", 1))

        if getattr(args, "fast_preprocess", False):
            restructured = pre_process_input_fast(cli_task)
            tag = "pre-processor:fast"
        else:
            pp_model = (getattr(args, "pre_processor_model", "") or settings.get("pre_processor_model")
                        or getattr(args, "model", settings.get("model_a", DEFAULT_MODEL_A)))
            pp_url = getattr(args, "ollama_url", OLLAMA_URL)
            restructured = pre_process_input(cli_task, pp_model, pp_url)
            tag = "pre-processor"
        if restructured and restructured != cli_task:
            print(f"[{tag}] → {restructured[:200]}{'...' if len(restructured) > 200 else ''}")
            # patch args.task with restructured version
            args.task = restructured

    if getattr(args, "recommend_techniques", False) and getattr(args, "task", ""):
        techniques = recommend_techniques(args.task)
        matched = match_topic(args.task)
        origin = f" (topic: {matched['label']})" if matched is not None else ""
        print(f"[techniques] auto-recommended{origin}: {','.join(str(t) for t in techniques)}")

    try:
        if args.command == "generate":
            topics_raw = "\n".join(args.topics) if args.topics else ""
            result = generate_multi_topics(
                model=args.model,
                user_input=args.task,
                topics_raw=topics_raw,
                temperature=args.temperature,
                use_memory=use_memory,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                techniques=techniques,
                use_web=use_web,
                mode=mode,
                max_web_pages=args.max_web_pages,
                draft=args.draft,
                summarize_web_pages=args.summarize_web_pages,
                web_context_override=deep_research_context,
            )
            if args.output:
                Path(args.output).write_text(result, encoding="utf-8")
                print(f"Output written to {args.output}")
            else:
                print(result)

        elif args.command == "parallel":
            topics_raw = "\n".join(args.topics) if args.topics else ""
            out_a, out_b, fa, fb = generate_parallel_both(
                args.task, topics_raw, args.temperature, use_memory,
                args.ollama_url, args.timeout, args.model_a, args.model_b, techniques,
                use_web=use_web, mode=mode, max_web_pages=args.max_web_pages,
                draft=args.draft, summarize_web_pages=args.summarize_web_pages,
                web_context_override=deep_research_context,
            )
            print(f"Output A ({args.model_a}): outputs/{Path(fa).name}")
            print(f"Output B ({args.model_b}): outputs/{Path(fb).name}")

        elif args.command == "synthesis":
            manifest_a = args.file_a.read_text(encoding="utf-8")
            manifest_b = args.file_b.read_text(encoding="utf-8")
            synthesis, out_path = run_synthesis(
                args.task, manifest_a, manifest_b, args.temperature, use_memory,
                args.ollama_url, args.timeout, args.synthesis_model, techniques,
            )
            if args.output:
                Path(args.output).write_text(synthesis, encoding="utf-8")
                print(f"Synthesis written to {args.output}")
            else:
                print(f"Synthesis saved: outputs/{Path(out_path).name}")

        elif args.command == "full":
            topics_raw = "\n".join(args.topics) if args.topics else ""
            out_a, out_b, fa, fb, synthesis, fs = run_full_pipeline(
                args.task, topics_raw, args.temperature, use_memory,
                args.ollama_url, args.timeout, args.model_a, args.model_b,
                args.synthesis_model, techniques, use_web=use_web, mode=mode,
                max_web_pages=args.max_web_pages, draft=args.draft,
                summarize_web_pages=args.summarize_web_pages,
                web_context_override=deep_research_context,
            )
            print(f"Output A ({args.model_a}): outputs/{Path(fa).name}")
            print(f"Output B ({args.model_b}): outputs/{Path(fb).name}")
            print(f"Synthesis           : outputs/{Path(fs).name}")

        elif args.command == "memory":
            if args.mem_cmd == "view":
                print(view_memory())
            elif args.mem_cmd == "clear":
                clear_memory()
                print("Memory cleared.")

        elif args.command == "web":
            from web_server import run_web_server
            run_web_server(port=args.port, open_browser=not args.no_browser)

    except (ValueError, RuntimeError, TimeoutError) as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)


# ----------------------------------------------------------------------
# Interactive launcher
# ----------------------------------------------------------------------
SETTINGS_FILE = _DATA_DIR / "settings.json"


def load_settings() -> dict:
    defaults = {
        "model_a": DEFAULT_MODEL_A,
        "model_b": DEFAULT_MODEL_B,
        "synthesis_model": DEFAULT_SYNTH_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "timeout": DEFAULT_TIMEOUT,
        "ollama_url": OLLAMA_URL,
        "techniques": list(DEFAULT_TECHNIQUES),
        "use_web": True,
        "stream": True,
        "output_mode": "full",
        "use_pre_processor": True,
        "pre_processor_model": "",
        "max_web_pages": 1,
        "draft_mode": False,
        "use_result_cache": True,
        "preprocessor_mode": "llm",
        "anonymize_pii": False,
        "encrypt_memory": False,
        "summarize_web_pages": False,
        "backend_type": "ollama",
    }
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            defaults.update(saved)
        except (json.JSONDecodeError, Exception):
            pass

    # Type-coerce every field so a corrupt settings.json never breaks the session
    try:
        defaults["temperature"] = float(defaults["temperature"])
        if not 0.0 <= defaults["temperature"] <= 2.0:
            defaults["temperature"] = DEFAULT_TEMPERATURE
    except (TypeError, ValueError):
        defaults["temperature"] = DEFAULT_TEMPERATURE

    try:
        defaults["timeout"] = int(defaults["timeout"])
        if defaults["timeout"] < 10:
            defaults["timeout"] = DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        defaults["timeout"] = DEFAULT_TIMEOUT

    try:
        defaults["max_web_pages"] = int(defaults["max_web_pages"])
        if not 0 <= defaults["max_web_pages"] <= 10:
            defaults["max_web_pages"] = 1
    except (TypeError, ValueError):
        defaults["max_web_pages"] = 1

    if defaults.get("preprocessor_mode") not in ("llm", "fast"):
        defaults["preprocessor_mode"] = "llm"

    if defaults.get("backend_type") not in ("ollama", "openai_compatible"):
        defaults["backend_type"] = "ollama"

    if not isinstance(defaults["model_a"], str) or not defaults["model_a"].strip():
        defaults["model_a"] = DEFAULT_MODEL_A
    if not isinstance(defaults["model_b"], str) or not defaults["model_b"].strip():
        defaults["model_b"] = DEFAULT_MODEL_B
    if not isinstance(defaults["synthesis_model"], str) or not defaults["synthesis_model"].strip():
        defaults["synthesis_model"] = DEFAULT_SYNTH_MODEL

    if not isinstance(defaults["techniques"], list):
        defaults["techniques"] = list(DEFAULT_TECHNIQUES)
    else:
        valid = [int(t) for t in defaults["techniques"] if str(t).lstrip("-").isdigit() and int(t) > 0]
        defaults["techniques"] = valid if valid else list(DEFAULT_TECHNIQUES)

    defaults["use_web"] = bool(defaults.get("use_web", True))
    defaults["stream"] = bool(defaults.get("stream", True))

    if defaults.get("output_mode") not in ("quick", "full"):
        defaults["output_mode"] = "full"

    defaults["use_pre_processor"] = bool(defaults.get("use_pre_processor", True))
    pp_m = defaults.get("pre_processor_model", "")
    defaults["pre_processor_model"] = sanitize_input(str(pp_m).strip(), "model") if pp_m else ""

    return defaults


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def prompt_input(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val if val else default


def prompt_int(label: str, default: int) -> int:
    val = input(f"  {label} [{default}]: ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"    Invalid value, using {default}")
        return default


def prompt_float(label: str, default: float) -> float:
    val = input(f"  {label} [{default}]: ").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"    Invalid value, using {default}")
        return default


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def print_banner(settings: dict):
    inet = "ON" if check_internet() else "OFF"
    web_st = "ON" if settings.get("use_web", True) and check_internet() else "OFF"
    stream_st = "ON" if settings.get("stream", True) else "OFF"
    print("=" * 62)
    print("   WILD_ROOT_PROMPT  —  Expert Prompt Enhancement Tool  v2.3")
    print("=" * 62)
    print(f"   Model A     : {settings['model_a']}")
    print(f"   Model B     : {settings['model_b']}")
    print(f"   Synthesis   : {settings['synthesis_model']}")
    print(f"   Temperature : {settings['temperature']}")
    mode_label = "QUICK (enhanced prompt)" if settings.get("output_mode") == "quick" else "FULL (12-section manifest)"
    print(f"   Techniques  : {len(settings['techniques'])} active / {len(TECHNIQUES_DB)} available")
    print(f"   Output mode : {mode_label}")
    use_pp = settings.get("use_pre_processor", True)
    pp_model_disp = settings.get("pre_processor_model") or settings.get("model_a", DEFAULT_MODEL_A)
    pp_label = f"ON   ({pp_model_disp})" if use_pp else "OFF"
    print(f"   Pre-process : {pp_label}")
    print(f"   Internet    : {inet}   Web enrichment : {web_st}   Streaming : {stream_st}")
    print("-" * 62)


def print_menu():
    print()
    print("  1.  Single generation         (1 model, streaming)")
    print("  2.  Parallel generation       (2 models, split screen)")
    print("  3.  Full pipeline             (parallel + synthesis)")
    print("  4.  Synthesize 2 files")
    print()
    print("  5.  Configure models")
    print("  6.  Configure techniques")
    print("  7.  Browse available techniques")
    print("  8.  Advanced settings          (temperature, timeout, url, web, stream)")
    print()
    print("  9.  View memory")
    print("  10. Clear memory")
    print()
    print("  h.  Help — metacommands & usage")
    print("  0.  Quit")
    print()


def print_help():
    print()
    print("  " + "=" * 58)
    print("  WILD_ROOT_PROMPT — Help")
    print("  " + "=" * 58)
    print()
    print("  HOW IT WORKS")
    print("  Enter your prompt in STEP 1. The tool enhances it using")
    print("  173 prompt-engineering techniques and sends the result to")
    print("  a local Ollama model. Output is saved to outputs/")
    print()
    print("  /SLASH METACOMMANDS  (type them in STEP 1 or STEP 2)")
    print("  ─" * 29)
    print("  Persona  : /expert /humain /enfant /philosophe /sceptique")
    print("             /mentor /cynique /serieux /humour /emphatique")
    print("  Format   : /tableau /json /markdown /points /checklist /code")
    print("  Depth    : /resume /concis /detaille /urgent /silence")
    print("             /minimal /limite:N  (e.g. /limite:200)")
    print("  Reasoning: /raisonnement /etapes /exemple /analogie")
    print("             /pourcontre /debat /reverse /iterer /ameliorer")
    print("             /critique /audit /comparatif")
    print("  Quality  : /precision /hypotheses /sources /risques")
    print("             /decision /verification /confiance /questions")
    print("             /sansbuzz")
    print("  Context  : /historique /futuriste /questionner /neuf")
    print("             /priorite /niveau:X  (debutant|intermediaire|expert)")
    print()
    print("  EXAMPLES")
    print("  > /expert /raisonnement Build a REST API in Python")
    print("  > /enfant /analogie Explain quantum computing")
    print("  > /json /critique Analyze this architecture: [description]")
    print()
    print("  TIPS FOR NOOBS")
    print("  - STEP 2 and STEP 3 are optional — just press ENTER to skip")
    print("  - STEP 4 shows your local models — pick by number")
    print("  - Option 6 lets you change which techniques are applied")
    print("  - Outputs are saved in outputs/ as .md files")
    print()
    print("  " + "=" * 58)
    print()
    input("  Press ENTER to return to menu...")


_METHOD_GROUPS = [
    ("Persona ",  "/expert /humain /enfant /philosophe /sceptique /mentor /cynique /serieux /humour /emphatique"),
    ("Format  ",  "/tableau /json /markdown /points /checklist /code"),
    ("Depth   ",  "/resume /concis /detaille /urgent /silence /minimal /limite:N"),
    ("Reason  ",  "/raisonnement /etapes /exemple /analogie /pourcontre /debat /reverse /iterer /ameliorer /critique /audit /comparatif"),
    ("Quality ",  "/precision /hypotheses /sources /risques /decision /verification /confiance /questions /sansbuzz"),
    ("Context ",  "/historique /futuriste /questionner /neuf /priorite /niveau:X"),
]

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9._:/\-]+$")
_MAX_TASK_LEN  = 8000
_MAX_TOPIC_LEN = 500
_MAX_MODEL_LEN = 100


def sanitize_input(value: str, kind: str = "text") -> str:
    """Zero-trust sanitizer for all user-provided strings.

    kind='text'  — task / topic text (strip null bytes, control chars, cap length)
    kind='model' — model name (allow only safe identifier chars, cap at 100)
    kind='url'   — ollama URL (must be http/https, host localhost/127.0.0.1 only)
    """
    if not isinstance(value, str):
        return ""
    # Strip null bytes and ASCII control characters (keep \n \t for text)
    value = value.replace("\x00", "")
    if kind != "text":
        value = re.sub(r"[\x00-\x1f\x7f]", "", value)

    if kind == "model":
        value = value.strip()[:_MAX_MODEL_LEN]
        if not _MODEL_NAME_RE.match(value):
            # Remove any character that doesn't match the allowed set
            value = re.sub(r"[^a-zA-Z0-9._:/\-]", "", value)
        return value

    if kind == "url":
        value = value.strip()
        try:
            p = urlparse(value)
            if p.scheme not in ("http", "https"):
                return ""
            if p.hostname not in ("localhost", "127.0.0.1", "::1"):
                return ""
        except Exception:
            return ""
        return value

    # kind == "text" (default) — always allow up to task max; callers slice further if needed
    value = value[:_MAX_TASK_LEN]
    return value.strip()


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum — used to avoid redacting arbitrary long digit
    runs (IDs, version strings) as if they were credit card numbers."""
    digits = re.sub(r"[ \-]", "", digits)
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_PII_STRUCTURED_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    ("EMAIL", re.compile(r"(?<![/@#])\b[A-Za-z0-9._%+\-]{2,}@[A-Za-z0-9.\-]{2,}\.[A-Za-z]{2,}\b")),
    ("IPV6", re.compile(r"(?<![:/\w])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![:/\w])")),
    ("IPV4", re.compile(r"\b(?!127\.|0\.0\.0\.0|10\.|172\.(?:1[6-9]|2\d|3[01])\.|192\.168\.)(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("MAC_ADDRESS", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?<!\d\.)(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3}[\s.\-]\d{3,4}[\s.\-]?\d{0,4}(?!\d)(?!\.\d)")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ \-]?){13,19}\b")),
]

_PII_NAME_TRIGGER_PATTERNS = [
    # (?i:...) scopes case-insensitivity to the trigger phrase ONLY — the name
    # capture group must stay case-SENSITIVE, since [A-ZÀ-Ý] is what tells a
    # real capitalized name apart from an ordinary lowercase word like "et"/"and".
    re.compile(r"\b((?i:my name is|i am|i'm))\s+([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+){0,2})"),
    re.compile(r"\b((?i:je m'appelle|je suis))\s+([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+){0,2})"),
]


def anonymize_pii(text: str) -> Tuple[str, List[str]]:
    """Redact common PII categories from text via regex — no ML dependency,
    opt-in only. Returns (redacted_text, labels_found).

    Conservative by design: only flags structured formats (email, phone, IP,
    MAC, SSN, Luhn-valid card numbers) plus explicit self-identification
    phrases ("my name is X", "je m'appelle X") — not a general named-entity
    recognizer, so it will miss names/PII mentioned without those triggers.
    """
    if not text:
        return text, []
    redacted = text
    labels_found: List[str] = []
    for label, pattern in _PII_STRUCTURED_PATTERNS:
        if label == "CREDIT_CARD":
            def _redact_card(m, _label=label):
                if _luhn_valid(m.group(0)):
                    labels_found.append(_label)
                    return f"[REDACTED_{_label}]"
                return m.group(0)
            redacted = pattern.sub(_redact_card, redacted)
            continue
        if pattern.search(redacted):
            labels_found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    for pattern in _PII_NAME_TRIGGER_PATTERNS:
        def _redact_name(m):
            return f"{m.group(1)} [REDACTED_NAME]"
        new_redacted = pattern.sub(_redact_name, redacted)
        if new_redacted != redacted:
            labels_found.append("NAME")
            redacted = new_redacted
    return redacted, labels_found


_PROMPT_INJECTION_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    ("IGNORE_INSTRUCTIONS", re.compile(r"(?i)\b(ignore|disregard)\s+(all\s+|everything\s+)?(previous|prior|above|preceding)\s+(instructions?|prompts?|rules?)\b")),
    ("IGNORE_INSTRUCTIONS_FR", re.compile(r"(?i)\b(ignore[rz]?|oublie[rz]?|disregarde[rz]?)\s+(toutes?\s+)?(les|tes|ces)\s+instructions?\s*(precedentes?|pr[ée]c[ée]dentes?|ci-dessus)?\b")),
    ("FORGET_INSTRUCTIONS", re.compile(r"(?i)\bforget\s+(everything|all|your\s+(instructions|training|rules))\b")),
    ("ROLE_OVERRIDE", re.compile(r"(?i)\byou\s+are\s+now\s+(a|an|no\s+longer)\b")),
    ("REVEAL_SYSTEM_PROMPT", re.compile(r"(?i)\b(reveal|show|print|repeat)\s+(your|the)\s+(system\s+prompt|instructions|initial\s+prompt)\b")),
    ("NO_RESTRICTIONS", re.compile(r"(?i)\b(act as if|pretend)\s+you\s+have\s+no\s+(restrictions|rules|limits|guidelines)\b")),
    ("JAILBREAK_TOKEN", re.compile(r"(?i)\bDAN\b|\bjailbreak(ed)?\b|do\s+anything\s+now\b")),
    ("FAKE_ROLE_TAG", re.compile(r"(?i)<\|im_(start|end)\|>|\[/?\s*(system|assistant|user)\s*\]")),
]


def detect_prompt_injection(text: str) -> List[str]:
    """Heuristic detection of prompt-injection / role-override attempts in
    raw user input (e.g. "ignore previous instructions", fake chat-template
    role tags). This is a defense-in-depth SIGNAL, not a hard block: it also
    fires on legitimate discussion of these topics (e.g. security research
    about jailbreaks), so callers should warn the user, not silently strip
    or reject the input. Returns the list of matched category labels."""
    if not text:
        return []
    labels: List[str] = []
    for label, pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            labels.append(label)
    return labels


def _detect_input_language(text: str) -> str:
    """Lightweight heuristic to detect French vs English for prompt injection."""
    t = text.lower()
    fr_markers = {
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
        "je", "tu", "il", "nous", "vous", "ils", "que", "qui", "est",
        "pas", "pour", "sur", "avec", "dans", "par", "mais", "comment",
        "aide", "moi", "mon", "ton", "son", "très", "être", "faire",
        "avoir", "veux", "peux", "dois", "faut", "quoi", "comment",
        "expliquer", "explique", "besoin", "vouloir", "cest", "jsuis",
    }
    en_markers = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "do", "does", "did", "will", "would", "can",
        "could", "should", "this", "that", "with", "from", "they",
        "their", "your", "how", "what", "when", "where", "why", "who",
        "i", "my", "me", "we", "you", "help", "make", "write", "build",
        "explain", "design", "implement", "create", "describe",
    }
    import re as _re
    words = set(_re.sub(r"[^\w\s]", " ", t).split())
    fr_score = len(words & fr_markers)
    en_score = len(words & en_markers)
    if fr_score > en_score:
        return "French"
    return "English"


_KEYWORD_STOPWORDS = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "with", "as", "by",
    "at", "this", "that", "these", "those", "it", "its", "from", "into",
    "about", "than", "then", "so", "such", "not", "no", "do", "does", "did",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "must", "you", "your", "we", "our", "they", "their", "he", "she", "his",
    "her", "him", "them", "me", "my", "if", "when", "where", "which", "who",
    "whom", "what", "how", "why", "have", "has", "had", "want", "need",
    "like", "make", "get", "please", "some", "any", "all", "each", "just",
    # French
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "je",
    "tu", "il", "elle", "nous", "vous", "ils", "elles", "que", "qui", "est",
    "pas", "pour", "sur", "avec", "dans", "par", "mais", "ou", "comme",
    "ce", "cette", "ces", "son", "sa", "ses", "mon", "ma", "mes", "ton",
    "ta", "tes", "notre", "votre", "leur", "leurs", "au", "aux", "se",
    "ne", "plus", "tout", "tous", "toute", "toutes", "être", "avoir",
    "fait", "faire", "cela", "ça", "donc", "alors", "être", "veux", "peux",
    "dois", "faut", "voudrais", "aimerais", "moi", "toi", "être", "tres",
}


def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """Lightweight frequency-based keyword extraction — no ML dependency.
    Ranks by frequency then length (longer terms tend to be more specific/
    technical), and prefers a capitalized/mixed-case rendering of each term
    when one exists in the source (likely a proper noun, acronym, or
    library/technique name worth preserving verbatim)."""
    if not text or not text.strip():
        return []
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_+#\-]{3,}", text)
    counts: Dict[str, int] = {}
    best_case: Dict[str, str] = {}
    for tok in tokens:
        low = tok.lower()
        if low in _KEYWORD_STOPWORDS:
            continue
        counts[low] = counts.get(low, 0) + 1
        if low not in best_case or (tok[:1].isupper() and not best_case[low][:1].isupper()):
            best_case[low] = tok
    if not counts:
        return []
    ranked = sorted(counts.keys(), key=lambda w: (counts[w], len(w)), reverse=True)
    return [best_case[w] for w in ranked[:max_keywords]]


_BEGINNER_LEVEL_MARKERS = {
    "eli5", "for beginners", "beginner", "explain simply", "simple terms",
    "in simple terms", "never done", "first time", "new to this",
    "not familiar", "i don't understand", "i dont understand",
    "je débute", "debutant", "débutant", "je ne comprends pas", "jamais fait",
    "nouveau dans", "pas familier", "explique simplement", "en termes simples",
    "je suis nul", "aide moi à comprendre", "c'est quoi",
}
_EXPERT_LEVEL_MARKERS = {
    "architecture", "production-grade", "production grade", "edge case",
    "edge-case", "optimiz", "optimis", "algorithm", "algorithme", "complexity",
    "complexité", "asynchron", "concurrency", "concurrence", "idempotent",
    "throughput", "latency", "latence", "scalab", "avancé", "expert",
    "benchmark", "profiling", "distributed", "distribué", "microservice",
    "orchestration", "thread-safe", "threadsafe",
}


def detect_user_level(text: str) -> str:
    """Heuristic beginner/intermediate/expert classification from vocabulary
    and sentence complexity — no ML dependency. Intended only as an
    auto-detected default: an explicit /niveau:X metacommand always overrides
    this (checked by the caller before using the result)."""
    if not text or not text.strip():
        return "intermediate"
    low = text.lower()
    beginner_hits = sum(1 for m in _BEGINNER_LEVEL_MARKERS if m in low)
    expert_hits = sum(1 for m in _EXPERT_LEVEL_MARKERS if m in low)

    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    avg_sentence_len = (len(words) / len(sentences)) if sentences else len(words)

    score = expert_hits - beginner_hits
    if avg_word_len >= 6.5 and len(words) >= 12:
        score += 1
    if avg_word_len <= 4.2 and avg_sentence_len <= 8:
        score -= 1

    if beginner_hits > 0 and expert_hits == 0:
        return "beginner"
    if score >= 2:
        return "expert"
    if score <= -1:
        return "beginner"
    return "intermediate"


_TECHNIQUE_RECOMMENDATION_TRIGGERS: Dict[str, set] = {
    "Tache analytique precise": {
        "analyze", "analysis", "analytic", "precise", "data", "metric", "metrics",
        "statistic", "statistics", "rigorous", "rigoureux", "données", "analytique", "chiffres",
    },
    "Exploration creatrice": {
        "creative", "brainstorm", "explore", "exploration", "imaginative", "créatif",
        "créative", "idées", "imaginer", "inventer", "concept",
    },
    "Generation de code robuste": {
        "code", "script", "function", "api", "program", "programme", "implement",
        "coding", "développer", "fonction", "robuste", "pipeline", "algorithm",
        "algorithme", "bug", "debug", "refactor",
    },
    "Decision critique": {
        "decide", "decision", "critical", "choice", "choose", "tradeoff", "décision",
        "critique", "choix", "arbitrage", "prioritize", "priorité",
    },
    "Apprentissage / explication": {
        "explain", "learn", "teach", "understand", "tutorial", "expliquer",
        "apprendre", "comprendre", "cours", "pédagogique", "beginner", "débutant",
    },
    "Audit / securite": {
        "audit", "security", "vulnerability", "vulnerabilities", "secure", "risk",
        "sécurité", "vulnérabilité", "risque", "pentest", "faille", "cve",
    },
    "Synthese de document long": {
        "summarize", "summary", "document", "report", "synthèse", "résumer",
        "rapport", "synthesis", "long", "digest",
    },
    "Output parseable / API": {
        "json", "schema", "parse", "parsing", "structured", "endpoint", "rest", "api",
    },
    "Deblocage creatif": {
        "stuck", "blocked", "unblock", "bloqué", "coincé", "débloquer", "panne",
    },
    "Recherche de qualite maximale": {
        "research", "thorough", "exhaustive", "recherche", "approfondi",
        "maximal", "maximum", "comprehensive",
    },
}


def match_topic(text: str) -> Optional[Dict]:
    """First-pass topic matching against topic_index triggers. Returns the
    best-scoring topic entry, or None when the index is absent or no trigger
    fires. Ties resolve to catalogue order so the result is deterministic."""
    if not TOPIC_INDEX or not text or not text.strip():
        return None
    low = text.lower()
    best: Optional[Dict] = None
    best_score = 0
    for topic in TOPIC_INDEX:
        hits = sum(1 for trigger in topic.get("triggers", []) if trigger in low)
        if hits > best_score:
            best, best_score = topic, hits
    return best


def recommend_techniques(text: str, max_techniques: int = 8) -> List[int]:
    """Per-task technique selection — no ML dependency. First pass: match the
    text against the topic_index and return the matched topic's core set
    (5-8 mutually coherent techniques). Legacy fallback when the index is
    absent or silent: score QUICK_REFERENCE buckets by keyword overlap.
    Falls back to DEFAULT_TECHNIQUES when nothing matches."""
    topic = match_topic(text)
    if topic is not None:
        core = sorted({tid for tid in topic.get("core", []) if tid in TECHNIQUES_DB})
        if core:
            return core
    if not QUICK_REFERENCE or not text or not text.strip():
        return list(DEFAULT_TECHNIQUES)
    low = text.lower()
    scores: Dict[str, int] = {}
    for bucket, triggers in _TECHNIQUE_RECOMMENDATION_TRIGGERS.items():
        if bucket not in QUICK_REFERENCE:
            continue
        hits = sum(1 for t in triggers if t in low)
        if hits:
            scores[bucket] = hits
    if not scores:
        return list(DEFAULT_TECHNIQUES)
    ranked_buckets = sorted(scores.keys(), key=lambda b: scores[b], reverse=True)
    result: List[int] = []
    for bucket in ranked_buckets:
        for tid in QUICK_REFERENCE[bucket]:
            if tid not in result:
                result.append(tid)
        if len(result) >= max_techniques:
            break
    return sorted(result[:max_techniques])


def _adaptive_preprocessor_max_tokens(raw_input: str) -> int:
    """Scale the pre-processor's token budget with input length.

    PRE_PROCESSOR_PROMPT requires >=3x word growth (80-word floor) plus a
    >=3-step breakdown for build/create tasks. A fixed 600-token cap silently
    truncates that requirement once the raw input itself is more than ~80-100
    words. ~2 tokens/word covers accented French text safely; +150 tokens of
    headroom covers numbered-list scaffolding.
    """
    word_count = len(raw_input.split())
    target = int(word_count * 3 * 2) + 150
    return max(PRE_PROCESSOR_MAX_TOKENS, min(target, PRE_PROCESSOR_MAX_TOKENS_CEILING))


def _adaptive_preprocessor_timeout(max_tokens: int) -> int:
    """A fixed 30s timeout was sized for the old fixed 600-token budget. Scale
    it with the token budget (conservative ~25 tok/s floor for small local
    models on modest hardware) so larger reconstructions aren't cut off
    mid-stream — which would silently fall back to the raw, unenhanced input.
    """
    return max(PRE_PROCESSOR_TIMEOUT, int(max_tokens / 25) + 10)


def pre_process_input_fast(raw_input: str) -> str:
    """Regex-only prompt cleanup — no Ollama call, effectively instant.
    Fixes whitespace, punctuation spacing, and capitalization only. Unlike
    pre_process_input(), it does NOT restructure, expand, or semantically
    strengthen the prompt — use it for simple/short tasks or whenever speed
    matters more than depth. Returns raw_input unchanged on any failure."""
    if not raw_input or not raw_input.strip():
        return raw_input
    try:
        text = raw_input.strip()
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Drop stray spaces before punctuation, ensure one space after
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])(?=[A-Za-zÀ-ÿ])", r"\1 ", text)
        # Collapse accidental repeats of a single punctuation mark
        text = re.sub(r"([,;:])\1+", r"\1", text)
        # Capitalize the very first letter and the first letter after sentence-ending punctuation
        text = re.sub(r"(^\s*)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), text)
        text = re.sub(r"([.!?]\s+)([a-zà-ÿ])", lambda m: m.group(1) + m.group(2).upper(), text)
        # Add terminal punctuation to statements that clearly lack any
        if len(text.split()) > 3 and text and text[-1] not in ".!?\"'”)":
            text += "."
        return text
    except Exception:
        return raw_input


def pre_process_input(
    raw_input: str,
    model: str,
    ollama_url: str,
    timeout: int = PRE_PROCESSOR_TIMEOUT,
    stream_callback=None,
) -> str:
    """Restructure raw user input into a clean prompt via a fast Ollama call.
    Returns raw_input unchanged on any failure — never raises."""
    if not raw_input or not raw_input.strip():
        return raw_input
    try:
        detected_lang = _detect_input_language(raw_input.strip())
        keywords = extract_keywords(raw_input)
        keywords_block = ""
        if keywords:
            keywords_block = (
                "\nKEY TERMS DETECTED IN THE RAW INPUT (preserve these verbatim — do not "
                "drop, genericize, or mistranslate them): " + ", ".join(keywords) + "\n"
            )
        level_block = ""
        if "/niveau:" not in raw_input.lower():
            detected_level = detect_user_level(raw_input)
            if detected_level != "intermediate":
                level_block = (
                    f"\nAUTO-DETECTED AUDIENCE LEVEL: {detected_level}. Calibrate depth, "
                    f"vocabulary, and assumed knowledge for a {detected_level}-level audience "
                    "in the reconstructed prompt (this is a heuristic default — an explicit "
                    "/niveau:X from the user always takes precedence).\n"
                )
        prompt = (
            PRE_PROCESSOR_PROMPT
            .replace("{RAW_INPUT}", raw_input.strip())
            .replace("{DETECTED_LANG}", detected_lang)
            .replace("{KEYWORDS_BLOCK}", keywords_block)
            .replace("{USER_LEVEL_BLOCK}", level_block)
        )
        max_tokens = _adaptive_preprocessor_max_tokens(raw_input)
        payload = {
            "model": sanitize_input(model, "model"),
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
                "stop": ["---", "<<<", "INPUT>>>", "\n\n\n"],
            },
        }
        effective_timeout = max(timeout, _adaptive_preprocessor_timeout(max_tokens))
        resp = requests.post(
            ollama_url, json=payload, timeout=effective_timeout,
            headers={"User-Agent": HTTP_USER_AGENT}, stream=True,
        )
        result = []
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                result.append(token)
                if stream_callback:
                    stream_callback(token)
            if chunk.get("done"):
                break
        out = sanitize_input("".join(result).strip(), "text")
        return out if out else raw_input
    except Exception:
        return raw_input


def collect_input(settings: dict) -> Tuple[str, str, str]:
    """
    Four-step input: prompt → methods → topic → model.
    Returns (full_task_with_metacommands, topics_raw, model_name).
    Runs without any further prompts after this.
    """
    print()
    print("  ─" * 31)
    print("  STEP 1 — Prompt to enhance  ('templates' to list, 'template:<id>[:topic]' to use one)")
    print("  ─" * 31)
    raw_task = input("  > ").strip()

    if raw_task.lower() == "templates":
        print()
        by_category: Dict[str, List[Dict]] = {}
        for t in PROMPT_TEMPLATES.values():
            by_category.setdefault(t.get("category", "Other"), []).append(t)
        for cat, tmpls in by_category.items():
            print(f"  [{cat}]")
            for t in tmpls:
                print(f"    {t['id']:<22s} {t['title']}")
        print()
        print("  Enter 'template:<id>' or 'template:<id>:<topic>', or a regular prompt:")
        raw_task = input("  > ").strip()

    if raw_task.lower().startswith("template:"):
        parts = raw_task.split(":", 2)
        tmpl_id = parts[1].strip() if len(parts) > 1 else ""
        topic_fill = parts[2].strip() if len(parts) > 2 else ""
        tmpl = PROMPT_TEMPLATES.get(tmpl_id)
        if not tmpl:
            print(f"  [!] Unknown template '{tmpl_id}'. Type 'templates' to see available ids.")
            return "", "", ""
        raw_task = tmpl["task"].replace("[TOPIC]", topic_fill) if topic_fill else tmpl["task"]
        print(f"  [template] {tmpl['title']}: {raw_task}")

    task = sanitize_input(raw_task, "text")
    if not task:
        print("  [!] Prompt cannot be empty.")
        return "", "", ""

    print()
    print("  ─" * 31)
    print("  STEP 2 — Methods  (ENTER to skip, 'auto' to auto-select techniques, 'research' for deep research)")
    for label, cmds in _METHOD_GROUPS:
        print(f"  {label}: {cmds}")
    print("  ─" * 31)
    methods_raw = sanitize_input(input("  > ").strip(), "text")

    # Merge any /commands already in the task with those typed in step 2
    task_cmds = [t for t in task.split() if t.startswith("/")]
    task_clean = " ".join(t for t in task.split() if not t.startswith("/"))
    extra_cmds = [t for t in methods_raw.split() if t.startswith("/")]
    all_cmds = task_cmds + extra_cmds

    if all_cmds:
        print(f"  [active] {' '.join(all_cmds)}")
        full_task = " ".join(all_cmds) + " " + task_clean
    else:
        full_task = task_clean

    if settings.get("anonymize_pii", False):
        full_task, pii_labels = anonymize_pii(full_task)
        if pii_labels:
            print(f"  [privacy] Redacted before sending to any model: {', '.join(sorted(set(pii_labels)))}")

    injection_labels = detect_prompt_injection(full_task)
    if injection_labels:
        print(f"  [warning] Input resembles a prompt-injection/jailbreak attempt ({', '.join(injection_labels)}). Proceeding — advisory only.")

    auto_techniques = methods_raw.strip().lower() == "auto"
    if auto_techniques:
        print("  [techniques] auto-selection will run after the pre-processor, on the enriched text")

    if methods_raw.strip().lower() in ("research", "deep-research", "deepresearch"):
        deep_context = run_deep_research(full_task)
        if deep_context:
            settings["_deep_research_context"] = deep_context

    print()
    print("  ─" * 31)
    print("  STEP 3 — Topic focus  (ENTER to skip)")
    print("  ─" * 31)
    topics_raw = sanitize_input(input("  > ").strip(), "text")[:_MAX_TOPIC_LEN]

    print()
    print("  ─" * 31)
    print("  STEP 4 — Model")
    print("  ─" * 31)
    chosen_model = pick_model_interactive("Select model", settings.get("model_a", DEFAULT_MODEL_A))

    # ── STEP 5 — Pre-processor ──────────────────────────────────────────────
    if settings.get("use_pre_processor", True):
        pp_model = settings.get("pre_processor_model") or chosen_model
        ollama_url = settings.get("ollama_url", OLLAMA_URL)
        do_stream = settings.get("stream", True) and sys.stdout.isatty()

        if (settings.get("use_web", True) and not settings.get("summarize_web_pages", False)
                and "_deep_research_context" not in settings):
            first_topic = next(iter(split_topics(topics_raw)), "")
            prefetch_web_context(full_task, first_topic, max_pages=settings.get("max_web_pages", 1))

        fast_mode = settings.get("preprocessor_mode", "llm") == "fast"

        print()
        print("  ─" * 31)
        if fast_mode:
            print("  STEP 5 — Pre-processor  (fast regex cleanup, no Ollama call)")
        else:
            print("  STEP 5 — Pre-processor  (restructuring your prompt...)")
        print("  ─" * 31)

        if fast_mode:
            restructured = pre_process_input_fast(full_task)
            print(f"  {restructured}")
        else:
            pp_buffer: List[str] = []

            def _pp_stream(token: str):
                pp_buffer.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()

            restructured = pre_process_input(
                raw_input=full_task,
                model=pp_model,
                ollama_url=ollama_url,
                timeout=PRE_PROCESSOR_TIMEOUT,
                stream_callback=_pp_stream if do_stream else None,
            )

            if not do_stream:
                print(f"  {restructured}")
            else:
                if not pp_buffer:
                    print(f"  {restructured}")

        print()
        print()
        print("  [ENTER] Accept  |  [e] Edit  |  [s] Skip")
        try:
            confirm = input("  > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            confirm = "s"

        if confirm == "e":
            print("  Edit (ENTER to confirm):")
            try:
                edited = input(f"  > ").strip()
                full_task = sanitize_input(edited, "text") if edited else restructured
            except (KeyboardInterrupt, EOFError):
                full_task = restructured
        elif confirm == "s":
            pass
        else:
            full_task = restructured or full_task
    # ── end STEP 5 ──────────────────────────────────────────────────────────

    if auto_techniques:
        recommended = recommend_techniques(full_task)
        settings["techniques"] = recommended
        matched = match_topic(full_task)
        origin = f" (topic: {matched['label']})" if matched is not None else ""
        print(f"  [auto-selected techniques{origin}] {','.join(str(t) for t in recommended)}")
    return full_task.strip(), topics_raw, chosen_model


def menu_generate(settings: dict):
    task, topics_raw, model = collect_input(settings)
    if not task:
        return
    use_web = settings.get("use_web", True)
    do_stream = settings.get("stream", True) and sys.stdout.isatty()
    mode = settings.get("output_mode", "full")
    deep_research_context = settings.pop("_deep_research_context", None)

    print(f"\n  → Model: {model}  |  Mode: {mode.upper()}")
    if use_web and check_internet():
        print("  → Web enrichment: ON")
    if do_stream:
        print(f"\n  {'─' * 58}\n")

    def print_token(t):
        sys.stdout.write(t)
        sys.stdout.flush()

    try:
        result = generate_multi_topics(
            model=model,
            user_input=task,
            topics_raw=topics_raw,
            temperature=settings["temperature"],
            use_memory=True,
            ollama_url=settings["ollama_url"],
            timeout=settings["timeout"],
            techniques=settings["techniques"],
            use_web=use_web,
            stream_callback=print_token if do_stream else None,
            mode=mode,
            max_web_pages=settings.get("max_web_pages", 1),
            draft=settings.get("draft_mode", False),
            summarize_web_pages=settings.get("summarize_web_pages", False),
            web_context_override=deep_research_context,
        )
        if do_stream:
            print("\n")
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        out_file = OUTPUT_DIR / f"{session_id}_{model.replace(':', '_')}.md"
        out_file.write_text(result, encoding="utf-8")
        print(f"\n  Saved: outputs/{out_file.name}")
    except (ValueError, RuntimeError, TimeoutError) as e:
        print(f"\n  [ERROR] {e}")


def menu_parallel(settings: dict):
    task, topics_raw, model_a = collect_input(settings)
    if not task:
        return
    model_b = pick_model_interactive("Select model B", settings.get("model_b", DEFAULT_MODEL_B))
    use_web = settings.get("use_web", True)
    do_stream = settings.get("stream", True) and sys.stdout.isatty()
    mode = settings.get("output_mode", "full")
    deep_research_context = settings.pop("_deep_research_context", None)

    print(f"\n  → {model_a}  vs  {model_b}  |  Mode: {mode.upper()}")
    if use_web and check_internet():
        print("  → Web enrichment: ON")
    if do_stream:
        print("  → Split screen active. Ctrl+C to interrupt.\n")
        time.sleep(1)

    try:
        _, _, fa, fb = generate_parallel_both(
            user_input=task,
            topics_raw=topics_raw,
            temperature=settings["temperature"],
            use_memory=True,
            ollama_url=settings["ollama_url"],
            timeout=settings["timeout"],
            model_a=model_a,
            model_b=model_b,
            techniques=settings["techniques"],
            use_web=use_web,
            live_display=do_stream,
            mode=mode,
            max_web_pages=settings.get("max_web_pages", 1),
            draft=settings.get("draft_mode", False),
            summarize_web_pages=settings.get("summarize_web_pages", False),
            web_context_override=deep_research_context,
        )
        print(f"\n  Output A: outputs/{Path(fa).name}")
        print(f"  Output B: outputs/{Path(fb).name}")
    except (ValueError, RuntimeError, TimeoutError) as e:
        print(f"\n  [ERROR] {e}")


def menu_full(settings: dict):
    task, topics_raw, model_a = collect_input(settings)
    if not task:
        return
    model_b = pick_model_interactive("Select model B", settings.get("model_b", DEFAULT_MODEL_B))
    synth = pick_model_interactive("Select synthesis model", settings.get("synthesis_model", DEFAULT_SYNTH_MODEL))
    use_web = settings.get("use_web", True)
    do_stream = settings.get("stream", True) and sys.stdout.isatty()
    mode = settings.get("output_mode", "full")
    deep_research_context = settings.pop("_deep_research_context", None)

    print(f"\n  → {model_a} + {model_b} → {synth}  |  Mode: {mode.upper()}")
    if use_web and check_internet():
        print("  → Web enrichment: ON")
    if do_stream:
        print("  → Phase 1: parallel split screen. Phase 2: synthesis streaming.\n")
        time.sleep(1)

    try:
        _, _, fa, fb, _, fs = run_full_pipeline(
            user_input=task,
            topics_raw=topics_raw,
            temperature=settings["temperature"],
            use_memory=True,
            ollama_url=settings["ollama_url"],
            timeout=settings["timeout"],
            model_a=model_a,
            model_b=model_b,
            synthesis_model=synth,
            techniques=settings["techniques"],
            use_web=use_web,
            live_display=do_stream,
            stream=do_stream,
            mode=mode,
            max_web_pages=settings.get("max_web_pages", 1),
            draft=settings.get("draft_mode", False),
            summarize_web_pages=settings.get("summarize_web_pages", False),
            web_context_override=deep_research_context,
        )
        print(f"\n  Output A   : outputs/{Path(fa).name}")
        print(f"  Output B   : outputs/{Path(fb).name}")
        print(f"  Synthesis  : outputs/{Path(fs).name}")
    except (ValueError, RuntimeError, TimeoutError) as e:
        print(f"\n  [ERROR] {e}")


def menu_synthesis(settings: dict):
    print()
    existing = sorted(OUTPUT_DIR.glob("*.md"))
    if not existing:
        print("  No files in outputs/. Run a generation first.")
        return
    print("  Available files in outputs/:")
    for i, f in enumerate(existing, 1):
        print(f"    {i:3d}. {f.name}")
    print()
    try:
        idx_a = int(input("  File A number: ").strip()) - 1
        idx_b = int(input("  File B number: ").strip()) - 1
        file_a = existing[idx_a]
        file_b = existing[idx_b]
    except (ValueError, IndexError):
        print("  [!] Invalid selection.")
        return
    task = sanitize_input(input("  Original task (short description):\n  > ").strip(), "text")
    if not task:
        task = "(not provided)"
    synth_model = pick_model_interactive("Synthesis model", settings["synthesis_model"])
    do_stream = settings.get("stream", True) and sys.stdout.isatty()
    print(f"\n  Synthesizing {file_a.name} + {file_b.name} with {synth_model} ...")
    try:
        manifest_a = file_a.read_text(encoding="utf-8")
        manifest_b = file_b.read_text(encoding="utf-8")
        _, out_path = run_synthesis(
            user_input=task,
            manifest_a=manifest_a,
            manifest_b=manifest_b,
            temperature=settings["temperature"],
            use_memory=True,
            ollama_url=settings["ollama_url"],
            timeout=settings["timeout"],
            synthesis_model=synth_model,
            techniques=settings["techniques"],
            stream=do_stream,
        )
        print(f"\n  Synthesis saved: outputs/{Path(out_path).name}")
    except (ValueError, RuntimeError, TimeoutError) as e:
        print(f"\n  [ERROR] {e}")


def menu_configure_models(settings: dict):
    print()
    print("  -- Default model configuration --")
    print(f"  Current: A={settings['model_a']}  B={settings['model_b']}  Synth={settings['synthesis_model']}")
    settings["model_a"] = pick_model_interactive("Model A (generation / single)", settings["model_a"])
    settings["model_b"] = pick_model_interactive("Model B (parallel)", settings["model_b"])
    settings["synthesis_model"] = pick_model_interactive("Synthesis model", settings["synthesis_model"])
    save_settings(settings)
    print("\n  Models saved.")


def menu_configure_techniques(settings: dict):
    print()
    print("  -- Technique configuration --")
    current = ",".join(str(t) for t in settings["techniques"])
    print(f"  Active: {current}")
    print()
    total = len(TECHNIQUES_DB)
    print("  Input examples:")
    print("    1,5,8,10,25     ->  specific techniques")
    print("    1-20             ->  range")
    print("    1-10,25,40-50    ->  combination")
    print("    default          ->  default set")
    print("    random           ->  random subset (creative exploration)")
    print("    random:N         ->  random subset of exactly N techniques")
    if QUICK_REFERENCE:
        print("    bundle:<# or name> -> a pre-configured bundle for a task type:")
        for i, task_type in enumerate(QUICK_REFERENCE.keys(), 1):
            print(f"      {i}. {task_type}")
        print("    (example: bundle:3)")
    print()
    raw = input("  Techniques: ").strip()
    if not raw:
        return
    if raw.lower() == "default":
        settings["techniques"] = list(DEFAULT_TECHNIQUES)
    else:
        settings["techniques"] = parse_techniques(raw)
    save_settings(settings)
    print(f"\n  {len(settings['techniques'])} techniques active.")


def _paged_print(lines: List[str], page_size: int = 22):
    """Print lines with ENTER-to-continue paging."""
    for i, line in enumerate(lines):
        print(line)
        if (i + 1) % page_size == 0 and i < len(lines) - 1:
            try:
                cont = input("  -- more -- (ENTER to continue, q to stop) ").strip().lower()
                if cont == "q":
                    return
            except (KeyboardInterrupt, EOFError):
                return


def menu_list_techniques():
    lines = []
    lines.append("")
    lines.append("  Available Prompt Engineering Techniques:")
    lines.append("  " + "=" * 62)
    if CATEGORIES_DB:
        for cat in CATEGORIES_DB:
            lines.append(f"\n  [{cat['id'].upper()}] {cat['name']}")
            lines.append(f"  {cat.get('description', '')}")
            lines.append("  " + "-" * 58)
            for tech in cat.get("techniques", []):
                lines.append(f"  {tech['id']:3d}. {tech['title']}")
                lines.append(f"       {tech['description']}")
    else:
        for tid in sorted(TECHNIQUES_DB.keys()):
            tech = TECHNIQUES_DB[tid]
            lines.append(f"  {tid:3d}. {tech['title']}")
            lines.append(f"       {tech['description']}")
    if ANTI_PATTERNS:
        lines.append(f"\n  [ANTI-PATTERNS] Common mistakes to avoid")
        lines.append("  " + "-" * 58)
        for ap in ANTI_PATTERNS:
            lines.append(f"  ! {ap['name']}: {ap['symptom']}")
            lines.append(f"    Fix: {ap['fix']}")
    if QUICK_REFERENCE:
        lines.append(f"\n  [QUICK REFERENCE] Recommended techniques by task type")
        lines.append("  " + "-" * 58)
        for task_type, ids in QUICK_REFERENCE.items():
            id_str = ",".join(str(i) for i in ids)
            lines.append(f"  {task_type}: {id_str}")
    _paged_print(lines)
    print()
    input("  Press ENTER to return to menu...")


def prompt_bool(label: str, default: bool) -> bool:
    dstr = "yes" if default else "no"
    val = input(f"  {label} [{dstr}] (yes/no): ").strip().lower()
    if not val:
        return default
    return val in ("yes", "y", "oui", "o", "1", "true")


def menu_advanced(settings: dict):
    print()
    print("  -- Advanced settings --")
    settings["temperature"] = prompt_float("Temperature (0.0 - 1.0)", settings["temperature"])
    settings["timeout"] = prompt_int("Timeout per call (seconds)", settings["timeout"])
    print()
    print("  Backend:")
    print("    ollama            —  native Ollama API (default)")
    print("    openai_compatible —  LM Studio / GPT4All server mode / text-generation-webui's OpenAI extension")
    raw_backend = input(f"  Backend [ollama/openai_compatible] [{settings.get('backend_type', 'ollama')}]: ").strip().lower()
    if raw_backend in ("ollama", "openai_compatible"):
        settings["backend_type"] = raw_backend
    if settings.get("backend_type", "ollama") == "openai_compatible":
        print("  Point the URL below at your server's completions endpoint, e.g.:")
        print("    http://localhost:1234/v1/completions           (LM Studio default port)")
        print("    http://localhost:5000/v1/chat/completions       (text-generation-webui)")
        print("  Note: only localhost/127.0.0.1 URLs are accepted (SSRF guard) — run the server on this machine.")
    raw_url = prompt_input("Ollama URL" if settings.get("backend_type", "ollama") == "ollama" else "Backend URL", settings["ollama_url"])
    settings["ollama_url"] = sanitize_input(raw_url, "url") or settings["ollama_url"]
    set_backend_type(settings.get("backend_type", "ollama"))
    set_backend_api_base(settings["ollama_url"])
    settings["use_web"] = prompt_bool("Automatic web enrichment", settings.get("use_web", True))
    if settings["use_web"]:
        settings["max_web_pages"] = prompt_int("Web pages to fetch full text from per search", settings.get("max_web_pages", 1))
        settings["summarize_web_pages"] = prompt_bool(
            "Summarize fetched pages via an extra Ollama call (vs. raw truncation)",
            settings.get("summarize_web_pages", False),
        )
    settings["stream"] = prompt_bool("Real-time streaming in terminal", settings.get("stream", True))
    settings["use_result_cache"] = prompt_bool(
        "Cache identical generations (skip Ollama on an exact repeat request)",
        settings.get("use_result_cache", True),
    )
    set_result_cache_enabled(settings["use_result_cache"])
    settings["anonymize_pii"] = prompt_bool(
        "Redact PII (emails/phones/IPs/names) from your input before it reaches any model",
        settings.get("anonymize_pii", False),
    )
    settings["encrypt_memory"] = prompt_bool(
        "Encrypt session memory at rest (memory/sessions.json)" +
        ("" if _FERNET_AVAILABLE else " [requires: pip install cryptography]"),
        settings.get("encrypt_memory", False),
    )
    set_memory_encryption_enabled(settings["encrypt_memory"])
    print()
    print("  Output mode:")
    print("    quick  —  Single enhanced prompt, ready to paste into any LLM")
    print("    full   —  Exhaustive 12-section instruction manifest (expert)")
    raw_mode = input(f"  Mode [quick/full] [{settings.get('output_mode','full')}]: ").strip().lower()
    if raw_mode in ("quick", "full"):
        settings["output_mode"] = raw_mode

    if settings.get("output_mode", "full") == "full":
        settings["draft_mode"] = prompt_bool(
            "Draft mode (only §1-2 of the manifest, for fast iteration)",
            settings.get("draft_mode", False),
        )

    settings["use_pre_processor"] = prompt_bool(
        "Pre-processor (restructure input before generation)",
        settings.get("use_pre_processor", True),
    )
    if settings["use_pre_processor"]:
        print()
        print("  Pre-processor mode:")
        print("    llm   —  Ollama call: cleans, expands, strengthens the prompt (default)")
        print("    fast  —  regex only, no Ollama call, instant — punctuation/spacing/capitalization only")
        raw_pp_mode = input(f"  Mode [llm/fast] [{settings.get('preprocessor_mode', 'llm')}]: ").strip().lower()
        if raw_pp_mode in ("llm", "fast"):
            settings["preprocessor_mode"] = raw_pp_mode

        if settings.get("preprocessor_mode", "llm") == "llm":
            print()
            print("  Pre-processor model (blank = use Model A):")
            cur_pp = settings.get("pre_processor_model") or ""
            raw_pp = input(f"  [{cur_pp or 'same as Model A'}] > ").strip()
            if raw_pp:
                settings["pre_processor_model"] = sanitize_input(raw_pp, "model")
            else:
                settings["pre_processor_model"] = ""

    save_settings(settings)
    print("\n  Settings saved.")


def interactive():
    # Suppress INFO-level log noise in the interactive terminal UI
    logging.disable(logging.INFO)

    settings = load_settings()
    set_result_cache_enabled(settings.get("use_result_cache", True))
    set_memory_encryption_enabled(settings.get("encrypt_memory", False))
    set_backend_type(settings.get("backend_type", "ollama"))
    set_backend_api_base(settings.get("ollama_url", OLLAMA_URL))
    if settings.get("backend_type", "ollama") == "ollama":
        ensure_ollama_ready()
        ensure_models_available(settings)

    _NO_PAUSE = {"0", "q", "quit", "exit", "7", "h", "help", "?"}

    while True:
        clear_screen()
        print_banner(settings)
        print_menu()

        try:
            choice = input("  Choice > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye.\n")
            break

        try:
            if choice == "1":
                menu_generate(settings)
            elif choice == "2":
                menu_parallel(settings)
            elif choice == "3":
                menu_full(settings)
            elif choice == "4":
                menu_synthesis(settings)
            elif choice == "5":
                menu_configure_models(settings)
            elif choice == "6":
                menu_configure_techniques(settings)
            elif choice == "7":
                menu_list_techniques()
            elif choice == "8":
                menu_advanced(settings)
            elif choice == "9":
                print()
                print(view_memory())
            elif choice == "10":
                clear_memory()
                print("\n  Memory cleared.")
            elif choice in ("h", "help", "?"):
                print_help()
            elif choice in ("0", "q", "quit", "exit"):
                print("\n  Goodbye.\n")
                break
            else:
                print("\n  [!] Unknown option. Type h for help.")
        except KeyboardInterrupt:
            print("\n\n  [Interrupted] Returning to menu...")

        if choice not in _NO_PAUSE:
            print()
            input("  Press ENTER to return to menu...")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--web":
            # quick shortcut: python prompt_expert_enhance.py --web [--port N]
            port = 7860
            if "--port" in sys.argv:
                try:
                    port = int(sys.argv[sys.argv.index("--port") + 1])
                except (ValueError, IndexError):
                    pass
            from web_server import run_web_server
            run_web_server(port=port, open_browser=True)
        else:
            main()
    else:
        interactive()
