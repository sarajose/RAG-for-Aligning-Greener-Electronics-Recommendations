#!/usr/bin/env python3
"""
main.py - Orchestrates:
  - EU evidence scraping+chunking (chunking_evidence.py)
  - recommendation extraction (chunking_recommendations.py)

Does NOT modify your modules. It tries to call their functions robustly.

usage: python main.py --mode recs --recs-dir "data/recommendations"
"""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import chunking_evidence
import chunking_recommendations


# --- Your EU evidence URLs (edit/extend as needed) ---
EU_URLS = [
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32011L0065",   # RoHS
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32012L0019",   # WEEE
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401781",     # ESPR 2024/1781
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:52020DC0098",  # CEAP 2020
    "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:52019DC0640",  # Green Deal 2019
]


# ----------------------------
# IO helpers
# ----------------------------
def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    if isinstance(x, (str, bytes)):
        return [x]
    if isinstance(x, Iterable):
        return list(x)
    return [x]


def write_jsonl(rows: List[Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, dict):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps({"text": str(r)}, ensure_ascii=False) + "\n")


# ----------------------------
# Function detection + calling
# ----------------------------
def list_module_functions(mod) -> List[str]:
    out = []
    for name, obj in inspect.getmembers(mod):
        if inspect.isfunction(obj):
            out.append(name)
    return sorted(out)


def score_fn(fn: Callable, keywords: List[str], param_hints: List[str]) -> int:
    """Higher score = more likely to be the correct entrypoint."""
    name = fn.__name__.lower()
    score = 0
    for kw in keywords:
        if kw in name:
            score += 5
    try:
        sig = inspect.signature(fn)
        params = [p.name.lower() for p in sig.parameters.values()]
        for ph in param_hints:
            if ph in params:
                score += 3
    except Exception:
        pass
    return score


def auto_pick_entry(mod, preferred: Optional[str], keywords: List[str], param_hints: List[str]) -> Callable:
    if preferred:
        if not hasattr(mod, preferred):
            raise RuntimeError(
                f"'{preferred}' not found in {mod.__name__}. "
                f"Available: {', '.join(list_module_functions(mod))}"
            )
        fn = getattr(mod, preferred)
        if not inspect.isfunction(fn):
            raise RuntimeError(f"'{preferred}' in {mod.__name__} is not a function.")
        return fn

    candidates: List[Tuple[int, Callable]] = []
    for name, obj in inspect.getmembers(mod):
        if inspect.isfunction(obj):
            candidates.append((score_fn(obj, keywords, param_hints), obj))

    candidates.sort(key=lambda t: t[0], reverse=True)
    if not candidates or candidates[0][0] == 0:
        raise RuntimeError(
            f"Could not auto-detect an entry function in {mod.__name__}. "
            f"Available: {', '.join(list_module_functions(mod))}\n"
            f"Pass it explicitly with --evidence-fn / --recs-fn."
        )
    return candidates[0][1]


def try_call(fn: Callable, **kwargs) -> Any:
    """Call fn with only the kwargs it accepts."""
    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    # Only call if we have arguments that match, or if function has no required params
    if accepted or not any(p.default == inspect.Parameter.empty for p in sig.parameters.values()):
        return fn(**accepted)
    raise TypeError(f"No matching parameters found for {fn.__name__}")


def call_entrypoint(fn: Callable, urls: Optional[List[str]] = None, rec_inputs: Optional[List[str]] = None, outdir: Optional[Path] = None) -> Any:
    """
    Tries common signature patterns without requiring changes to your modules.
    """
    # Positional fallbacks for functions that don't accept keywords
    if rec_inputs is not None:
        try:
            if outdir is not None:
                return fn(rec_inputs, str(outdir))
            return fn(rec_inputs)
        except TypeError:
            pass

    if urls is not None:
        try:
            if outdir is not None:
                return fn(urls, str(outdir))
            return fn(urls)
        except TypeError:
            pass
    # Common argument names used in these pipelines
    attempts = []
    outdir_arg = {"outdir": str(outdir) if outdir else None}

    if rec_inputs is not None:
        attempts.extend([
            {"pdf_paths": rec_inputs, **outdir_arg},
            {"pdfs": rec_inputs, **outdir_arg},
            {"paths": rec_inputs, **outdir_arg},
            {"input_paths": rec_inputs, **outdir_arg},
            {"input_dir": str(Path(rec_inputs[0]).parent) if rec_inputs else None, **outdir_arg},
        ])

    if urls is not None:
        attempts.extend([
            {"urls": urls, **outdir_arg},
            {"url_list": urls, **outdir_arg},
            {"sources": urls, **outdir_arg},
            {"links": urls, **outdir_arg},
            {"inputs": urls, **outdir_arg},
        ])

    # fallback: outdir only, then no args
    attempts.append(outdir_arg)
    attempts.append({})

    last_err: Optional[Exception] = None
    for kw in attempts:
        # drop None values so we don't pass them unnecessarily
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            return try_call(fn, **kw)
        except TypeError as e:
            last_err = e
        except Exception as e:
            # If function ran but failed internally, surface that (it’s real)
            raise

    raise RuntimeError(f"Could not call {fn.__name__} with common signatures. Last error: {last_err}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["evidence", "recs", "all"], default="all")
    ap.add_argument("--outdir", default="out", help="Output folder (JSONL written here if returned data exists)")
    ap.add_argument("--evidence-fn", default=None, help="Exact function name in chunking_evidence.py to call")
    ap.add_argument("--recs-fn", default=None, help="Exact function name in chunking_recommendations.py to call")

    # optional: recommendations input files/dir (if your rec pipeline expects PDFs)
    ap.add_argument("--recs-dir", default=None, help="Directory containing recommendation source files (e.g., PDFs)")
    ap.add_argument("--recs-files", nargs="*", default=None, help="Explicit list of recommendation files")

    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent
    outdir = (repo_root / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {"mode": args.mode, "outdir": str(outdir)}

    # ---- Evidence (EU URLs) ----
    if args.mode in ("evidence", "all"):
        ev_fn = auto_pick_entry(
            chunking_evidence,
            preferred=args.evidence_fn,
            keywords=["evidence", "eurlex", "chunk", "scrape", "build", "corpus", "run", "main"],
            param_hints=["urls", "url_list", "sources", "links", "outdir"],
        )

        print(f"[evidence] Using function: chunking_evidence.{ev_fn.__name__}()")
        ev_result = call_entrypoint(ev_fn, urls=EU_URLS, outdir=outdir / "evidence")

        ev_rows = ensure_list(ev_result)
        # If your function returns chunks, we persist them; if it returns nothing (writes files itself), we just note it.
        if ev_rows:
            ev_out = outdir / "evidence" / "evidence_corpus.jsonl"
            write_jsonl(ev_rows, ev_out)
            manifest["evidence"] = {"fn": ev_fn.__name__, "returned_rows": len(ev_rows), "written": str(ev_out)}
            print(f"[evidence] Wrote: {ev_out} ({len(ev_rows)} rows)")
        else:
            manifest["evidence"] = {"fn": ev_fn.__name__, "returned_rows": 0, "written": None}
            print("[evidence] Function returned no rows (it may have written files internally).")

    # ---- Recommendations (your “actions” to map) ----
    if args.mode in ("recs", "all"):
        rec_inputs: List[str] = []
        if args.recs_files:
            rec_inputs = [str(Path(p).resolve()) for p in args.recs_files]
        elif args.recs_dir:
            # Don’t assume extension; pass directory and let your function decide
            # but also gather PDFs as a common case.
            p = Path(args.recs_dir)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            rec_inputs = [str(x) for x in sorted(p.glob("*.pdf"))] or [str(p)]
        else:
            # Default to data/recommendations when present
            default_dir = repo_root / "data" / "recommendations"
            if default_dir.exists():
                rec_inputs = [str(x) for x in sorted(default_dir.glob("*.pdf"))] or [str(default_dir)]

        rec_fn = auto_pick_entry(
            chunking_recommendations,
            preferred=args.recs_fn,
            keywords=["recommend", "recs", "chunk", "extract", "build", "dataset", "run", "main"],
            param_hints=["pdf_paths", "pdfs", "paths", "input_paths", "input_dir", "outdir"],
        )

        if not rec_inputs:
            raise RuntimeError(
                "No recommendation inputs found. Pass --recs-dir or --recs-files, "
                "or place PDFs under data/recommendations."
            )

        print(f"[recs] Using function: chunking_recommendations.{rec_fn.__name__}()")
        rec_result = call_entrypoint(rec_fn, rec_inputs=rec_inputs, outdir=outdir / "recommendations")

        rec_rows = ensure_list(rec_result)
        if rec_rows:
            rec_out = outdir / "recommendations" / "recommendations_dataset.jsonl"
            write_jsonl(rec_rows, rec_out)
            manifest["recommendations"] = {"fn": rec_fn.__name__, "returned_rows": len(rec_rows), "written": str(rec_out)}
            print(f"[recs] Wrote: {rec_out} ({len(rec_rows)} rows)")
        else:
            manifest["recommendations"] = {"fn": rec_fn.__name__, "returned_rows": 0, "written": None}
            print("[recs] Function returned no rows (it may have written files internally).")

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Manifest: {outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()