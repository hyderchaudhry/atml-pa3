"""Provided Parquet ingestion, strict alignment checks, and fixed split/order IDs.

SQLite keeps the supplied token arrays off the Python heap until one question is
processed. Ingestion transactions and a row cursor make interrupted imports resumable.
"""

import importlib.metadata
import json
import shutil
import sqlite3
import urllib.request
import zlib
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from atml.common.io import environment, read_json, sha256, write_json

from .config import DATASETS, MODEL, SEED, protocol
from .scoring import Scorer
from .selection import Pool

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"
FILES = {
    "README.MD": ("1RARGiUK_jZpOii-b8kyGLoejaldMUAa3", 4413),
    "qwen25-math/generations.parquet": ("1SaWxZ4mpycw2BWZ5G0dahoYZ6Hh8tPWP", 26932880),
    "qwen25-math/probability_metrics.parquet": ("1RCYHrTnUKtF7gHLSTe92x6W9ZsCOekfQ", 155221097),
    "qwen25vl-mathvista/generations.parquet": ("1sedMpWLI_UgPvAQVj7tS3HYVH0ARS1yA", 17098339),
    "qwen25vl-mathvista/probability_metrics.parquet": (
        "1f2XAu_J5NviYHEhtkRrFbKc7b2us1WiW",
        81118521,
    ),
    "qwen25vl-mathvista/attention_metrics.parquet": (
        "1Nzg2VwE2qFzA10BF1lTxEcBZCePBe51q",
        138497144,
    ),
}
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}
METADATA_URL = "https://raw.githubusercontent.com/lupantech/MathVista/{revision}/data/testmini.json"


def fetch(paths):
    import gdown

    target = paths.artifacts / "supplied"
    for name, (file_id, size) in FILES.items():
        file = target / name
        if file.exists() and file.stat().st_size == size:
            continue
        file.parent.mkdir(parents=True, exist_ok=True)
        temporary = file.with_suffix(file.suffix + ".part")
        result = gdown.download(id=file_id, output=str(temporary), quiet=False, resume=True)
        if result is None or temporary.stat().st_size != size:
            raise IOError(f"Assignment data download incomplete: {name}")
        temporary.replace(file)
    return target


def fetch_metadata(paths):
    """Resolve the manual-linked metadata revision once, then retain exact bytes."""
    lock_path = paths.artifacts / "metadata/source.json"
    target = paths.artifacts / "metadata/testmini.json"
    if lock_path.exists():
        lock = read_json(lock_path)
        if not target.exists() or sha256(target) != lock["sha256"]:
            raise IOError("Released MathVista metadata is missing or changed")
        return target
    request = urllib.request.Request(
        "https://api.github.com/repos/lupantech/MathVista/commits/main",
        headers={"User-Agent": "atml-pa3-data"},
    )
    with urllib.request.urlopen(request, timeout=60) as stream:
        revision = json.load(stream)["sha"]
    url = METADATA_URL.format(revision=revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with urllib.request.urlopen(url, timeout=60) as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    temporary.replace(target)
    write_json(lock_path, dict(url=url, revision=revision, sha256=sha256(target)))
    return target


def encode(row):
    return zlib.compress(
        json.dumps(row, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(), 1
    )


def decode(blob):
    return json.loads(zlib.decompress(blob)) if blob is not None else None


def database(paths, readonly=False):
    file = paths.artifacts / "prepared.sqlite"
    file.parent.mkdir(parents=True, exist_ok=True)
    conn = (
        sqlite3.connect(f"file:{file.as_posix()}?mode=ro", uri=True)
        if readonly
        else sqlite3.connect(file)
    )
    conn.execute("PRAGMA mmap_size=268435456")
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rollout(dataset TEXT, sample TEXT, rollout INTEGER, generation BLOB, probability BLOB, attention BLOB, PRIMARY KEY(dataset,sample,rollout))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ingest(path TEXT PRIMARY KEY, digest TEXT, rows INTEGER)"
        )
    return conn


REQUIRED = {
    "generation": {
        "dataset",
        "sample_id",
        "rollout_id",
        "model",
        "question_text",
        "generated_text",
        "generated_token_ids",
        "n_generated_text_tokens",
        "hit_max_new_tokens",
        "gold_answer",
        "extracted_answer",
    },
    "probability": {"dataset", "sample_id", "rollout_id", "logp", "kl_u_p"},
    "attention": {
        "dataset",
        "sample_id",
        "rollout_id",
        "n_image_tokens",
        "n_prompt_text_tokens",
        "img_attn_total",
        "prompt_text_attn_total",
        "generated_text_attn_total",
        "n_generated_text_tokens",
    },
}


def ingest_file(conn, file, dataset, component, expected_rows=32000):
    digest = sha256(file)
    previous = conn.execute("SELECT digest,rows FROM ingest WHERE path=?", (str(file),)).fetchone()
    cursor = previous[1] if previous else 0
    if previous and previous[0] != digest:
        raise ValueError("Supplied source changed during ingestion")
    parquet = pq.ParquetFile(file)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"{file}: expected {expected_rows} rows, found {parquet.metadata.num_rows}"
        )
    if not REQUIRED[component] <= set(parquet.schema_arrow.names):
        raise ValueError(f"Unexpected supplied {component} schema: {parquet.schema_arrow}")
    offset = 0
    for batch in parquet.iter_batches(batch_size=128, columns=sorted(REQUIRED[component])):
        end = offset + len(batch)
        if end <= cursor:
            offset = end
            continue
        with conn:
            for position, row in enumerate(batch.to_pylist(), offset):
                if position < cursor:
                    continue
                sample, rid = str(row["sample_id"]), int(row["rollout_id"])
                if not 0 <= rid < 32:
                    raise ValueError("Rollout ID outside 0..31")
                existing = conn.execute(
                    f"SELECT {component} FROM rollout WHERE dataset=? AND sample=? AND rollout=?",
                    (dataset, sample, rid),
                ).fetchone()
                if existing and existing[0] is not None:
                    raise ValueError(f"Duplicate join key in {file}: {sample}/{rid}")
                conn.execute(
                    "INSERT OR IGNORE INTO rollout(dataset,sample,rollout) VALUES(?,?,?)",
                    (dataset, sample, rid),
                )
                conn.execute(
                    f"UPDATE rollout SET {component}=? WHERE dataset=? AND sample=? AND rollout=?",
                    (encode(row), dataset, sample, rid),
                )
            conn.execute("INSERT OR REPLACE INTO ingest VALUES(?,?,?)", (str(file), digest, end))
        offset = end
    return dict(
        sha256=digest,
        bytes=file.stat().st_size,
        rows=offset,
        schema=str(parquet.schema_arrow.remove_metadata()),
        arrow_metadata={
            k.decode(): v.decode() for k, v in (parquet.schema_arrow.metadata or {}).items()
        },
    )


def source_rows(conn, dataset, key):
    data = conn.execute(
        "SELECT rollout,generation,probability,attention FROM rollout WHERE dataset=? AND sample=? ORDER BY rollout",
        (dataset, key),
    ).fetchall()
    if [r[0] for r in data] != list(range(32)):
        raise ValueError(f"Question must contain exactly 32 unique rollouts: {dataset}/{key}")
    result = []
    for rid, g, p, a in data:
        if g is None or p is None or (dataset == "mathvista" and a is None):
            raise ValueError(f"Missing joined component for {dataset}/{key}/{rid}")
        g, p, a = decode(g), decode(p), decode(a)
        for other in (p, a):
            if other is not None and any(
                g[k] != other[k] for k in ("dataset", "sample_id", "rollout_id")
            ):
                raise ValueError("Supplied join-key values disagree across components")
        result.append((g, p, a))
    return result


def validated_arrays(data, dataset):
    generations, logp, kl, visual = [], [], [], []
    for g, p, a in data:
        expected_model = "qwen2_5_3b_instruct" if dataset == "math" else "qwen2_5_vl_3b"
        if g["model"] != expected_model:
            raise ValueError("Supplied model identifier does not match its assigned dataset folder")
        n = g["n_generated_text_tokens"]
        if n != len(g["generated_token_ids"]) or n != len(p["logp"]) or n != len(p["kl_u_p"]):
            raise ValueError("Supplied continuation/token likelihood alignment failure")
        if n <= 0:
            raise ValueError(
                "Zero retained-token rollout: manual's token means are undefined; inspect source"
            )
        lp, k = np.asarray(p["logp"], dtype=np.float64), np.asarray(p["kl_u_p"], dtype=np.float64)
        if (
            not np.isfinite(lp).all()
            or not np.isfinite(k).all()
            or (lp > 1e-5).any()
            or (k < -1e-5).any()
        ):
            raise ValueError("Invalid supplied log probabilities/KL values")
        if a is not None:
            att = [
                np.asarray(a[name], dtype=np.float64)
                for name in (
                    "img_attn_total",
                    "prompt_text_attn_total",
                    "generated_text_attn_total",
                )
            ]
            if any(len(v) != n or not np.isfinite(v).all() or (v < 0).any() for v in att):
                raise ValueError("Invalid supplied attention arrays")
            if (
                a["n_generated_text_tokens"] != list(range(n))
                or a["n_image_tokens"] <= 0
                or a["n_prompt_text_tokens"] <= 0
            ):
                raise ValueError("Attention group-count or preceding-token alignment mismatch")
            denominator = sum(att)
            if (denominator <= 0).any():
                raise ValueError("Undefined visual engagement: no eligible attention mass")
            visual.append(att[0] / denominator)
        generations.append(g)
        logp.append(lp)
        kl.append(k)
    if (
        len({g["question_text"] for g in generations}) != 1
        or len({str(g["gold_answer"]) for g in generations}) != 1
    ):
        raise ValueError("Question/gold reference is not constant across supplied rollouts")
    return generations, logp, kl, visual


@lru_cache(maxsize=2)
def released_metadata(path):
    return read_json(Path(path))


def load_question(paths, dataset, key, metadata=None):
    with closing(database(paths, readonly=True)) as conn:
        data = source_rows(conn, dataset, key)
    if dataset == "mathvista":
        metadata = (
            metadata
            if metadata is not None
            else released_metadata(str(paths.artifacts / "metadata/testmini.json"))
        )
        meta = metadata[key]
    else:
        meta = {}
    scorer = Scorer(dataset, meta)
    generations, logp, kl, visual = validated_arrays(data, dataset)
    answers = [scorer.answer(g["extracted_answer"]) for g in generations]
    pool = Pool(
        list(range(32)),
        answers,
        np.asarray([g["n_generated_text_tokens"] for g in generations]),
        logp,
        kl,
        visual if dataset == "mathvista" else None,
    )
    correct = np.asarray(
        [scorer.correct(g["extracted_answer"], g["gold_answer"]) for g in generations],
        dtype=np.int8,
    )
    return dict(pool=pool, correct=correct, scorer=scorer, generations=generations, metadata=meta)


def prepare(paths, source=None, metadata=None, expected_questions=1000):
    marker = paths.evidence / "preparation.json"
    if marker.exists():
        previous = read_json(marker)
        if sha256(paths.artifacts / "prepared.sqlite") != previous["database_sha256"]:
            raise IOError("Prepared source database checksum mismatch")
        return
    source = Path(source) if source else fetch(paths)
    readme = source / "README.MD"
    text = readme.read_text()
    import re

    match = re.search(r"Model:\s*`" + re.escape(MODEL) + r"`, revision `([a-f0-9]{40})`", text)
    if not match:
        raise ValueError("Exact original MATH model revision missing from supplied README.MD")
    paths.evidence.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(readme, paths.evidence / "supplied_README.MD")
    metadata = Path(metadata) if metadata else fetch_metadata(paths)
    target = paths.artifacts / "metadata/testmini.json"
    if metadata.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata, target)
    released = read_json(target)
    conn = database(paths)
    files, statistics = {}, {}
    for dataset, folder in FOLDERS.items():
        components = [
            ("generation", "generations.parquet"),
            ("probability", "probability_metrics.parquet"),
        ]
        if dataset == "mathvista":
            components.append(("attention", "attention_metrics.parquet"))
        for component, name in components:
            file = source / folder / name
            files[f"{folder}/{name}"] = ingest_file(
                conn, file, dataset, component, expected_questions * 32
            )
        provenance = [v["arrow_metadata"] for k, v in files.items() if k.startswith(folder + "/")]
        for field in ("config_sha256", "model_revision", "model_id", "dataset_revision"):
            values = [p.get(field) for p in provenance]
            # The MathVista export omits config_sha256/dataset_revision;
            # retain that fact instead of inventing a source revision.
            if len(set(values)) != 1 or (
                (dataset == "math" or field in ("model_id", "model_revision")) and not all(values)
            ):
                raise ValueError(f"Supplied {dataset} {field} differs across joined files")
        if dataset == "math" and provenance[0]["model_revision"] != match[1]:
            raise ValueError("MATH Parquet model revision differs from supplied README")
    splits, orders = {}, {}
    for dataset in DATASETS:
        keys = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT sample FROM rollout WHERE dataset=? ORDER BY sample", (dataset,)
            )
        ]
        if len(keys) != expected_questions:
            raise ValueError(
                f"Expected {expected_questions} {dataset} questions, found {len(keys)}"
            )
        if dataset == "mathvista" and not set(keys) <= set(released):
            raise ValueError("MathVista IDs absent from released question metadata")
        shuffled = np.random.default_rng(SEED).permutation(keys).tolist()
        dev = 200 if expected_questions == 1000 else max(1, expected_questions // 5)
        splits[dataset] = dict(development=shuffled[:dev], evaluation=shuffled[dev:])
        order_rng = np.random.default_rng(SEED)
        orders[dataset] = {
            key: [order_rng.permutation(32).tolist() for _ in range(10)] for key in keys
        }
        statistics[dataset] = dict(retained_tokens=0, capped_rollouts=0, null_extractions=0)
        for key in keys:
            # Validate shapes, IDs and references before declaring preparation complete.
            generations, *_ = validated_arrays(source_rows(conn, dataset, key), dataset)
            for g in generations:
                statistics[dataset]["retained_tokens"] += g["n_generated_text_tokens"]
                statistics[dataset]["capped_rollouts"] += int(g["hit_max_new_tokens"])
                statistics[dataset]["null_extractions"] += int(g["extracted_answer"] is None)
            if dataset == "mathvista":
                meta = released[key]
                if not {"question_type", "answer_type", "answer"} <= set(meta):
                    raise ValueError("Released MathVista answer metadata is incomplete")
                if meta["question_type"] == "multi_choice" and not meta.get("choices"):
                    raise ValueError("Released multiple-choice options missing")
                if meta["answer_type"] == "float" and meta.get("precision") is None:
                    raise ValueError("Released float precision missing")
        if expected_questions == 1000:
            label = "MATH" if dataset == "math" else "MathVista"
            declared = re.search(
                r"\|\s*" + label + r"\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|", text
            )
            if not declared or [int(x.replace(",", "")) for x in declared.groups()] != list(
                statistics[dataset].values()
            ):
                raise ValueError(f"Supplied {dataset} totals disagree with its README")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.close()
    write_json(
        paths.evidence / "splits.json", dict(**splits, live_math=splits["math"]["evaluation"][:50])
    )
    write_json(paths.evidence / "orders.json", orders)
    write_json(paths.evidence / "protocol.json", protocol())
    write_json(
        paths.evidence / "scorer.json",
        dict(
            math_verify=importlib.metadata.version("math-verify"),
            sympy=importlib.metadata.version("sympy"),
            latex_parser=importlib.metadata.version("latex2sympy2_extended"),
            implementation_sha256=sha256(Path(__file__).with_name("scoring.py")),
            mathvista_metadata_sha256=sha256(target),
            mathvista_rules="option letters/closest-choice text, integer conversion, declared float precision; missing is NO_ANSWER",
        ),
    )
    write_json(
        paths.evidence / "asset_lock.json",
        dict(
            model=MODEL,
            model_revision=match[1],
            supplied_readme_sha256=sha256(readme),
            metadata_sha256=sha256(target),
        ),
    )
    write_json(paths.evidence / "environment.json", environment())
    write_json(
        marker,
        dict(
            schema=1,
            source=DRIVE_FOLDER,
            files=files,
            statistics=statistics,
            database_sha256=sha256(paths.artifacts / "prepared.sqlite"),
            questions_per_dataset=expected_questions,
            supplied_readme_sha256=sha256(readme),
            metadata_sha256=sha256(target),
        ),
    )
    print(
        "[prepare] Supplied sources, join keys, split IDs and ten nested orders saved", flush=True
    )
