"""Evidence tables, plots and image worksheets; no report interpretation is generated."""

import base64
import csv
import html
import io
import json
import os
import zipfile
from collections import Counter, defaultdict

os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from atml.common.io import read_json, read_jsonl, sha256, write_json

from .config import CONDITIONS, DEV_UPDATES
from .data import COCO_BASE, assert_isolation, download, flatten, image_path
from .metrics import STAT_NAMES, bootstrap, pair_statistics
from .training import latest, load_checkpoint


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(paths):
    checks = {}
    manifest = read_json(paths.manifest)
    assert_isolation(manifest)
    checks["image_isolation"] = True
    expected_sizes = {
        "caption_train": 18000,
        "caption_dev": 96,
        "natural_train": 3072,
        "natural_dev": 64,
        "natural_test": 192,
        "grounding_dev": 48,
        "grounding_test": 128,
    }
    for name, size in expected_sizes.items():
        checks[f"selection/{name}"] = len(flatten(manifest["sets"][name])) == size
    checks["caption_train_6000_images_3_each"] = Counter(
        Counter(r["image_id"] for r in manifest["sets"]["caption_train"]).values()
    ) == {3: 6000}
    for name in ("natural_train", "natural_dev", "natural_test"):
        examples = flatten(manifest["sets"][name])
        checks[f"unique_questions/{name}"] = len({r["id"] for r in examples}) == len(examples)
    for name, quota in (
        ("natural_train", {"yes/no": 640, "number": 256, "other": 640}),
        ("natural_dev", {"yes/no": 12, "number": 8, "other": 12}),
        ("natural_test", {"yes/no": 36, "number": 24, "other": 36}),
    ):
        checks[f"quotas/{name}"] = (
            dict(Counter(p[0]["answer_type"] for p in manifest["sets"][name])) == quota
        )
    for name in (
        "protocol.json",
        "asset_lock.json",
        "sources.json",
        "calibration.json",
        "feature_cache.json",
        "environment.json",
        "resources.json",
        "interface_checks.json",
        "tables/final.csv",
        "tables/correct_image_advantages.csv",
        "tables/bootstrap.json",
        "tables/development.csv",
        "tables/resources.csv",
        "plots/learning_curves.png",
        "qualitative.html",
    ):
        checks[f"evidence/{name}"] = (paths.evidence / name).exists()
    for update in (0, *range(250, 4501, 250)):
        file = paths.checkpoint(update) / "complete.json"
        checks[f"checkpoint/{update}"] = file.exists()
        if file.exists():
            receipt = read_json(file)
            checks[f"checkpoint_hashes/{update}"] = all(
                (file.parent / name).exists() and sha256(file.parent / name) == digest
                for name, digest in receipt["files"].items()
            )
    for update in DEV_UPDATES:
        marker = paths.evidence / f"development/update_{update:04d}/complete.json"
        checks[f"development/{update}"] = marker.exists()
        if marker.exists():
            expected = {
                "caption_dev/correct": 96,
                "grounding_dev/correct": 48,
                "grounding_dev/complementary": 48,
                "grounding_dev/no_image": 48,
                "natural_dev/correct": 64,
                "natural_dev/complementary": 64,
                "natural_dev/no_image": 64,
            }
            checks[f"development_counts/{update}"] = read_json(marker)["counts"] == expected
            checks[f"development_hashes/{update}"] = all(
                sha256(marker.parent / n) == v for n, v in read_json(marker)["journals"].items()
            )
    for suite in ("grounding", "natural"):
        wanted = [r["id"] for r in flatten(manifest["sets"][suite + "_test"])]
        for condition in CONDITIONS:
            file = paths.evidence / "final" / f"{suite}_test_{condition}.jsonl"
            rows = read_jsonl(file) if file.exists() else []
            checks[f"final/{suite}/{condition}"] = sorted(r["id"] for r in rows) == sorted(wanted)
    final_marker = paths.evidence / "final/complete.json"
    checks["final_complete_marker"] = final_marker.exists()
    if final_marker.exists():
        checks["final_hashes"] = all(
            sha256(final_marker.parent / n) == v
            for n, v in read_json(final_marker)["journals"].items()
        )
    if latest(paths) == 4500:
        state = load_checkpoint(paths, 4500)
        checks["4500_applied_updates"] = [r["update"] for r in state["history"]] == list(
            range(1, 4501)
        )
        checks["72000_presentations"] = sum(state["mixture"].values()) == 72000
    review = paths.evidence / "grounding_review.csv"
    if review.exists():
        with review.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        reviewed = {r["id"] for r in rows if r["label_checked"].strip() and r["notes"].strip()}
        expected = {
            r["id"]
            for n in ("grounding_dev", "grounding_test")
            for r in flatten(manifest["sets"][n])
        }
        checks["human_grounding_label_inspection"] = expected <= reviewed
    else:
        checks["human_grounding_label_inspection"] = False
    result = dict(
        checks=checks,
        computational_complete=all(v for k, v in checks.items() if not k.startswith("human_")),
        pending=[k for k, v in checks.items() if not v],
    )
    write_json(paths.evidence / "completeness.json", result)
    return result


def summarize(paths):
    folder = paths.evidence
    table_rows, intervals, advantages = [], {}, []
    for suite in ("grounding", "natural"):
        controls = {}
        for condition in CONDITIONS:
            file = folder / "final" / f"{suite}_test_{condition}.jsonl"
            if not file.exists():
                continue
            rows = read_jsonl(file)
            rows.sort(key=lambda r: (r["pair_id"], r["id"]))
            controls[condition] = rows
            types = ["all"] + sorted({r["answer_type"] for r in rows})
            for kind in types:
                selected = [r for r in rows if kind == "all" or r["answer_type"] == kind]
                values = pair_statistics(selected)
                stats = bootstrap(values)
                key = f"{suite}/{condition}/{kind}"
                intervals[key] = stats
                result = dict(suite=suite, condition=condition, answer_type=kind, pairs=len(values))
                for i, metric in enumerate(STAT_NAMES):
                    result[metric] = stats["mean"][i]
                    result[metric + "_low"] = stats["low"][i]
                    result[metric + "_high"] = stats["high"][i]
                table_rows.append(result)
        if "correct" in controls and "complementary" in controls:
            correct, swapped = controls["correct"], controls["complementary"]
            if [r["id"] for r in correct] != [r["id"] for r in swapped]:
                raise ValueError("Cannot compute paired differences on different examples")
            for kind in ["all"] + sorted({r["answer_type"] for r in correct}):
                c = pair_statistics(
                    [r for r in correct if kind == "all" or r["answer_type"] == kind]
                )
                s = pair_statistics(
                    [r for r in swapped if kind == "all" or r["answer_type"] == kind]
                )
                for metric, values in (("accuracy", c[:, 0] - s[:, 0]), ("nll", s[:, 4] - c[:, 4])):
                    stats = bootstrap(values[:, None])
                    advantages.append(
                        dict(
                            suite=suite,
                            answer_type=kind,
                            metric=metric,
                            mean=stats["mean"][0],
                            low=stats["low"][0],
                            high=stats["high"][0],
                        )
                    )
    if table_rows:
        write_csv(folder / "tables/final.csv", table_rows)
        write_json(
            folder / "tables/bootstrap.json",
            dict(resamples=4000, seed=6304, unit="complete pair", intervals=intervals),
        )
    if advantages:
        write_csv(folder / "tables/correct_image_advantages.csv", advantages)
    development = []
    for update in DEV_UPDATES:
        root = folder / f"development/update_{update:04d}"
        for suite in ("caption", "grounding", "natural"):
            values = {}
            for condition in ("correct", "complementary", "no_image"):
                file = root / f"{suite}_dev_{condition}.jsonl"
                if file.exists():
                    values[condition] = np.mean([r["nll"] for r in read_jsonl(file)])
            if values:
                development.append(
                    dict(
                        update=update,
                        suite=suite,
                        correct=values["correct"],
                        complementary=values.get("complementary", ""),
                        no_image=values.get("no_image", ""),
                        swap_penalty=values["complementary"] - values["correct"]
                        if "complementary" in values
                        else "",
                    )
                )
    if development:
        write_csv(folder / "tables/development.csv", development)
    if (folder / "resources.json").exists():
        resources = [
            dict(stage=key, **value) for key, value in read_json(folder / "resources.json").items()
        ]
        write_csv(folder / "tables/resources.csv", resources)
    if (folder / "training.jsonl").exists():
        training = read_jsonl(folder / "training.jsonl")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        if training:
            axes[0, 0].plot(
                [r["update"] for r in training], [r["loss"] for r in training], alpha=0.5
            )
        axes[0, 0].set_ylabel("Training example-mean CE (includes EOT)")
        for suite, ax in (
            ("caption", axes[0, 1]),
            ("grounding", axes[1, 0]),
            ("natural", axes[1, 1]),
        ):
            rows = [r for r in development if r["suite"] == suite]
            for key in ("correct", "complementary", "no_image", "swap_penalty"):
                selected = [r for r in rows if r[key] != ""]
                if selected:
                    ax.plot(
                        [r["update"] for r in selected],
                        [r[key] for r in selected],
                        marker=".",
                        label=key,
                    )
            ax.set_ylabel(suite + " reference loss / NLL")
            if rows:
                ax.legend(fontsize=8)
        for ax in axes.flat:
            for boundary in (2250, 3750):
                ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Applied main-run update")
        fig.tight_layout()
        (folder / "plots").mkdir(exist_ok=True)
        fig.savefig(folder / "plots/learning_curves.png", dpi=160)
        plt.close(fig)
    worksheets(paths, include_predictions=True)
    return audit(paths)


def image_html(path):
    if not path.exists():
        return html.escape(path.name) + " (run images command to download)"
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((320, 240))
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=85)
    return (
        '<img alt="COCO image" src="data:image/jpeg;base64,'
        + base64.b64encode(buffer.getvalue()).decode()
        + '">'
    )


def ensure_worksheet_images(paths):
    """Rehydrate only worksheet images after fresh-disk backup recovery."""
    from concurrent.futures import ThreadPoolExecutor

    manifest = read_json(paths.manifest)
    selected = []
    for name in ("grounding_dev", "grounding_test"):
        selected.extend(flatten(manifest["sets"][name]))
    counts = Counter()
    for pair in manifest["sets"]["natural_test"]:
        kind = pair[0]["answer_type"]
        if counts[kind] < 2:
            selected.extend(pair)
            counts[kind] += 1
    images = sorted({(r["split"], r["image_id"]) for r in selected})

    def fetch(item):
        split, image = item
        path = image_path(paths, split, image)
        download(COCO_BASE + f"{split}/{path.name}", path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, images))


def worksheets(paths, include_predictions=False):
    """Review labels independently of predictions; keep human entries on reruns."""
    manifest = read_json(paths.manifest)
    review_path = paths.evidence / "grounding_review.csv"
    if not review_path.exists():
        write_csv(
            review_path,
            [
                dict(
                    id=r["id"],
                    pair_id=r["pair_id"],
                    split=r["split"],
                    image_id=r["image_id"],
                    question=r["question"],
                    annotation_target=r["target"],
                    label_checked="",
                    notes="",
                )
                for n in ("grounding_dev", "grounding_test")
                for r in flatten(manifest["sets"][n])
            ],
        )
    label_rows = []
    for name in ("grounding_dev", "grounding_test"):
        for pair in manifest["sets"][name]:
            cells = [html.escape(pair[0]["question"])]
            for row in pair:
                cells.append(
                    image_html(image_path(paths, row["split"], row["image_id"]))
                    + "<pre>"
                    + html.escape(
                        json.dumps(
                            {k: row[k] for k in ("id", "image_id", "category_id", "target")},
                            indent=2,
                        )
                    )
                    + "</pre>"
                )
            label_rows.append("<tr>" + "".join("<td>" + c + "</td>" for c in cells) + "</tr>")
    style = '<meta charset="utf-8"><style>td{padding:12px;vertical-align:top}table{border-collapse:collapse}tr{border-bottom:1px solid #aaa}pre{white-space:pre-wrap}</style>'
    (paths.evidence / "grounding_labels.html").write_text(
        style + "<h1>Annotation label inspection</h1>"
        "<p>Inspect images and annotation labels independently of model predictions. Record checks and any ambiguity in grounding_review.csv.</p><table>"
        + "".join(label_rows)
        + "</table>",
        encoding="utf-8",
    )
    if not include_predictions:
        return
    records = defaultdict(dict)
    for file in (paths.evidence / "final").glob("*.jsonl"):
        for row in read_jsonl(file):
            records[row["id"]][row["condition"]] = row.get("prediction", "")
    selected, counts = [], Counter()
    for suite in ("grounding_test", "natural_test"):
        for pair in manifest["sets"][suite]:
            kind = (suite, pair[0]["answer_type"])
            if counts[kind] >= 2:
                continue
            counts[kind] += 1
            selected.append(
                "<tr><td>"
                + html.escape(pair[0]["question"])
                + "</td>"
                + "".join(
                    "<td>"
                    + image_html(image_path(paths, row["split"], row["image_id"]))
                    + "<pre>"
                    + html.escape(
                        json.dumps(
                            dict(
                                id=row["id"],
                                references=row.get("references", [row["target"]]),
                                predictions=records[row["id"]],
                            ),
                            indent=2,
                        )
                    )
                    + "</pre></td>"
                    for row in pair
                )
                + "</tr>"
            )
    (paths.evidence / "qualitative.html").write_text(
        style + "<h1>Aligned complementary examples</h1>"
        "<p>Selection: first two pairs of each answer type in the saved selection order, fixed before model evaluation.</p><table>"
        + "".join(selected)
        + "</table>",
        encoding="utf-8",
    )


def export(paths):
    target = paths.root / "task2-evidence.zip"
    tmp = target.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(paths.evidence.rglob("*")):
            if file.is_file():
                z.write(file, file.relative_to(paths.evidence.parent))
    tmp.replace(target)
    return target
