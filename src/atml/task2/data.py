"""Official source downloads, deterministic selection, and paired curriculum draws."""

import copy
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atml.common.io import read_json, sha256, write_json

from .config import QWEN, SEED, SIGLIP
from .metrics import consensus, normalize

VQA_BASE = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org/"
PLURALS = {
    "person": "people",
    "knife": "knives",
    "mouse": "mice",
    "sheep": "sheep",
    "scissors": "scissors",
    "skis": "skis",
    "bus": "buses",
    "bench": "benches",
    "sandwich": "sandwiches",
    "toothbrush": "toothbrushes",
    "wine glass": "wine glasses",
    "teddy bear": "teddy bears",
    "sports ball": "sports balls",
    "broccoli": "broccoli",
}


def download(url, path):
    """Atomic downloads; a failed transfer never becomes a completed input file."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as source, tmp.open("wb") as output:
        shutil.copyfileobj(source, output, 1024 * 1024)
        expected = source.headers.get("Content-Length")
    if expected and tmp.stat().st_size != int(expected):
        raise IOError(f"Incomplete download: {url}")
    tmp.replace(path)
    return path


def acquire(paths):
    raw = paths.artifacts / "raw"
    sources = {}
    archives = {"coco": COCO_BASE + "annotations/annotations_trainval2014.zip"}
    for split in ("Train", "Val"):
        for kind in ("Annotations", "Questions", "Complementary_Pairs"):
            archives[f"{kind}_{split}"] = VQA_BASE + f"v2_{kind}_{split}_mscoco.zip"
    for name, url in archives.items():
        archive = download(url, raw / f"{name}.zip")
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.endswith(".json"):
                    # Flatten only JSON members; never trust archive paths.
                    target = raw / Path(member).name
                    if not target.exists():
                        tmp = target.with_suffix(".part")
                        with z.open(member) as src, tmp.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        tmp.replace(target)
        sources[name] = dict(url=url, sha256=sha256(archive), bytes=archive.stat().st_size)
    write_json(paths.evidence / "sources.json", sources)


def content_ok(tokenizer, text, cap):
    return 0 < len(tokenizer.encode(text, add_special_tokens=False)) <= cap


def caption_candidates(raw, split, tokenizer):
    grouped = defaultdict(list)
    for a in read_json(raw / f"captions_{split}.json")["annotations"]:
        if content_ok(tokenizer, a["caption"], 64):
            grouped[a["image_id"]].append(
                dict(
                    id=f"caption:{a['id']}",
                    caption_id=a["id"],
                    image_id=a["image_id"],
                    split=split,
                    question="Describe this image.",
                    target=a["caption"],
                    component="caption",
                )
            )
    return grouped


def select_captions(grouped, count, per_image, rng, excluded=()):
    ids = sorted(i for i, rows in grouped.items() if len(rows) >= per_image and i not in excluded)
    rng.shuffle(ids)
    if len(ids) < count:
        raise ValueError("Not enough eligible caption images")
    rows = []
    for image in ids[:count]:
        choices = sorted(grouped[image], key=lambda r: r["caption_id"])
        rng.shuffle(choices)
        rows.extend(choices[:per_image])
    return rows


def natural_candidates(raw, split, tokenizer):
    questions = {
        r["question_id"]: r
        for r in read_json(raw / f"v2_OpenEnded_mscoco_{split}_questions.json")["questions"]
    }
    annotations = {
        r["question_id"]: r
        for r in read_json(raw / f"v2_mscoco_{split}_annotations.json")["annotations"]
    }
    pairs = read_json(raw / f"v2_mscoco_{split}_complementary_pairs.json")
    candidates, seen = [], set()
    for ids in pairs:
        ids = tuple(sorted(ids))
        if ids in seen:
            continue
        seen.add(ids)
        qs, anns = [questions[i] for i in ids], [annotations[i] for i in ids]
        if qs[0]["question"] != qs[1]["question"] or qs[0]["image_id"] == qs[1]["image_id"]:
            continue
        if anns[0]["answer_type"] != anns[1]["answer_type"]:
            continue
        answers = [[a["answer"] for a in ann["answers"]] for ann in anns]
        if any(len(a) != 10 for a in answers):
            continue
        votes = [consensus(a) for a in answers]
        if (
            min(v[2] for v in votes) < 8
            or votes[0][1] == votes[1][1]
            or not all(v[1] for v in votes)
        ):
            continue
        if not content_ok(tokenizer, qs[0]["question"], 80):
            continue
        if not all(content_ok(tokenizer, v[0], 16) for v in votes):
            continue
        pair_id = f"natural:{ids[0]}:{ids[1]}"
        candidates.append(
            [
                dict(
                    id=f"vqa:{qid}",
                    question_id=qid,
                    pair_id=pair_id,
                    image_id=q["image_id"],
                    partner_image_id=qs[1 - j]["image_id"],
                    split=split,
                    question=q["question"],
                    target=v[0],
                    canonical=v[1],
                    agreement=v[2],
                    references=ans,
                    component="natural",
                    answer_type=ann["answer_type"],
                )
                for j, (qid, q, ann, v, ans) in enumerate(zip(ids, qs, anns, votes, answers))
            ]
        )
    return candidates


def select_natural(candidates, quotas, rng, used, unique_images=True):
    candidates = sorted(candidates, key=lambda p: p[0]["pair_id"])
    rng.shuffle(candidates)
    counts, chosen, questions = Counter(), [], set()
    for pair in candidates:
        kind = pair[0]["answer_type"]
        images = {r["image_id"] for r in pair}
        qids = {r["question_id"] for r in pair if "question_id" in r}
        if counts[kind] >= quotas.get(kind, 0) or images & used or qids & questions:
            continue
        chosen.append(pair)
        counts[kind] += 1
        questions.update(qids)
        if unique_images:
            used.update(images)
        if dict(counts) == quotas:
            return chosen
    raise ValueError(f"Insufficient eligible natural pairs: {dict(counts)} vs {quotas}")


def instance_index(raw, split):
    obj = read_json(raw / f"instances_{split}.json")
    categories = {a["id"]: a["name"] for a in obj["categories"]}
    counts = defaultdict(Counter)
    crowd = defaultdict(set)
    for a in obj["annotations"]:
        if a.get("iscrowd", 0):
            crowd[a["image_id"]].add(a["category_id"])
        else:
            counts[a["image_id"]][a["category_id"]] += 1
    return sorted(r["id"] for r in obj["images"]), categories, counts, crowd


def grounding_pairs(index, split, kind, count, rng, allowed=None, used=None):
    ids, categories, counts, crowd = index
    ids = [i for i in ids if allowed is None or i in allowed]
    used = used if used is not None else set()
    pools = {}
    for cat in sorted(categories):
        bins = defaultdict(list)
        for i in ids:
            if i not in used and cat not in crowd[i]:
                n = counts[i][cat]
                if kind == "presence":
                    bins[int(n > 0)].append(i)
                elif 1 <= n <= 4:
                    bins[n].append(i)
        if len(bins) >= 2:
            pools[cat] = bins
    result, seen = [], set()
    # Categories are chosen uniformly, then eligible counts/images uniformly.
    for _ in range(count * 1000):
        cat = rng.choice(sorted(pools))
        bins = {n: [i for i in images if i not in used] for n, images in pools[cat].items()}
        bins = {n: images for n, images in bins.items() if images}
        if len(bins) < 2:
            continue
        numbers = rng.sample(sorted(bins), 2)
        images = [rng.choice(bins[n]) for n in numbers]
        key = (cat, *sorted(images))
        if key in seen:
            continue
        seen.add(key)
        name = categories[cat]
        plural = PLURALS.get(name, name + "s")
        question = (
            f"Does this image contain {name}?"
            if kind == "presence"
            else f"How many {plural} can you see?"
        )
        pair_id = f"{kind}:{split}:{cat}:{min(images)}:{max(images)}"
        result.append(
            [
                dict(
                    id=f"{pair_id}:{image}",
                    pair_id=pair_id,
                    image_id=image,
                    partner_image_id=images[1 - j],
                    question=question,
                    category_id=cat,
                    target=("yes" if n else "no") if kind == "presence" else str(n),
                    canonical=("yes" if n else "no") if kind == "presence" else str(n),
                    component=kind,
                    answer_type=kind,
                    split=split,
                )
                for j, (image, n) in enumerate(zip(images, numbers))
            ]
        )
        # Evaluation caller supplies a shared set; training allows image reuse.
        if allowed is None:
            used.update(images)
        if len(result) == count:
            return result
    raise ValueError(f"Could construct only {len(result)}/{count} {kind} pairs")


def fix_mismatches(pairs, rng):
    for pair in pairs:
        forbidden = {r["image_id"] for r in pair}
        candidates = [
            p
            for p in pairs
            if p[0]["pair_id"] != pair[0]["pair_id"]
            and not forbidden.intersection(r["image_id"] for r in p)
        ]
        for row in pair:
            other = rng.choice(rng.choice(candidates))
            row["mismatch_image_id"] = other["image_id"]
            row["mismatch_pair_id"] = other["pair_id"]


def assert_isolation(manifest):
    sets = {}
    for name, rows in manifest["sets"].items():
        for pair in rows:
            if not isinstance(pair, list):
                continue
            assert len(pair) == 2 and pair[0]["image_id"] != pair[1]["image_id"], name
            assert pair[0]["pair_id"] == pair[1]["pair_id"], name
            assert pair[0]["question"] == pair[1]["question"], name
            assert normalize(pair[0]["target"]) != normalize(pair[1]["target"]), name
            assert pair[0]["partner_image_id"] == pair[1]["image_id"], name
            assert pair[1]["partner_image_id"] == pair[0]["image_id"], name
        flat = flatten(rows)
        sets[name] = {r["image_id"] for r in flat}
        expected = "train2014" if name.endswith("train") else "val2014"
        assert all(r["split"] == expected for r in flat), name
        for r in flat:
            assert r.get("partner_image_id", r["image_id"]) in sets[name], name
    training = set.union(*(s for n, s in sets.items() if n.endswith("train")))
    evaluation = [(n, s) for n, s in sets.items() if not n.endswith("train")]
    for i, (name, images) in enumerate(evaluation):
        assert not images & training, name
        for other, values in evaluation[i + 1 :]:
            assert not images & values, (name, other)
    return {name: len(ids) for name, ids in sets.items()}


def flatten(rows):
    return [r for item in rows for r in (item if isinstance(item, list) else [item])]


def prepare(paths):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    if paths.manifest.exists():
        assert_isolation(read_json(paths.manifest))
        return
    acquire(paths)
    lock_path = paths.evidence / "asset_lock.json"
    if not lock_path.exists():
        write_json(lock_path, {name: HfApi().model_info(name).sha for name in (QWEN, SIGLIP)})
    tokenizer = AutoTokenizer.from_pretrained(QWEN, revision=read_json(lock_path)[QWEN])
    tokenizer.save_pretrained(paths.artifacts / "tokenizer")
    rng, raw = random.Random(SEED), paths.artifacts / "raw"
    sets, used = {}, set()
    sets["caption_train"] = select_captions(
        caption_candidates(raw, "train2014", tokenizer), 6000, 3, rng
    )
    sets["caption_dev"] = select_captions(caption_candidates(raw, "val2014", tokenizer), 96, 1, rng)
    used.update(r["image_id"] for r in sets["caption_dev"])
    val = natural_candidates(raw, "val2014", tokenizer)
    for name, quotas in (
        ("natural_dev", {"yes/no": 12, "number": 8, "other": 12}),
        ("natural_test", {"yes/no": 36, "number": 24, "other": 36}),
    ):
        sets[name] = select_natural(val, quotas, rng, used)
    del val
    sets["natural_train"] = select_natural(
        natural_candidates(raw, "train2014", tokenizer),
        {"yes/no": 640, "number": 256, "other": 640},
        rng,
        set(),
        unique_images=False,
    )
    index = instance_index(raw, "val2014")
    for name, size in (("grounding_dev", 12), ("grounding_test", 32)):
        sets[name] = grounding_pairs(
            index, "val2014", "presence", size, rng, used=used
        ) + grounding_pairs(index, "val2014", "count", size, rng, used=used)
    del index
    allowed = {r["image_id"] for n in ("caption_train", "natural_train") for r in flatten(sets[n])}
    index = instance_index(raw, "train2014")
    for kind in ("presence", "count"):
        sets[kind + "_train"] = grounding_pairs(
            index, "train2014", kind, 4096, rng, allowed=allowed
        )
    for name in ("grounding_test", "natural_test"):
        fix_mismatches(sets[name], rng)
    manifest = dict(
        seed=SEED,
        sets=sets,
        qualitative_rule="first two pairs of each answer type in selected order",
    )
    manifest["image_counts"] = assert_isolation(manifest)
    write_json(paths.manifest, manifest)
    write_json(paths.evidence / "selection_checks.json", manifest["image_counts"])


def image_path(paths, split, image):
    return paths.artifacts / "images" / split / f"COCO_{split}_{image:012d}.jpg"


def fetch_images(paths, manifest):
    ids = sorted(
        {(r["split"], r["image_id"]) for rows in manifest["sets"].values() for r in flatten(rows)}
    )

    def fetch(item):
        split, image = item
        path = image_path(paths, split, image)
        return download(COCO_BASE + f"{split}/{path.name}", path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(fetch, ids)):
            if index % 256 == 0:
                print(f"[images] {index + 1}/{len(ids)}", flush=True)


class Sampler:
    """No quota balancing: eight independent two-example draws per optimizer update."""

    def __init__(self, manifest, state=None):
        self.sets = manifest["sets"]
        self.rng = random.Random(SEED)
        self.position = 0
        self.caption_order = []
        if state:
            self.rng.setstate(state["rng"])
            self.position = state["position"]
            self.caption_order = state["caption_order"]

    def state(self):
        return copy.deepcopy(
            dict(rng=self.rng.getstate(), position=self.position, caption_order=self.caption_order)
        )

    def batch(self, stage):
        rows, mixture = [], Counter()
        if stage.number == 1:
            captions = self.sets["caption_train"]
            for _ in range(16):
                if self.position % len(captions) == 0:
                    self.caption_order = list(range(len(captions)))
                    self.rng.shuffle(self.caption_order)
                rows.append(captions[self.caption_order[self.position % len(captions)]])
                self.position += 1
            return rows, {"caption": 16}
        for _ in range(8):
            kind = self.rng.choices(stage.components, weights=stage.probabilities)[0]
            if kind == "caption":
                pair = self.rng.sample(self.sets["caption_train"], 2)
            else:
                pair = self.rng.choice(self.sets[kind + "_train"])
            rows.extend(pair)
            mixture[kind] += 2
        self.position += 16
        return rows, dict(mixture)
