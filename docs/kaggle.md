# Kaggle: copy, paste, run

Use these cells in your existing Kaggle session. You do not need a separate notebook file from this repo.

Before starting:

- Push the updated code to GitHub, then enable **GPU** and **Internet** in Kaggle.
- Create a Hugging Face token with write access. Add it to Kaggle Secrets as **`HF_TOKEN`** and enable it for the session.
- No Kaggle input dataset is needed for Task 1: the launcher downloads the model, training data, benchmarks, and the question YAML linked by the manual.

GitHub holds your code and final results. A private Hugging Face backup holds the large checkpoints and a recovery copy of your results. The backup repository is created automatically.

## 1. Clone and install

Paste this into a Kaggle code cell. It also works when the repository is already cloned.

```python
from pathlib import Path
import subprocess

repo = Path("/kaggle/working/PA3")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/hyderchaudhry/atml-pa3.git", str(repo)], check=True)
else:
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)

%cd /kaggle/working/PA3
!bash scripts/kaggle.sh setup
```

Wait for **Setup complete**. The launcher finds a supported Python automatically; if Kaggle only provides a newer version, it installs Python 3.11 in scratch space. You do not need to change the notebook kernel.

## 2. Set your backup name

Replace `YOUR_HF_USERNAME` below. Paste this into the next cell.

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["PA3_BACKUP_REPO"] = "YOUR_HF_USERNAME/atml-pa3-task1"
os.environ.pop("PA3_QUESTIONS", None)  # Clear an old placeholder; download the official YAML.
```

Use the same backup repository for this experiment in every session. Run only one writing session at a time.

## 3. Run

```python
!bash scripts/kaggle.sh run
```

That is the main command. It automatically:

- Restores saved progress when you start a fresh session.
- Checks the environment, imports the questions, and prepares the fixed data and probes.
- Runs the baseline, five training methods, required evaluations, and diagnostics.
- Saves and backs up every required recovery checkpoint before freeing its local copy.
- Produces tables, plots, qualitative responses, and a downloadable results ZIP.

Training prints its method, update counts, losses, and ETA. The launcher prints **`[next]`** before each step and **`[backup]`** when saving it. Let the backup finish before ending a session.

The manual's examples, batch sizes, precision, and evaluation settings remain fixed. The launcher also checks maximum-length GPU memory before training each method; it stops on failure instead of changing the recipe.

## 4. Continue after a session ends

Repeat **cells 1, 2, and 3**. Use the same backup repository. Saved progress restores automatically; you do not need to choose a checkpoint or remember which method was running.

If a session ends during a training segment, work after the last saved checkpoint may repeat. An unfinished benchmark split also reruns. Completed saved steps are reused.

## 5. Bring the results into VS Code

When the launcher prints **`[done]`**, download:

```text
/kaggle/working/pa3/task1-evidence.zip
```

Extract its `task1/` folder into your local repository's `evidence/` folder. Your results will be at `evidence/task1/`.

Fill **`evidence/task1/qualitative/annotations.csv`** yourself: the manual requires your annotations for 42 responses. Write your own report. Commit and push the small evidence files to GitHub; keep the large checkpoints in the private backup.

## Only if you need them

| What you want | Paste into a Kaggle cell |
| --- | --- |
| See saved progress | `!bash scripts/kaggle.sh status` |
| Run just one next step | `!bash scripts/kaggle.sh next` |
| Retry a failed backup | `!bash scripts/kaggle.sh backup` |
| Refresh the results ZIP | `!bash scripts/kaggle.sh export` |

If a command fails, read the error immediately above it. Fix that issue, then run the same command again. Do not delete checkpoints or re-prepare data to get past an error.

If an older clone stops with **Use Python 3.10-3.12**, run this recovery cell. The clone cell above now updates an existing checkout too.

```python
!git -C /kaggle/working/PA3 pull --ff-only
!bash /kaggle/working/PA3/scripts/kaggle.sh setup
```

The current launcher installs a supported Python automatically. If you still see the old `assert` traceback, the updated launcher has not reached the Kaggle checkout; check that the latest local code has been pushed to GitHub.

The question file comes from the [exact source linked by the manual](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml). Its original text, JSON system instructions, and SHA-256 are saved with the evidence. No external experiment code or model-judge implementation is copied or run. If you already have a supplied YAML, optionally set `PA3_QUESTIONS` to its real path; the original schema needs no prompt map.

Detailed implementation and evidence descriptions are in the [README](../README.md). The platform's storage/session rules are in [Kaggle's documentation](https://www.kaggle.com/docs/notebooks).
