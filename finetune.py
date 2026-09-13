"""Fine-tune all-MiniLM-L6-v2 on the dair-ai/emotion TRAINING split (CPU, ~20 min).

    python finetune.py

build.py calls the same function when artifacts/minilm_ft/ is missing; this
script only exists so the slow step can run on its own.
"""

import json
import time

from src import classifiers as C
from src import data as DATA

if __name__ == "__main__":
    train = DATA.load_emotion()["train"]
    t0 = time.time()
    info = C.finetune(train.text, train.label.to_numpy(), log=lambda s: print(s, flush=True))
    (C.FT_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {C.FT_DIR}")
