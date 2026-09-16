import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharegpt", type=Path, required=True)
    parser.add_argument("--wildchat", type=Path, required=True)
    parser.add_argument("--instructcoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    sharegpt = json.loads(args.sharegpt.read_text(encoding="utf-8"))
    rng.shuffle(sharegpt)
    (args.output_dir / f"sharegpt-{args.count}-seed{args.seed}.json").write_text(
        json.dumps(sharegpt[: args.count], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    wildchat = json.loads(args.wildchat.read_text(encoding="utf-8"))
    (args.output_dir / f"wildchat-{args.count}-seed{args.seed}.json").write_text(
        json.dumps(wildchat[: args.count], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    coder = json.loads(args.instructcoder.read_text(encoding="utf-8"))
    rng.shuffle(coder)
    converted = []
    for index, row in enumerate(coder[: args.count]):
        prompt = row["instruction"]
        if row.get("input"):
            prompt += "\n\n" + row["input"]
        converted.append(
            {
                "id": f"instructcoder-{index:03d}",
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": row["output"]},
                ],
            }
        )
    (args.output_dir / f"instructcoder-{args.count}-seed{args.seed}.json").write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
