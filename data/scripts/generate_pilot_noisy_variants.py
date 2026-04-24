#!/usr/bin/env python3
"""Create noisier synthetic dialogue variants for the pilot gold samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "raw_sources" / "manual_gold"
DEFAULT_OUTPUT = ROOT / "synthetic" / "pilot_noisy"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


NOISY_USER_TURNS = {
    "manual_001_amazing_lunch_indicator": {
        2: "Basically a phone app for finding food spots near me and checking what each place offers.",
        4: "Regular app users, restaurant people using the portal, and whoever manages the whole thing.",
        6: "People should be able to look up places to eat, bounce between map and list results, open the full place details, and get directions. The restaurant side should update venue info, and management should approve those restaurant accounts.",
        8: "Yeah, the mobile side needs sign-in, and the portal side needs accounts too for the restaurant side and the people running it.",
        10: "Searches need to feel quick, like two seconds tops, and if GPS or the internet dies the app should say so instead of just hanging.",
        12: "Login traffic needs protecting, and after three bad tries the portal account should cool off for about thirty minutes."
    },
    "manual_002_restaurant_menu_ordering_system": {
        2: "An in-restaurant ordering setup with table screens, staff tablets, and kitchen displays.",
        4: "Diners, floor staff, kitchen crew, and managers.",
        6: "Guests should send orders and buzz staff, servers should take cash or card payments, the kitchen needs to mark dishes ready, and managers should handle refunds.",
        8: "Staff tablets need sign-in with usernames and passwords, but customers at the table screens should not have to log in.",
        10: "The backend should cope with at least two hundred connected devices, and it cannot drop live orders or payments.",
        12: "The wireless side needs encryption, and a server should not be signed into multiple tablets at once."
    },
    "manual_003_keepass_password_safe": {
        2: "A password vault tool so people can keep account creds together safely.",
        4: "Mostly regular users, plus IT admins in some cases.",
        6: "They need to spin up and unlock encrypted vaults, find saved items, tweak entries, and move password data in or out.",
        8: "Yep, access should rely on a master password, and maybe a key file too if they set up that combined key option.",
        10: "If someone copies a password, don't leave it sitting around in memory for more than ten seconds.",
        12: "Everything should stay encrypted, the vault must not open without the needed master password or key file, and there should be no recovery backdoor."
    }
}


def load_samples(input_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("template_"):
            continue
        samples.append(json.loads(path.read_text()))
    return samples


def build_variant(sample: dict) -> dict:
    sample_id = sample["sample_id"]
    variant = json.loads(json.dumps(sample))
    variant["sample_id"] = f"{sample_id}_noisy"
    variant["metadata"]["source_type"] = "synthetic"
    variant["metadata"]["parent_id"] = sample_id
    variant["metadata"]["dialogue_style"] = "noisy"

    replacements = NOISY_USER_TURNS[sample_id]
    for turn in variant["dialogue"]:
        if turn["role"] == "user" and turn["turn_id"] in replacements:
            turn["text"] = replacements[turn["turn_id"]]

    return variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for sample in load_samples(args.input_dir):
        variant = build_variant(sample)
        output_path = args.output_dir / f"{variant['sample_id']}.json"
        output_path.write_text(json.dumps(variant, indent=2))
        summary.append(
            {
                "sample_id": variant["sample_id"],
                "parent_id": variant["metadata"]["parent_id"],
                "path": display_path(output_path),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
