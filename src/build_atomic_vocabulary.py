"""Build atomic sensor and actuator tokens for the selected tasks."""
import argparse
import json
from pathlib import Path

from definitions import ENV_CONFIG_PATH, PADDING_KEY, VALUE_KEY
from envs.environment_factory import EnvironmentFactory
from vocabulary import composite_key, set_vocabulary_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/atomic_vocabulary.json"))
    args = parser.parse_args()
    set_vocabulary_mode("compositional")
    keys = set()
    for task in dict.fromkeys(args.tasks):
        with open(Path(ENV_CONFIG_PATH) / f"arnold_{task}_config.json") as stream:
            config = json.load(stream)
        env = EnvironmentFactory.create(**config)
        for specs in env.get_obs_ids_dict().values():
            keys.update(composite_key(spec) for spec in specs)
        keys.update(composite_key(spec) for spec in env.get_action_specs())
        env.close()
    vocabulary = {PADDING_KEY: 0, VALUE_KEY: 1}
    vocabulary.update({key: index for index, key in enumerate(sorted(keys), start=2)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vocabulary, indent=2) + "\n")


if __name__ == "__main__":
    main()
