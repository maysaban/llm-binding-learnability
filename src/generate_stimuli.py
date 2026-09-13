from stimuli import EXPERIMENTS


def main():
    for experiment in EXPERIMENTS:
        items = experiment.generate()
        out = experiment.name
        print(
            f"{out}: {len(items)} items / {len(items) * 4} sentences"
        )
        print(f"  -> data/{out}/generated/base_items.csv")
        print(f"  -> data/{out}/generated/stimuli.csv")
        print(f"  -> data/{out}/generated/sentences.csv")


if __name__ == "__main__":
    main()
