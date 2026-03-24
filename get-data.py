import argparse
from pathlib import Path
import pandas as pd


def combine_csvs(parent_folder: str, output_file: str = "testing-data.csv") -> None:
    parent_path = Path(parent_folder)

    if not parent_path.exists():
        raise FileNotFoundError(f"Folder not found: {parent_folder}")

    if not parent_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {parent_folder}")

    combined_frames = []

    for subfolder in parent_path.iterdir():
        if not subfolder.is_dir():
            continue

        folder_name = subfolder.name
        csv_files = list(subfolder.rglob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df.insert(0, "type", folder_name)
                combined_frames.append(df)
            except Exception as e:
                print(f"Skipping {csv_file} due to error: {e}")

    if not combined_frames:
        print("No CSV files found in subfolders.")
        return

    combined_df = pd.concat(combined_frames, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Created combined file: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine CSV files from subfolders into one training-data.csv file."
    )
    parser.add_argument(
        "parent_folder",
        help="Path to the folder containing subfolders with CSV files"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="online_spices_validation.csv",
        help="Output CSV filename (default: training-data.csv)"
    )

    args = parser.parse_args()
    combine_csvs(args.parent_folder, args.output)


if __name__ == "__main__":
    main()