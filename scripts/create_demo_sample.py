import argparse
import random
import csv

def reservoir_sample_csv(source_path, out_path, n, encoding="utf-8"):
    """
    Reservoir sampling to pick n rows from a large CSV without loading whole file.
    Keeps header + random n rows.
    """
    with open(source_path, "r", encoding=encoding, errors="replace", newline="") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)

        # Reservoir for samples
        reservoir = []
        for t, row in enumerate(reader, 1):
            if len(reservoir) < n:
                reservoir.append(row)
            else:
                m = random.randint(0, t - 1)
                if m < n:
                    reservoir[m] = row

    with open(out_path, "w", encoding=encoding, newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(reservoir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create demo CSV sample using reservoir sampling")
    parser.add_argument("--source", type=str, required=True, help="Path to source large CSV")
    parser.add_argument("--out", type=str, required=True, help="Path to output small demo CSV")
    parser.add_argument("--rows", type=int, default=30000, help="Number of rows for demo sample")
    args = parser.parse_args()

    print(f"📂 Using source file: {args.source}")
    reservoir_sample_csv(args.source, args.out, args.rows)
    print(f"✅ Created demo sample: {args.out} with {args.rows} rows")
