import os
import glob
import csv
import time
import requests
import pandas as pd

BASE_URL = "https://rest.uniprot.org"
CHUNK_SIZE = 50000


def process_chunk(ids, is_first_chunk, output_file):
    print(f"\n--- Processing batch of {len(ids)} IDs ---")
    # Step 1: Submit Job
    payload = {'from': 'UniProtKB_AC-ID', 'to': 'UniProtKB', 'ids': ",".join(ids)}
    res = requests.post(f"{BASE_URL}/idmapping/run", data=payload)
    res.raise_for_status()
    job_id = res.json()['jobId']
    # Step 2: Poll
    while True:
        status_res = requests.get(f"{BASE_URL}/idmapping/status/{job_id}", allow_redirects=False)
        if status_res.status_code == 303:
            break
        data = status_res.json()
        if data.get('jobStatus') == 'FINISHED':
            break
        time.sleep(5)
    # Step 3: Stream and Logic with Header Mapping
    params = {'format': 'tsv', 'fields': 'accession,reviewed,annotation_score'}
    result_url = f"{BASE_URL}/idmapping/uniprotkb/results/stream/{job_id}"
    mode = 'w' if is_first_chunk else 'a'
    with requests.get(result_url, params=params, stream=True) as r:
        r.raise_for_status()
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            lines = r.iter_lines(decode_unicode=True)
            try:
                raw_header = next(lines)
                header_cols = raw_header.split('\t')
                idx_acc = header_cols.index("Entry")
                idx_rev = header_cols.index("Reviewed")
                idx_score = header_cols.index("Annotation")
                if is_first_chunk:
                    writer.writerow(['query_id', 'entry_type', 'annotation_score', 'status'])
            except (StopIteration, ValueError) as e:
                print(f"Skipping batch: No data or unexpected header format ({e})")
                return
            for line in lines:
                if not line:
                    continue
                cols = line.split('\t')
                acc = cols[idx_acc]
                rev_status = cols[idx_rev]
                score_val = cols[idx_score]
                try:
                    score = float(score_val) if score_val else 0.0
                except ValueError:
                    score = 0.0
                if score > 0.0:
                    entry_type = "UniProtKB/Swiss-Prot" if rev_status == "reviewed" else "UniProtKB/TrEMBL"
                    writer.writerow([acc, entry_type, score, "Active"])
                else:
                    writer.writerow([acc, "Inactive/Deleted", score, "Inactive"])


if __name__ == "__main__":
    PREDICTIONS_DIR = "../predictions_screen"

    csv_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "pred_s*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{PREDICTIONS_DIR}'. Check the path and try again.")
        exit(1)

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  • {f}")

    for input_file in csv_files:
        folder = os.path.dirname(input_file)
        basename = os.path.splitext(os.path.basename(input_file))[0]  # e.g. "info_asgard"
        out_basename = "status_" + basename[len("info_"):] if basename.startswith("info_") else "status_" + basename
        output_file = os.path.join(folder, out_basename + ".csv")

        print(f"\n{'═'*60}")
        print(f"  INPUT : {input_file}")
        print(f"  OUTPUT: {output_file}")

        seq_info = pd.read_csv(input_file)
        all_ids = seq_info['UniProtID'].dropna().unique().tolist()
        print(f"  {len(all_ids)} unique UniProt IDs found")

        try:
            for i in range(0, len(all_ids), CHUNK_SIZE):
                process_chunk(all_ids[i:i + CHUNK_SIZE], is_first_chunk=(i == 0), output_file=output_file)
            print(f"  ✓ Success! Results saved to {output_file}")
        except Exception as e:
            print(f"  ✗ FATAL ERROR on {input_file}: {e}")
