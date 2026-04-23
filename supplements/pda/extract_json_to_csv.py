#!/usr/bin/env python3

import json
import csv
import os
from pathlib import Path

def extract_data_from_json(json_file_path):
    """
    Extract query_title, accession, sciname, and percent_id from a JSON file.
    Returns a list of dictionaries with the extracted data.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Navigate to the search results
        search_results = data['BlastOutput2']['report']['results']['search']
        query_title = search_results['query_title']
        query_len = search_results['query_len']
        
        extracted_data = []
        
        # Process each hit
        if 'hits' in search_results:
            for hit in search_results['hits']:
                # Get description info (first description entry)
                if hit['description']:
                    desc = hit['description'][0]
                    accession = desc.get('accession', '')
                    sciname = desc.get('sciname', '')
                
                # Process each HSP for this hit
                if 'hsps' in hit:
                    for hsp in hit['hsps']:
                        identity = hsp.get('identity', 0)
                        align_len = hsp.get('align_len', query_len)
                        
                        # Calculate percent identity
                        percent_id = (identity / align_len * 100) if align_len > 0 else 0
                        
                        extracted_data.append({
                            'query_title': query_title,
                            'accession': accession,
                            'sciname': sciname,
                            'percent_id': round(percent_id, 2)
                        })
        
        return extracted_data
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return []

def main():
    # Directory containing JSON files
    json_dir = Path("GHMD9VAE014-Alignment.json")
    
    # Output CSV file
    output_csv = "extracted_blast_results.csv"
    
    if not json_dir.exists():
        print(f"Directory {json_dir} not found!")
        return
    
    all_data = []
    
    # Process all JSON files in the directory
    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process...")
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        data = extract_data_from_json(json_file)
        all_data.extend(data)
    
    # Write to CSV
    if all_data:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['query_title', 'accession', 'sciname', 'percent_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"Successfully extracted {len(all_data)} records to {output_csv}")
    else:
        print("No data extracted!")

if __name__ == "__main__":
    main()