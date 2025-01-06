import kagglehub
import os
import csv

# Download latest version
# path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

# print("Path to dataset files:", path)


dataset_dir = "Dataset"
output_csv = "dataset_metadata.csv"

def create_csv_from_folders(dataset_dir, output_csv):

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath", "label"])
        
        for root, _, files in os.walk(dataset_dir):
            label = os.path.basename(root)
            
            for file in files:
                filepath = os.path.join(root, file)
                writer.writerow([filepath, label])

create_csv_from_folders(dataset_dir, output_csv)