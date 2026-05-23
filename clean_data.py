import pandas as pd
import re

print("--- Starting Phase I: Dataset Harmonization & Filtering ---")


def extract_subject_id(sample_name):
    """Convert sample columns like '3_Brain_HC-LAR-5_L1_1' to 'HC-LAR-5'."""
    sample_name = str(sample_name)
    tissue_match = re.match(r"^\d+_(?:Liver|Brain)_(.+?)_L1_1$", sample_name)
    if tissue_match:
        return tissue_match.group(1)

    mouse_match = re.search(r"(Mouse[_ ]\d+|M\d+)", sample_name)
    if mouse_match:
        return mouse_match.group(1).replace(" ", "_")

    return sample_name


# 1. LOAD THE RAW DOWNLOADS
# (Make sure these filenames match exactly what you downloaded)
mirna_raw = pd.read_csv("GSE294046_miRNA_complete_quantification_raw.tsv.gz", sep="\t", index_col=0)
brain_raw = pd.read_csv("GSE295428_series_matrix.txt.gz", sep="\t", comment="!", index_col=0)

liver_columns = [col for col in mirna_raw.columns if re.match(r"^\d+_Liver_", col)]
brain_columns = [col for col in mirna_raw.columns if re.match(r"^\d+_Brain_", col)]

if not liver_columns:
    raise ValueError("No Liver sample columns were found in the miRNA matrix.")
if not brain_columns:
    raise ValueError("No Brain sample columns were found in the miRNA matrix.")

liver_raw = mirna_raw[liver_columns]

# ==========================================
# STEP 1 & 2: FEATURE SCREENING & FILTERING
# ==========================================

# Apply Michelle's threshold filtering rule to Liver miRNAs
# (Keep rows where at least 10% of columns have counts >= 5)
min_samples_threshold = max(1, int(0.10 * liver_raw.shape[1]))
liver_filtered = liver_raw[(liver_raw >= 5).sum(axis=1) >= min_samples_threshold]
print(f"Liver miRNAs filtered down from {liver_raw.shape[0]} to {liver_filtered.shape[0]} features.")

# Slice Brain Dataset down to Michelle's target genes for Pipeline V2 & V1
# (Add the exact target gene names as they appear in the dataset row headers)
target_brain_genes = ["Sod1", "Sod2", "Cat", "Gpx1", "Gpx4", "Nfkb1"] 
# Ensure we only pick genes that actually exist in the matrix to avoid KeyError
available_genes = [gene for gene in target_brain_genes if gene in brain_raw.index]
brain_filtered = brain_raw.loc[available_genes]

if brain_filtered.empty:
    print("Brain GEO series matrix has no expression rows for target genes.")
    print("Using matching Brain miRNA samples from the quantification matrix instead.")
    brain_filtered = mirna_raw.loc[liver_filtered.index, brain_columns]
    brain_filtered = brain_filtered[(brain_filtered >= 5).sum(axis=1) >= min_samples_threshold]
    print(f"Brain miRNAs filtered down to {brain_filtered.shape[0]} features.")
else:
    print(f"Brain transcriptome sliced down to {len(available_genes)} critical target pathways.")

# ==========================================
# STEP 3: ID ALIGNMENT & TRANSPOSITION
# ==========================================

# Transpose matrices so Rows = Mouse Samples, Columns = Biological Features
X_features = liver_filtered.T
y_targets = brain_filtered.T

# Clean the Index IDs so they match perfectly across both tissues
# (This strips out extra text so 'GSMxxx_Liver_Mouse1' and 'GSMyyy_Brain_Mouse1' both become 'Mouse1')
X_features.index = X_features.index.map(extract_subject_id)
y_targets.index = y_targets.index.map(extract_subject_id)

# Intersect to keep ONLY the matching subjects
matching_subjects = X_features.index.intersection(y_targets.index)
X_final = X_features.loc[matching_subjects].sort_index()
y_final = y_targets.loc[matching_subjects].sort_index()

if X_final.empty or y_final.empty:
    raise ValueError("No matching Liver/Brain subjects were found after ID alignment.")

print(f"Successfully synchronized {len(matching_subjects)} matching rodent subjects.")

# Save clean matrices out to clean CSVs for your AI script
X_final.to_csv("liver_features_clean.csv")
y_final.to_csv("brain_targets_clean.csv")
print("Saved: 'liver_features_clean.csv' and 'brain_targets_clean.csv'. Ready for AI engine!")
