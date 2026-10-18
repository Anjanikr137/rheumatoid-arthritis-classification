RA PROJECT — DATASET DOWNLOAD KIT
===================================

PRIMARY DATASET
Title: Rheumatic and autoimmune diseases dataset
Size reported in paper: 12,085 records; 14 features; 7 diagnostic classes
Collection: one hospital and three laboratories in Baghdad, Iraq, 2019–2024
Repository DOI: 10.7910/DVN/VM4OR3
License: Harvard Dataverse catalog lists CC0 1.0 Universal.
Paper DOI: 10.1016/j.dib.2025.111623
Related ML paper DOI: 10.7717/peerj-cs.3096

DIRECT LINKS
1) Harvard Dataverse:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VM4OR3

2) Open data paper:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12152559/

3) Related PeerJ paper and supplementary Excel/code:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12453786/

IMPORTANT LIMITATIONS
- The cohort is from Iraq, not India.
- Participants reportedly had joint symptoms for more than six months.
- Do not claim this is a verified early-stage RA dataset unless the original dataset explicitly provides onset/diagnosis timing labels.
- The target is multi-class rheumatic/autoimmune diagnosis; it can support RA identification/differential diagnosis but not automatically future-onset prediction.
- Check class counts, missingness, duplicated rows, exact column names, license, and possible label leakage before modeling.
- The published related paper already applies ML and explainability to this dataset. The project should not claim that applying ML to this dataset is entirely novel.

HOW TO DOWNLOAD
Option A (recommended): Open the Harvard Dataverse link above, choose the latest published version, and download the dataset file.
Option B: Run download_dataset.py in a Python environment with internet access. It queries the Dataverse API for file metadata and downloads available files. If API access fails, use the browser link above.

This kit contains the downloader and source notes, not the clinical XLSX itself; this session could not retrieve the binary dataset file directly.
