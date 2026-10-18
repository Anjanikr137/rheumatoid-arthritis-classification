RA DATASET AUDIT KIT — READ FIRST
=================================

PRIMARY SOURCE
Title: Rheumatic and autoimmune diseases dataset
Repository: Harvard Dataverse
Persistent DOI: 10.7910/DVN/VM4OR3
Direct page:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VM4OR3

SOURCE PAPER
Mahdi, M. F., Jahani, A., & Abd, D. H. (2025). Diagnosis of rheumatic and autoimmune diseases dataset.
Data in Brief, 60, 111623. DOI: 10.1016/j.dib.2025.111623
https://doi.org/10.1016/j.dib.2025.111623

RELATED PAPERS THAT MUST BE READ BEFORE CLAIMING NOVELTY
1. Mahdi et al. (2025), PeerJ Computer Science, 11:e3096.
   https://doi.org/10.7717/peerj-cs.3096
2. Elmesiry et al. (published Dec 2025), BMC Rheumatology.
   https://doi.org/10.1186/s41927-025-00607-7

REPORTED PROFILE
- 12,085 records, 14 reported features, 7 diagnostic classes.
- Collection described as one hospital and three laboratories in Iraq, 2019–2024.
- Reported RA records: 2,848 (23.6%).
- Dataset record lists CC0 1.0 Universal.
- Features include age, sex and laboratory markers (ESR, CRP, RF, anti-CCP, HLA-B27, ANA, anti-Ro, anti-La, anti-dsDNA, anti-Sm, C3, C4).
- Some lab features have substantial missingness; see the original paper's missingness table.

IMPORTANT SCIENTIFIC CAVEATS
- The cohort is from Iraq, not India.
- Participants reportedly had joint symptoms for >6 months; verified early-stage RA/onset labels are not established by the documentation reviewed.
- Diagnostic labels were assigned by a specialist after review of lab results. Using related biomarkers as predictors can create incorporation/circularity bias.
- The same dataset has already been used in published ML/XAI studies. Do not claim that ML or SHAP on this dataset is new.
- Working project focus: RA-vs-non-RA classification and a predefined feature-ablation/biomarker-dependence audit, subject to checking whether the exact analysis already exists in prior papers.
- The source data file is not included in this kit. Download it from the official repository page.
