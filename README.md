# -Strategic-Cell-Line-and-Compound-Selection-Using-Drug-Response-Profiles
Advancing Drug Development Through Strategic Cell Line and Compound Selection Using Drug Response Profiles

Scripts:
1. Results - predictive performance - drug panel and mRNA descriptors: Predicting cell line response to drugs
2. Results - MCF7 and MCF10A differential response prediction: Predicting differential drug response in MCF7 and MCF10A
3. Results - explainability: Generate feature importances, correlative features and shaps bee-swarm plots

Datasets:
GDSC_1_2_non_redundant_drug_responses: GDSC 1 and 2 consolidated dataset with drug responses
PCA_omics: PCA transformed mRNA expression data log2(TPM+1)
MCF7 and MCF10A differential prediction dataset: measured drug respones for MCF10A panel, predicted drug responeses for the same panel across all cell lines (CHEMBL IDs). Dataset also contains GDSC 1 and 2 drug respones for all cell lines (except MCF10A). 
cell_expression_1var: mRNA expression data from sanger cell model passports (log2(TPM+1)). (Threshold for inclusion is dataset.var() > 1, removing half the genes for upload to github, link to original dataset available in manuscript.

