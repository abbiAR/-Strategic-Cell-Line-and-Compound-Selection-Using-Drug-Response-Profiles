import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, pearsonr
import shap
from sklearn.metrics import mean_squared_error as MSE
import random
import warnings
warnings.filterwarnings('ignore')

def get_omics(dfd):
    omics = pd.read_csv('cell_expression_1var.csv') # read in mRNA expression data (log2(TPM+1))
    omics.set_index('Unnamed: 0', inplace=True)
    common=dfd.index.intersection(omics.index)
    omics=omics.loc[common]
    return omics

def get_dfd():
    dfd=pd.read_csv('GDSC_1_2_non_redundant_drug_responses.csv')
    dfd.set_index('Unnamed: 0', inplace=True)
    lid = dfd.isna().sum()
    quality_drug=lid.loc[lid<97].index # less than 10% missing values
    lip= dfd[quality_drug].T.isna().sum()
    quality_pat=lip.loc[lip<30].index #less than 10% missing values
    dfd=dfd.loc[quality_pat][quality_drug]
    return dfd


dfd=get_dfd()
omics = get_omics(dfd)
dfd=dfd.loc[omics.index]


##CHOSE DRUG AND ASSOCIATED PREDICTED CELL LINES BELOW

#Refametinib and predicted high (pos_cell) and low responders (neg_cell)
drugs = ['Refametinib']
pos_cell=['WM278', 'NCI-H747', 'HT-29', 'HT-144', 'LB2518-MEL', 'KY821', 'IGR-37', 'UACC-62', 'CHP-212', 'DU-4475', 'NCI-H1836', 'DU-145', 'J82', 'COLO-680N', 'SNU-423', 'SISO', 'HT-1376', 'DBTRG-05MG', 'SNG-M', 'HCC1954']
neg_cell=['EoL-1-cell', 'BT-474', 'VMRC-RCZ', 'SNU-5', 'OCUM-1', 'C32', 'AsPC-1','HCC-827', 'NCI-H322M', 'D-283MED', 'EW-13', 'RL', 'LAMA-84', 'GDM-1',       'MKN45', 'A549', 'NCI-H209', 'COLO-829', 'MOLM-13', 'IST-SL2', 'BICR22','OVMIU', 'MY-M12', 'EBC-1', 'CTB-1', 'ESO26', 'DK-MG', 'MRK-nu-1','COLO-205', 'NB4', 'A204', 'OVCAR-5', 'EB2', 'NH-12', 'IGROV-1', 'HuO9','DB']

#PD0325901 and predicted high (pos_cell) and low responders (neg_cell)
##drugs = ['PD0325901']
##pos_cell=['NCI-H727', 'LoVo', 'NCI-H460', 'Hep3B2-1-7', 'HT-29', 'HT-144', 'IGR-37', 'UACC-62', 'CHP-212', 'DU-4475', 'EFM-19', 'COLO-680N', 'NCI-H661', 'NCI-H1836', 'J82', 'HT-1376', 'HCC1599', 'LP-1', 'GB-1', 'HuO-3N1']
##neg_cell=['EoL-1-cell', 'BT-474', 'VMRC-RCZ', 'NCI-N87', 'SNU-5', 'OCUM-1', 'HCC-827', 'NCI-H322M', 'C32', 'D-283MED', 'EW-13', 'LAMA-84', 'GDM-1', 'RL', 'MKN45', 'COLO-829', 'NCI-H209', 'A549', 'MOLM-13', 'IST-SL2', 'BICR22', 'OVMIU', 'MY-M12', 'EBC-1', 'ESO26', 'CTB-1', 'DK-MG', 'MRK-nu-1', 'COLO-205', 'A204', 'OVCAR-5', 'NB4', 'EB2', 'NH-12', 'IGROV-1', 'HuO9', 'DB', 'MC-IXC']

#QL-X-138 and predicted high (pos_cell) and low responders (neg_cell)
##drugs = ['QL-X-138']
##pos_cell=['THP-1', 'KMS-12-BM', 'KCL-22', 'ST486', 'KARPAS-620', 'SU-DHL-4', 'A3-KAW', 'PF-382', 'CESS', 'SU-DHL-16', 'Hs-578-T', 'huH-1', 'NCI-H460', 'FTC-133', 'LB2241-RCC', 'HT55', 'HT-1376', 'VMRC-MELG', 'SW684', 'RO82-W-1']
##neg_cell=['EoL-1-cell', 'BT-474', 'VMRC-RCZ', 'OCUM-1', 'NCI-N87', 'SNU-5', 'AsPC-1', 'NCI-H322M', 'C32', 'D-283MED', 'EW-13', 'SH-4', 'GDM-1', 'LAMA-84', 'RL', 'MKN45', 'NCI-H209', 'COLO-829', 'A549', 'IST-SL2', 'OVMIU', 'MY-M12', 'BICR22', 'EBC-1', 'ESO26', 'CTB-1', 'DK-MG', 'MRK-nu-1', 'COLO-205', 'NB4', 'OVCAR-5', 'A204', 'Farage', 'NH-12', 'HuO9', 'DB', 'EW-7']

for col in drugs:
    y=dfd[col].copy()
    y.dropna(how='any', inplace=True)

    cell_panel=pos_cell+neg_cell

    X=omics.copy()
    common=list(set(X.index)&set(y.index))
    y=y.loc[common]
    X=X.loc[common]
    
    Xtr=X.loc[cell_panel]#.sample(frac=0.7)#
    Xts=X.drop(Xtr.index)
    ytr=y.loc[Xtr.index]
    yts=y.loc[Xts.index]

    cr=Xtr.corrwith(ytr).sort_values() # correlation matrix, drug target values and omics
    cr_abs=abs(cr)
    abs_idx=list(cr_abs.dropna().sort_values().iloc[-500:].index) #use top 500 correlated genes for analysis

    random.shuffle(abs_idx)
    Xtr=Xtr[abs_idx]
    Xts=Xts[abs_idx]
    

    rf=RandomForestRegressor(n_estimators=200)
    rf.fit(Xtr,ytr)

##    #predictive performance
##    yp=rf.predict(Xts)
##    print(spearmanr(yts,yp), pearsonr(yts,yp))
##    print('MSE', MSE(yts,yp))


    #feature importances    

    # print 30 most important features
    feat_imps=pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values().iloc[-30:] 
    print(feat_imps)
    print(feat_imps.index)

    #top and bottom 25 response correlated genes
    print(cr.dropna().iloc[-25:].index) 
    print(cr.dropna().iloc[:25].index)

    #SHAPley feature importance analysis
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xts)    
    shap_t=Xts
    shap.summary_plot(shap_values, shap_t, show=True)
    #shap_df = pd.DataFrame(shap_values, columns=shap_t.columns)



