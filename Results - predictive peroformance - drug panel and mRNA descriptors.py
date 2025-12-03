import random
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE


def panel_selection(dff, cor_n, vari_thresh, state):
    df_c=dff.copy()
    vari = df_c.T.var().sort_values(ascending=False) #calculate variability and sort values
    half=int(len(vari)/vari_thresh)
    topX = vari.iloc[:half].index
    df_c= df_c.loc[topX]# keep upper half of samples (higher variability)

    corr=df_c.T.corr('pearson')# correlation matrix
    out=[]
    approved=[]
    reorder=(corr.median().sort_values(ascending=True).index) # sort samples based on their median correlation with other samples
    corr=corr[reorder]
    corr=corr.loc[reorder] #reshape new matrix based on this order
    
    drugs = list(corr.columns) # list drugs and start iterating for panel 
    for d in drugs:
        if d not in out:
            approved.append(d)
            line=corr[d]
            line=line.loc[line>cor_n] # correlation coefficient cutoff, drugs that are more than 0p2 pearson are thrown away
            out=out+list(line.index)

    return(approved)


def get_omics(dfd):
    omics=pd.read_csv('PCA_omics.csv')
    omics.set_index('Unnamed: 0', inplace=True)
    return omics


def get_dfd():
    dfd=pd.read_csv('GDSC_1_2_non_redundant_drug_responses.csv')
    dfd.set_index('Unnamed: 0', inplace=True)
    mid = dfd.isna().sum()
    quality_drug=mid.loc[mid<97].index # less than 10% missing values for drugs
    mic= dfd[quality_drug].T.isna().sum()
    quality_pat=mic.loc[mic<30].index #less than 10% missing values for cell lines
    dfd=dfd.loc[quality_pat][quality_drug]
    return dfd


#READ DATASETS
dfd=get_dfd()
omics = get_omics(dfd)
dfd=dfd.loc[omics.index]


#Separating datasets into train, test and validation:
seed1=963598
seed2=554250
seed3=87183

all_patients_idx = list(dfd.sample(frac=1, random_state=seed1).index)
test_patients = ['KY821', 'ST486', 'IHH-4', 'JVM-2', 'MKN7', 'CESS', 'LK-2', '8505C', 'HEC-1', 'SCC-15', 'HC-1', 'NALM-6', 'GB-1', 'U-118-MG', 'CS1', 'CAL-78', 'YKG-1', 'HSC-3', 'NCI-H2731', 'EMC-BAC-2', 'SNU-423', 'SK-HEP-1', 'NB14', 'Hep3B2-1-7', 'SNU-407', 'NCI-H747', 'GAK', 'HCC-15', 'LB831-BLC', 'COR-L105', 'IST-SL1', 'NCI-H1568', 'HDLM-2', 'RERF-GC-1B', 'MES-SA', 'A427', 'NCI-H1734', 'JVM-3', 'T24', 'AU565', 'SU-DHL-4', '42-MG-BA', 'KYSE-180', 'KMS-12-BM', 'NCI-H2795', 'NCI-H513', 'A3-KAW', 'KCL-22', 'NCI-H661', 'NCI-H2869', 'D-247MG', '639-V', 'NB13', 'Hs-578-T', 'HT-1376', 'SK-N-DZ', 'HCC-44', 'VMRC-MELG', 'H-EMC-SS', 'NCI-H1299', 'SBC-1', 'UACC-62', 'SBC-5', 'KNS-42', 'RH-1', 'THP-1', 'WM1552C', 'HuO-3N1', 'SW1088', 'LB2241-RCC', 'SJRH30', 'NCI-H2369', 'HCC2157', 'LB2518-MEL', 'SW684', 'LS-411N', 'NCI-H727', 'NCI-H2228', 'LS-180', 'C-4-I', 'ES-2', 'LoVo', 'J82', 'EFM-19', 'HCC1599', 'SAS', 'MDA-MB-361', 'D-566MG', 'WM278', 'huH-1', 'OV-17R', 'EFO-27', 'CAL-27', 'COLO-678', 'SK-LMS-1', 'HPAF-II', 'SCC-9', 'UACC-893', 'CHP-212', 'SK-MEL-3', 'NCI-H2373', 'SISO', 'FTC-133', 'SW1417', 'NCI-H1836', 'NCI-H460', 'NCI-H526', '786-0', 'SW48', 'EM-2', 'IGR-37', 'DU-4475', 'NB10', 'FLO-1', 'SK-MM-2', 'U031', 'LNCaP-Clone-FGC', 'VA-ES-BJ', 'CAL-54', 'NB6', 'COLO-680N', 'SNU-1', 'MEG-01', 'HDQ-P1', 'PA-TU-8988T', 'SK-MG-1', 'PF-382', 'VCaP', 'NCI-H2170', 'TT2609-C02', 'RKN', 'SK-PN-DW', 'HT55', 'HT-144', 'U-698-M', 'NUGC-3', 'JEG-3', 'CCK-81', 'SNU-387', 'OACp4C', 'HT-29', 'DU-145', 'OE19', 'IA-LM', 'MHH-PREB-1', 'VAL', 'TE-8', 'HT-1197', 'SF539', 'RCC-JW', 'G-401', 'DBTRG-05MG', 'LP-1', 'SNG-M', 'KARPAS-620', 'HCC1954', 'RO82-W-1', 'SU-DHL-16', 'KNS-81-FD', 'BEN']
#test_patients = random.Random(seed2).sample(all_patients_idx, 160) # 20%
remaining_patients = [item for item in all_patients_idx if item not in test_patients]
#validation_patients=random.Random(seed3).sample(remaining_patients, 80) # 10%
validation_patients=['SUP-HD1', 'HT-1080', 'LAN-6', 'SKG-IIIa', 'NCI-H2122', 'NCI-H1437', 'LU-135', 'CaR-1', 'HCC1187', 'KLE', 'D-336MG', 'KOSC-2', 'SF268', 'SW948', 'RCC-JF', 'OCUB-M', 'NCI-H2722', 'LOUCY', 'Daoy', 'A388', 'ME-1', 'A2780', 'KYSE-140', 'SLVL', 'JHH-1', 'HCC-78', 'KP-4', 'PE-CA-PJ15', 'TK10', 'MDA-MB-231', 'MZ2-MEL', 'MZ7-mel', 'ONS-76', 'HT', 'HSC-4', 'NCI-H522', 'Ramos-2G6-4C10', 'SR', 'GRANTA-519', 'CAL-39', 'ROS-50', 'NCI-H2052', 'SK-UT-1', 'NCI-H1435', 'ESS-1', 'NCI-H1975', 'HOP-62', 'CHL-1', 'HCC1500', 'U-87-MG', 'ATN-1', 'A101D', 'CA46', 'NY', 'ARH-77', 'Hey', 'OVCA420', 'SBC-3', 'P32-ISH', 'RPMI-6666', 'BFTC-909', 'KM12', 'NCI-H1876', 'TOV-21G', 'GOTO', 'MC116', 'NCI-H1651', 'SK-ES-1', 'NCI-H1048', 'SU-DHL-10', 'LC-2-ad', 'A431', 'OCI-AML2', 'SW620', 'NEC8', 'KP-N-YN', 'JHU-011', 'MS751', 'DAN-G', 'MOLT-4'] #10%
train_patients = [item for item in remaining_patients if item not in validation_patients]



#COMPARING DRUG PANELS AND mRNA DESCRIPTORS ON MODEL PERFORMANCE

results=[]
for i in range(5): # five experiments 
    dcc={}
    print('completed:', i)
    test_drugs=list(dfd.T.sample(frac=1, random_state=420).index) #randomise sample order    
    seed4=random.randint(0,1000000)
    seed5=random.randint(0,1000000)
    trp=random.Random(seed5).sample(train_patients, 509) # select subset of patients from training data to use for panel extraction

    for drug in test_drugs[:5]: # for each drug
        seed6=random.randint(0,1000000) 
        train_drugs = list(dfd.T.drop(drug).index)
        #select drug panel, adjust second parameter for panel size
        drug_panel = panel_selection(dfd.loc[trp][train_drugs].T, 0.55, 2, seed6) 

        Xd=dfd[drug_panel].copy() #drug panel descriptors
        Xr=omics.copy() #omics descriptors
        seed7=random.randint(0,1000000)
        #select cell line panel, adjust second parameter for panel size
        sub_train_patients=panel_selection(dfd.loc[trp], 0.72, 2, seed7) 


        #TARGET VALUES
        y=dfd[drug].copy()
        ytr= y.loc[sub_train_patients]
        ytr.dropna(how='any', inplace=True)
        yts= y.loc[validation_patients]
        yts.dropna(how='any', inplace=True)

        #drug panel values for model training
        Xdtr = Xd.loc[ytr.index]
        Xdtr.fillna(Xdtr.mean(), inplace=True)
        Xdts = Xd.loc[yts.index]
        Xdts.fillna(Xdts.mean(), inplace=True)

        #omics values for model training
        Xrtr = Xr.loc[ytr.index]
        Xrts = Xr.loc[yts.index]
        runs = [[Xdtr, Xdts, 'drug'], [Xrtr, Xrts,'rna']] #comparing drug panel and omics descriptors

        for run in runs: # for each run (drug panel vs mRNA descriptors):
            rf=GradientBoostingRegressor(max_depth = 2, n_estimators =100, random_state=seed6)
            rf.fit(run[0],ytr)
            yp=rf.predict(run[1])

            #save results to dictionary for each descriptor and experiment
            key=run[2]+'_'+str(i)
            current = dcc.get(key)
            if current == None:
                dcc[key]=[list(yp),list(yts)]
            else:
                dcc[key]=[current[0]+list(yp), current[1]+list(yts)]

    #calculate metrics for each of the five experiment             
    for k in dcc.keys():
        lyse = pd.DataFrame(dcc.get(k)).T.round(3)
        pear=np.round(pearsonr(lyse[0],lyse[1])[0],3)
        spear=np.round(spearmanr(lyse[0],lyse[1])[0],3)
        mse=np.round(MSE(lyse[0],lyse[1]),3)
        rmse=np.round(np.sqrt(MSE(lyse[0],lyse[1])),3)
        mae=np.round(MAE(lyse[0],lyse[1]),3)
        results.append([k, pear, spear, mse, rmse, mae])


#present result in a dataframe
results_df = pd.DataFrame(results)
results_df.set_index(0, inplace=True)
results_df.columns = ['pearson', 'spearman', 'MSE','RMSE', 'MAE']
