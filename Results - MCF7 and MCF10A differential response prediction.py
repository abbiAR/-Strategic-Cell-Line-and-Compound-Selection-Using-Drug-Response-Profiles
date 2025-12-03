import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error as MSE
import random, warnings
import numpy as np
warnings.filterwarnings("ignore")


df3=pd.read_csv('MCF7 and MCF10A differential prediction dataset.csv')
df3.set_index('Unnamed: 0', inplace=True)


#cells we want to predict drug library against
test_cells=['MCF7','MCF10A']
#drug library that we want to predict values for in test cell lines MCF7 and MCF10A
test_drugs=['Pictilisib', 'Flavopiridol', 'Navitoclax', 'WIKI4', 'Lapatinib', 'Dasatinib', 'JQ1', 'CZC24832', 'P22077', 'EPZ5676', 'GSK1904529A', 'Trametinib', 'BI-2536', 'AZD7762', 'Wnt-C59', 'NVP-ADW742', 'Linsitinib', 'WZ4003', 'XAV939', 'AZD5582', 'OSI-027', 'RVX-208', 'Cediranib', 'Telomerase Inhibitor IX', 'Olaparib', '5-Fluorouracil', 'GSK2606414', 'Buparlisib', 'Uprosertib', 'Rapamycin', 'Tamoxifen', 'Leflunomide', 'ZM447439', 'PRIMA-1MET', 'GNE-317', 'LY2109761', 'OF-1', 'LGK974', 'Afuresertib', 'Entospletinib', 'GSK343', 'AZD6482', 'Erlotinib', 'BMS-536924', 'GDC0810', 'KU-55933', 'AGI-6780', 'Palbociclib', 'Teniposide', 'MK-1775', 'Ibrutinib', 'MG-132', 'Entinostat', 'Sabutoclax', 'AZD5991', 'LCL161', 'Axitinib', 'Foretinib', 'Tozasertib', 'Acetalax', 'Etoposide', 'LJI308', 'AZD8055', 'Dihydrorotenone', 'PF-4708671', 'Sorafenib', 'YK-4-279', 'Alisertib', 'Ulixertinib', 'GSK2578215A', 'MK-8776', 'Dactolisib', 'Pemetrexed', 'IWP-2', 'AZD1480', 'Selumetinib', 'MN-64', 'I-BET-762', 'Crizotinib', 'Venetoclax', 'AZD2014', 'AMG-319', 'AZD1208', 'Nelarabine', 'Fludarabine', 'Afatinib', 'Ipatasertib', 'Oxaliplatin', 'UMI-77', 'Gefitinib', 'Pazopanib', 'AZD8186', 'BMS-345541', 'VE821', 'MIM1', 'Temozolomide', 'PLX-4720', 'VX-11e', 'AT13148', 'SB505124', 'Taselisib', 'Dinaciclib', 'Mitoxantrone', 'Alpelisib', 'AZD5438', 'Ribociclib', 'Ponatinib', 'EPZ004777', 'Nilotinib', 'PCI-34051', 'Ruxolitinib', 'Gemcitabine', 'OTX015', 'AZD3759', 'Cyclophosphamide', 'AZ960', 'Osimertinib', 'Obatoclax Mesylate', 'SB216763', 'Docetaxel', 'ABT737', 'AZD6738', 'Luminespib', 'ML323', 'Epothilone B', 'AZD4547', 'GSK269962A', 'Cisplatin', 'Talazoparib', 'Dabrafenib', 'NU7441', 'PD0325901', 'Doramapimod', 'Vorinostat', 'BIBF-1120', 'Bortezomib', 'Cytarabine', 'Pevonedistat', 'Bosutinib', 'Staurosporine', 'BMS-754807', 'Camptothecin', 'I-BRD9', 'GSK591', 'Sunitinib', 'RO-3306', 'Vismodegib', 'SCH772984', 'BIBR-1532', 'Fulvestrant', 'Sepantronium bromide', 'AZ6102', 'Paclitaxel', 'PD173074', 'Irinotecan', 'Niraparib', 'AGI-5198']
#Drug panel: drugs that we have predicted (MCF7) or measured (MCF10A) values for. 
known_drugs = ['CHEMBL120697', 'CHEMBL4078014', 'CHEMBL94710', 'CHEMBL429747', 'CHEMBL261492', 'CHEMBL4645966', 'CHEMBL2430269', 'CHEMBL4852772', 'CHEMBL4438250', 'CHEMBL4278703', 'CHEMBL3137331', 'CHEMBL4278651', 'CHEMBL1824185', 'CHEMBL554741', 'CHEMBL44918', 'CHEMBL1163916', 'CHEMBL4789605', 'CHEMBL4444862', 'CHEMBL4750849', 'CHEMBL50', 'CHEMBL137828', 'CHEMBL4064997', 'CHEMBL4080242', 'CHEMBL2071220', 'CHEMBL4555294', 'CHEMBL4281455', 'CHEMBL4226953', 'CHEMBL4167015', 'CHEMBL4081255', 'CHEMBL5196880', 'CHEMBL2430282', 'CHEMBL4442413', 'CHEMBL319204', 'CHEMBL4544866', 'CHEMBL4526315', 'CHEMBL4642991', 'CHEMBL4290798', 'CHEMBL4469292', 'CHEMBL5206140', 'CHEMBL1822705', 'CHEMBL4579601', 'CHEMBL2071143', 'CHEMBL2071144', 'CHEMBL2071218', 'CHEMBL4641062', 'CHEMBL96581', 'CHEMBL4743389', 'CHEMBL328253', 'CHEMBL4875764', 'CHEMBL4852946', 'CHEMBL4853095', 'CHEMBL4461112', 'CHEMBL4211706', 'CHEMBL4646467', 'CHEMBL4643967', 'CHEMBL2430279', 'CHEMBL2430271', 'CHEMBL4176400', 'CHEMBL4166442', 'CHEMBL4643611', 'CHEMBL343071', 'CHEMBL1097830', 'CHEMBL4162600', 'CHEMBL2071219', 'CHEMBL2071222', 'CHEMBL22969', 'CHEMBL2431002', 'CHEMBL2430281', 'CHEMBL137829', 'CHEMBL96817', 'CHEMBL2430276', 'CHEMBL94637', 'CHEMBL1098149', 'CHEMBL4237922', 'CHEMBL4162342', 'CHEMBL4169862', 'CHEMBL4288267', 'CHEMBL4590434', 'CHEMBL4846984', 'CHEMBL142135', 'CHEMBL489', 'CHEMBL5081446', 'CHEMBL137897', 'CHEMBL4868545', 'CHEMBL4789526', 'CHEMBL93371', 'CHEMBL4163642', 'CHEMBL4289246', 'CHEMBL4283507', 'CHEMBL4853299', 'CHEMBL5192197', 'CHEMBL4539712', 'CHEMBL4568783', 'CHEMBL4788923', 'CHEMBL4521935', 'CHEMBL4452083', 'CHEMBL4756675', 'CHEMBL5073857', 'CHEMBL4536294', 'CHEMBL2380472', 'CHEMBL4532601', 'CHEMBL4852654', 'CHEMBL4856651', 'CHEMBL4874138', 'CHEMBL4170234', 'CHEMBL4294194', 'CHEMBL5175350', 'CHEMBL4175541', 'CHEMBL5090553', 'CHEMBL176599', 'CHEMBL4778388', 'CHEMBL4172831', 'CHEMBL4161633', 'CHEMBL4439260', 'CHEMBL4643815', 'CHEMBL2430270', 'CHEMBL3236392', 'CHEMBL440529', 'CHEMBL4441885', 'CHEMBL2430272', 'CHEMBL4177481', 'CHEMBL1337170', 'CHEMBL4288543', 'CHEMBL4291316', 'CHEMBL4471915', 'CHEMBL4850487', 'CHEMBL4854050', 'CHEMBL1631464', 'CHEMBL4861751', 'CHEMBL4458550', 'CHEMBL4855587', 'CHEMBL4878354', 'CHEMBL4437722', 'CHEMBL4648843', 'CHEMBL4171106', 'CHEMBL4161502', 'CHEMBL4173292', 'CHEMBL1208572', 'CHEMBL2426504', 'CHEMBL4593654', 'CHEMBL4438750', 'CHEMBL4439138', 'CHEMBL2430274', 'CHEMBL4649534', 'CHEMBL4164923', 'CHEMBL4280344', 'CHEMBL4633297', 'CHEMBL4643144', 'CHEMBL2426506', 'CHEMBL486997', 'CHEMBL4166091', 'CHEMBL98', 'CHEMBL4086093', 'CHEMBL1822709', 'CHEMBL4848298', 'CHEMBL5286671', 'CHEMBL5172561', 'CHEMBL5209175', 'CHEMBL2430277', 'CHEMBL3236397', 'CHEMBL4278403', 'CHEMBL4787889', 'CHEMBL4592182', 'CHEMBL5201919', 'CHEMBL5278131', 'CHEMBL3331077', 'CHEMBL1186460', 'CHEMBL1190487', 'CHEMBL4635263', 'CHEMBL4633675', 'CHEMBL4646128', 'CHEMBL1958076', 'CHEMBL4760016', 'CHEMBL4853768', 'CHEMBL4856633', 'CHEMBL210635', 'CHEMBL4795528', 'CHEMBL5283128', 'CHEMBL5266154', 'CHEMBL4552642', 'CHEMBL4439525', 'CHEMBL5200485', 'CHEMBL5184181', 'CHEMBL5189115', 'CHEMBL4540105', 'CHEMBL4778965', 'CHEMBL4854779', 'CHEMBL5170465', 'CHEMBL5190895', 'CHEMBL9470', 'CHEMBL5170139', 'CHEMBL2326743', 'CHEMBL5176855', 'CHEMBL4778092', 'CHEMBL4163767', 'CHEMBL5169594', 'CHEMBL5175126', 'CHEMBL5188773', 'CHEMBL5084288', 'CHEMBL5199842', 'CHEMBL65', 'CHEMBL5179947', 'CHEMBL5197276', 'CHEMBL129795', 'CHEMBL5075190', 'CHEMBL5291443', 'CHEMBL5289713', 'CHEMBL5275352', 'CHEMBL5278626', 'CHEMBL5276336', 'CHEMBL4864239', 'CHEMBL5092817', 'CHEMBL5070510', 'CHEMBL5200039', 'CHEMBL5197834', 'CHEMBL5208904', 'CHEMBL5270178', 'CHEMBL4171830', 'CHEMBL5094712', 'CHEMBL5082245', 'CHEMBL359744', 'CHEMBL5201046', 'CHEMBL5189179']



predicted=[]
for test_cell in test_cells:
    pred=[]
    true=[]
    for drug in test_drugs: #predict for each test drug
        y=df3[drug].copy()
        X=df3[known_drugs].copy()
        X.fillna(X.mean(), inplace=True)

        Xts=X.loc[test_cell]
        Xtr=X.drop(test_cells) #dropping both MCF7 and MCF10A, since we predict values for both

        yts=y.loc[test_cell]
        ytr=y.drop(test_cells)
        ytr.dropna(how='any', inplace=True)

        Xtr=Xtr.loc[ytr.index]
        Xtr.fillna(Xtr.mean(),inplace=True)
        
        rf=GradientBoostingRegressor(n_estimators=100, max_depth=4)
        gs=rf.fit(Xtr,ytr)
        yp=rf.predict(np.array(Xts).reshape(1,-1))

        pred.append(np.round(yp[0],2))
        
    predicted.append(pred)
    

dfp=pd.DataFrame([predicted[0], predicted[1]], columns=test_drugs).T
dfp.columns= ['MCF7_predicted','MCF10A_predicted']
dfp['delta']=dfp['MCF7_predicted']-dfp['MCF10A_predicted']
dfp.sort_values('delta', inplace=True)
