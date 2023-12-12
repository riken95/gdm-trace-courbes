import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

repertoire = 'fichiers'
cols = ['mm','N','Def','Con']

def treat_csv(df):
    df = df[cols]
    for col in cols:
        df[col] = df[col].str.replace(',','.')
        df[col] = df[col].astype(np.float64)
    return df

def average_curve(dfs):
    x_min = np.min([np.min(x['Def']) for x in dfs])
    x_max = np.max([np.max(x['Def']) for x in dfs])

    x = np.linspace(x_min,x_max,200)

    fs = []

    for df in dfs:
        f = interp1d(df['Def'],df['Con'],fill_value='extrapolate')
        fs.append(f)
    
    ys = []
    for f in fs:
        ys.append(f(x))
    
    y = np.mean(ys,axis=0)
    return x, y

def lire_csv(chemin:str):
    essaie = pd.read_csv(chemin)
    essaie = essaie.drop(essaie.index[0])
    essaie.columns = cols
    essaie = treat_csv(essaie)
    plt.plot(essaie['Def'],essaie['Con'])



for nom_fichier in os.listdir(repertoire):
    chemin_complet = os.path.join(repertoire,nom_fichier)

    if os.path.isfile(chemin_complet):
        lire_csv(chemin_complet)
    
"""

plt.subplot(1,3,1)

# plt.plot(pla_exl_1['Def'],pla_exl_1['Con'],label='ExL 1',color='tab:blue')
# plt.plot(pla_exl_2['Def'],pla_exl_2['Con'],label='ExL 2',color='tab:blue')
# plt.plot(pla_exl_3['Def'],pla_exl_3['Con'],label='ExL 3',color='tab:blue')
x, avg = average_curve([pla_exl_1,pla_exl_2,pla_exl_3])
plt.plot(x,avg,label='ExL Avg',color='tab:blue')

# plt.plot(pla_loxl_1['Def'],pla_loxl_1['Con'],'--',label='LoxL 1',color='tab:red')
# plt.plot(pla_loxl_2['Def'],pla_loxl_2['Con'],'--',label='LoxL 2',color='tab:red')
# plt.plot(pla_loxl_3['Def'],pla_loxl_3['Con'],'--',label='LoxL 3',color='tab:red')
x, avg = average_curve([pla_loxl_1,pla_loxl_2,pla_loxl_3])
plt.plot(x,avg,label='LoxL Avg',color='tab:red')

plt.plot(pla_lxe_1['Def'],pla_lxe_1['Con'],'--',label='LoxE 1',color='tab:green')
# plt.plot(pla_lxe_2['Def'],pla_lxe_2['Con'],'--',label='LxE 2',color='tab:green')
# plt.plot(pla_lxe_3['Def'],pla_lxe_3['Con'],'--',label='LxE 3',color='tab:green')
x, avg = average_curve([pla_lxe_2,pla_lxe_3])
plt.plot(x,avg,label='LoxE Avg',color='tab:green')

plt.xlabel('Deformation')
plt.ylabel('Contrainte')
plt.ylim([0,65])
plt.xlim([0,8])
plt.legend()
plt.title('PLA Pigmenté')

plt.subplot(1,3,2)

poudre_1 = pd.read_csv('poudre_1.csv')
poudre_1 = treat_csv(poudre_1)

poudre_2 = pd.read_csv('poudre_2.csv')
poudre_2 = treat_csv(poudre_2)

poudre_1l = pd.read_csv('poudre_1l.csv')
poudre_1l = treat_csv(poudre_1l)

poudre_2l = pd.read_csv('poudre_2l.csv')
poudre_2l = treat_csv(poudre_2l)

# plt.plot(poudre_1['Def'],poudre_1['Con'],'--',label='Poudre 1',color='tab:blue')
# plt.plot(poudre_2['Def'],poudre_2['Con'],'--',label='Poudre 2',color='tab:blue')

x, avg = average_curve([poudre_1,poudre_2])

plt.plot(x,avg,label='Moyenne',color='tab:blue')
plt.legend()
plt.title('Poudre')
plt.xlabel('Deformation')
plt.ylabel('Contrainte')
plt.ylim([0,65])

resine_45_1 = pd.read_csv('resine_45_1.csv')
resine_45_1 = treat_csv(resine_45_1)

resine_45_2 = pd.read_csv('resine_45_2.csv')
resine_45_2 = treat_csv(resine_45_2)

resine_exl_2 = pd.read_csv('resine_exl_2.csv')
resine_exl_2 = treat_csv(resine_exl_2)

resine_exlo_1 = pd.read_csv('resine_exlo_1.csv')
resine_exlo_1 = treat_csv(resine_exlo_1)

resine_exlo_2 = pd.read_csv('resine_exlo_2.csv')
resine_exlo_2 = treat_csv(resine_exlo_2)

resine_lxlo_1 = pd.read_csv('resine_lxlo_1.csv')
resine_lxlo_1 = treat_csv(resine_lxlo_1)

resine_lxlo_2 = pd.read_csv('resine_lxlo_2.csv')
resine_lxlo_2 = treat_csv(resine_lxlo_2)

plt.subplot(1,3,3)

# plt.plot(resine_45_1['Def'],resine_45_1['Con'],label='Resine 45 1',color='tab:blue')
# plt.plot(resine_45_2['Def'],resine_45_2['Con'],label='Resine 45 2',color='tab:blue')
x, avg = average_curve([resine_45_1,resine_45_2])
plt.plot(x,avg,label='45 Avg',color='tab:blue')

plt.plot(resine_exl_2['Def'],resine_exl_2['Con'],label='Resine ExL 2',color='tab:red')

# plt.plot(resine_exlo_1['Def'],resine_exlo_1['Con'],label='Resine ExLo 1',color='gold')
# plt.plot(resine_exlo_2['Def'],resine_exlo_2['Con'],label='Resine ExLo 2',color='gold')
x, avg = average_curve([resine_exlo_1,resine_exlo_2])
plt.plot(x,avg,label='ExLo Avg',color='gold')

# plt.plot(resine_lxlo_1['Def'],resine_lxlo_1['Con'],label='Resine LxLo 1',color='tab:green')
# plt.plot(resine_lxlo_2['Def'],resine_lxlo_2['Con'],label='Resine LxLo 2',color='tab:green')
x, avg = average_curve([resine_lxlo_1,resine_lxlo_2])
plt.plot(x,avg,label='LxLo Avg',color='tab:green')

plt.xlabel('Deformation')
plt.ylabel('Contrainte')
plt.ylim([0,65])
plt.legend()
plt.title('Resine')

plt.tight_layout()
"""
plt.show()