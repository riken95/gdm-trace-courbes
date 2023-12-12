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
    #plt.plot(essaie['Def'],essaie['Con'])
    return essaie


plt.title("moyennes totales")
for nom_dossier in os.listdir(repertoire):
    chemin_complet = os.path.join(repertoire,nom_dossier)
    
    dfs = []
    if os.path.isdir(chemin_complet):
        for fichier in os.listdir(chemin_complet):
            chemin_complet_tot = os.path.join(chemin_complet,fichier)
            
            if os.path.isfile(chemin_complet_tot):
                dfs.append(lire_csv(chemin_complet_tot))
    x,y = average_curve(dfs)
    plt.plot(x,y,label=nom_dossier)
plt.legend()
plt.xlabel('Déformation en %')
plt.ylabel('Contrainte (MPa)')
plt.show()
