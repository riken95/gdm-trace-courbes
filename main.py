import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
from scipy.interpolate import make_interp_spline



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
    #On renomme les noms des colonnes
    essaie = essaie.drop(essaie.index[0])
    essaie.columns = cols
    #On renvoie les données formatées
    essaie = treat_csv(essaie)
    return essaie

def afficher_categories(): 
    plt.title("Moyennes totales")

    #On parcourt les dossiers
    for nom_dossier in os.listdir(repertoire):
        chemin_complet = os.path.join(repertoire,nom_dossier)
        
        if os.path.isdir(chemin_complet):
            dfs = []
            for nom_sous_dossier in os.listdir(chemin_complet):
                #Pour chaque dossier on fait la moyenne
                sous_chemin_complet = os.path.join(chemin_complet,nom_sous_dossier)

                if os.path.isdir(sous_chemin_complet):
                    for fichier in os.listdir(sous_chemin_complet):
                        chemin_complet_tot = os.path.join(sous_chemin_complet,fichier)
                        
                        if os.path.isfile(chemin_complet_tot):
                            dfs.append(lire_csv(chemin_complet_tot))
            if(len(dfs)):
                x,y = average_curve(dfs)
                plt.plot(x,y,label=nom_dossier)
    #On trace le graphique final
    plt.legend()
    plt.xlabel('Déformation en %')
    plt.ylabel('Contrainte (MPa)')
    plt.show()


def afficher_sous_categories(): 
    plt.title("Moyennes totales")

    #On parcourt les dossiers
    for nom_dossier in os.listdir(repertoire):
        chemin_complet = os.path.join(repertoire,nom_dossier)
        
        if os.path.isdir(chemin_complet):
            
            for nom_sous_dossier in os.listdir(chemin_complet):
                #Pour chaque dossier on fait la moyenne
                sous_chemin_complet = os.path.join(chemin_complet,nom_sous_dossier)

                if os.path.isdir(sous_chemin_complet):
                    dfs = []
                    for fichier in os.listdir(sous_chemin_complet):
                        chemin_complet_tot = os.path.join(sous_chemin_complet,fichier)
                        
                        if os.path.isfile(chemin_complet_tot):
                            dfs.append(lire_csv(chemin_complet_tot))
                    if(len(dfs)):
                        x,y = average_curve(dfs)
                        plt.plot(x,y,label=nom_sous_dossier)
    #On trace le graphique final
    plt.legend()
    plt.xlabel('Déformation en %')
    plt.ylabel('Contrainte (MPa)')
    plt.show()


afficher_sous_categories()
afficher_categories()
