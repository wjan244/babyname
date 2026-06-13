import pandas as pd
import numpy as np
import altair as alt
import os
import webbrowser

alt.data_transformers.disable_max_rows()

alt.data_transformers.enable('json')

data = pd.read_csv('data/dpt2020.csv',sep=";")

df = data[(data["annais"] != "XXXX") & (data["dpt"] != "XX")]

# PRÉPARATION DES DONNÉES
# -------------------------

repartition_sexe = df.groupby(['preusuel', 'sexe'])['nombre'].sum().unstack(fill_value=0)

repartition_sexe['total'] = repartition_sexe.sum(axis=1)
repartition_sexe['part_minoritaire'] = repartition_sexe.drop(columns=['total']).min(axis=1) / repartition_sexe['total']

# Règle des 10%, pour filtrer le bruit et les erreurs d'annotation (Marie en H par exemple)
vrais_epicenes = repartition_sexe[(repartition_sexe['part_minoritaire'] >= 0.10) & (repartition_sexe['total'] >= 500)].index
df_filtre = df[df['preusuel'].isin(vrais_epicenes)]
evolution_annuelle = df_filtre.groupby(['preusuel', 'sexe', 'annais'])['nombre'].sum().reset_index()

evolution_pivot = evolution_annuelle.pivot_table(
    index=['preusuel', 'annais'], 
    columns='sexe', 
    values='nombre', 
    fill_value=0
)

# Filtre de robustesse sur le nombre d'années que le prenom a été donné (si trop faible, la corrélation ne porte pas de sens)
comptage_annees = evolution_annuelle.groupby('preusuel')['annais'].nunique()
prenoms_solides = comptage_annees[comptage_annees >= 10].index
evolution_pivot_filtre = evolution_pivot.loc[prenoms_solides]

correlations = evolution_pivot_filtre.groupby('preusuel').apply(
    lambda x: x.iloc[:, 0].corr(x.iloc[:, 1], min_periods=5)
).reset_index(name='correlation_H_F')

prenoms_epicenes = df_filtre.groupby(['preusuel', 'sexe'])['nombre'].sum().reset_index()
prenoms_epicenes_pop = prenoms_epicenes.groupby(['preusuel'])['nombre'].sum().reset_index()
prenoms_epicenes_pop = prenoms_epicenes_pop.merge(correlations, on='preusuel', how='left')
prenoms_epicenes_pop = prenoms_epicenes_pop.sort_values(by='correlation_H_F', ascending=True)

# PREPA VISU
# ----
evolution_annuelle['nombre_miroir'] = np.where(
    evolution_annuelle['sexe'].isin([1]), 
    evolution_annuelle['nombre'], 
    -evolution_annuelle['nombre']
)
evolution_annuelle['sexe'] = evolution_annuelle['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})
prenoms_epicenes_pop = prenoms_epicenes_pop.rename(columns={'nombre': 'nombre_total'})

# On fusionne pour que toutes les infos soient dans la même table
df_visu = evolution_annuelle.merge(
    prenoms_epicenes_pop[['preusuel', 'nombre_total', 'correlation_H_F']], 
    on='preusuel', 
    how='inner'
)

# CONSTRUCTION DE LA VISU
# -----------------------

# On identifie le prénom le plus donné (en triant par la popularité)
prenom_top = prenoms_epicenes_pop.sort_values(by='nombre_total', ascending=False).iloc[0]['preusuel']

selection = alt.selection_point(
    fields=['preusuel'], 
    empty=False,
    value=[{'preusuel': prenom_top}] # Sélection par défaut
)

# Graphique de gauche (Scatter plot)
scatter_plot = alt.Chart(prenoms_epicenes_pop).mark_circle(size=60).encode(
    x=alt.X('nombre_total:Q', 
            scale=alt.Scale(type='log'), 
            title='Popularité totale (cumul H+F) - Échelle Log'),
    
    y=alt.Y('correlation_H_F:Q', 
            title='Corrélation Homme / Femme (-1 à 1)'),
    
    tooltip=[
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('nombre_total:Q', title='Popularité totale'),
        alt.Tooltip('correlation_H_F:Q', title='Corrélation H/F', format='.2f')
    ],
    
    # Modification des couleurs : Corail si sélectionné, Bleu moderne sinon
    color=alt.condition(selection, alt.value('#e76f51'), alt.value('#4ea8de')),
    
    opacity=alt.condition(selection, alt.value(0.9), alt.value(0.75))
).add_params(
    selection
).properties(
    width=400,
    height=400,
    title='1. Cliquez sur un prénom'
)

# Graphique de droite (Courbes en miroir)
courbes = alt.Chart(df_visu).mark_area(opacity=0.7).encode(
    x=alt.X('annais:Q', title='Année de naissance', axis=alt.Axis(format='d')), 
    y=alt.Y('nombre_miroir:Q', title='Attributions (H en +, F en -)'),
    
    color=alt.Color('sexe:N', 
                    title='Sexe', 
                    scale=alt.Scale(
                        domain=['Homme', 'Femme'], 
                        range=['#2a9d8f', '#e76f51']
                    ),
                    legend=alt.Legend(orient='top')),
    
    detail='preusuel:N',
    
    tooltip=[
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('annais:Q', title='Année'),
        alt.Tooltip('sexe:N', title='Sexe'), 
        alt.Tooltip('nombre:Q', title='Nombre réel (positif)') 
    ]
).transform_filter(
    selection
).properties(
    width=450,
    height=400,
    title='2. Évolution temporelle'
)

tableau_de_bord = scatter_plot | courbes
file = 'tableau_de_bord_epicenes.html'
tableau_de_bord.save(file)
chemin_complet = 'file://' + os.path.realpath(file)
webbrowser.open(chemin_complet)