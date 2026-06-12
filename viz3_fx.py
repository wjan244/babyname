import pandas as pd
import numpy as np
import altair as alt
import os
import webbrowser

alt.data_transformers.disable_max_rows()

alt.data_transformers.enable('json')

data = pd.read_csv('data/dpt2020.csv',sep=";")

df = data[(data["annais"] != "XXXX") & (data["dpt"] != "XX")]

# TOTAUX ANNUELS NATIONAUX PAR SEXE (le dénominateur)
# -------------------------
# Calculé sur TOUT le dataset (tous prénoms), sinon le dénominateur serait faux.
# Donne, pour chaque (année, sexe), le nombre total de naissances en France.
totaux_annuels = df.groupby(['annais', 'sexe'])['nombre'].sum().reset_index()
totaux_annuels = totaux_annuels.rename(columns={'nombre': 'total_naissances_sexe'})

# ============================================================
# MÉTRIQUE GLOBALE 1 : PART DU TOP 1 (par année et par sexe)
# Pour chaque (année, sexe), quel % des naissances représente
# le prénom le plus donné cette année-là.
# Calculé sur TOUS les vrais prénoms (hors agrégats '_').
# ============================================================
df_vrais = df[~df['preusuel'].str.startswith('_')]
# total par (prénom, année, sexe)
totaux_prenom_an = df_vrais.groupby(['annais', 'sexe', 'preusuel'])['nombre'].sum().reset_index()
# pour chaque (année, sexe), on garde la ligne du prénom max
idx_max = totaux_prenom_an.groupby(['annais', 'sexe'])['nombre'].idxmax()
top1 = totaux_prenom_an.loc[idx_max].copy()
top1 = top1.merge(totaux_annuels, on=['annais', 'sexe'], how='left')
top1['part_top1'] = top1['nombre'] / top1['total_naissances_sexe']
top1 = top1.rename(columns={'preusuel': 'prenom_top1'})
# table finale : annais, sexe, part_top1, prenom_top1
top1_metrique = top1[['annais', 'sexe', 'part_top1', 'prenom_top1']]

# PRÉPARATION DES DONNÉES
# -------------------------

repartition_sexe = df.groupby(['preusuel', 'sexe'])['nombre'].sum().unstack(fill_value=0)

repartition_sexe['total'] = repartition_sexe.sum(axis=1)
repartition_sexe['part_minoritaire'] = repartition_sexe.drop(columns=['total']).min(axis=1) / repartition_sexe['total']

# Part masculine globale du prénom : 0 = entièrement féminin, 1 = entièrement masculin.
# Colonne 1 = sexe masculin dans le dataset INSEE.
repartition_sexe['part_masculine'] = repartition_sexe[1] / repartition_sexe['total']

# Règle des 10%, pour filtrer le bruit et les erreurs d'annotation (Marie en H par exemple)
vrais_epicenes = repartition_sexe[(repartition_sexe['part_minoritaire'] >= 0.10) & (repartition_sexe['total'] >= 500)].index
# Exclusion de la catégorie fourre-tout "_PRENOMS_RARES" (et tout préfixe '_'),
# qui n'est pas un vrai prénom mais un agrégat INSEE des prénoms rares.
# Elle reste comptée dans totaux_annuels (le dénominateur), mais ne doit pas
# apparaître comme un "prénom" dans l'analyse.
df_filtre = df[df['preusuel'].isin(vrais_epicenes) & (~df['preusuel'].str.startswith('_'))]
evolution_annuelle = df_filtre.groupby(['preusuel', 'sexe', 'annais'])['nombre'].sum().reset_index()

# Passage en PARTS : on divise par le total national du sexe pour cette année.
# Neutralise la démographie (plus de naissances en 2000 qu'en 1900).
evolution_annuelle = evolution_annuelle.merge(totaux_annuels, on=['annais', 'sexe'], how='left')
evolution_annuelle['part'] = evolution_annuelle['nombre'] / evolution_annuelle['total_naissances_sexe']

evolution_pivot = evolution_annuelle.pivot_table(
    index=['preusuel', 'annais'], 
    columns='sexe', 
    values='part', 
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
prenoms_epicenes_pop = prenoms_epicenes_pop.merge(
    repartition_sexe[['part_masculine']], on='preusuel', how='left'
)
prenoms_epicenes_pop = prenoms_epicenes_pop.sort_values(by='correlation_H_F', ascending=True)

# PREPA VISU
# ----
evolution_annuelle['part_miroir'] = np.where(
    evolution_annuelle['sexe'].isin([1]), 
    evolution_annuelle['part'], 
    -evolution_annuelle['part']
)

evolution_annuelle['sexe'] = evolution_annuelle['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})
prenoms_epicenes_pop = prenoms_epicenes_pop.rename(columns={'nombre': 'nombre_total'})

# ====== COUCHE 1 : données PAR PRÉNOM (filtrées par le clic) ======
df_prenom = evolution_annuelle.merge(
    prenoms_epicenes_pop[['preusuel', 'nombre_total', 'correlation_H_F']], 
    on='preusuel', 
    how='inner'
)
df_prenom['metrique'] = 'Part H/F du prénom'
df_prenom['valeur'] = df_prenom['part_miroir']
df_prenom['valeur_abs'] = df_prenom['valeur'].abs()
df_prenom['titre_affiche'] = df_prenom['preusuel']

# ====== COUCHE 2 : données GLOBALES top 1 (indépendantes du prénom) ======
# Une ligne par (année, sexe). PAS de colonne prénom -> le clic ne peut pas la filtrer.
df_top1 = top1_metrique.copy()
df_top1['sexe'] = df_top1['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})
df_top1['top1_miroir'] = np.where(
    df_top1['sexe'] == 'Homme', df_top1['part_top1'], -df_top1['part_top1']
)
df_top1['metrique'] = 'Part du top 1 (global)'
df_top1['valeur'] = df_top1['top1_miroir']
df_top1['valeur_abs'] = df_top1['valeur'].abs()
df_top1['titre_affiche'] = 'Poids du prénom n°1 (par année)'

# CONSTRUCTION DE LA VISU
# -----------------------

# On identifie le prénom le plus donné (en triant par la popularité)
prenom_top = prenoms_epicenes_pop.sort_values(by='nombre_total', ascending=False).iloc[0]['preusuel']

selection = alt.selection_point(
    fields=['preusuel'], 
    empty=False,
    value=[{'preusuel': prenom_top}] # Sélection par défaut
)

# Dropdown métrique (défini ici car référencé par le scatter ET l'aire)
metrique_dropdown = alt.binding_select(
    options=['Part H/F du prénom', 'Part du top 1 (global)'],
    name='Métrique : '
)
param_metrique = alt.selection_point(
    fields=['metrique'],
    bind=metrique_dropdown,
    value='Part H/F du prénom'
)

# Graphique de gauche (Scatter plot)
scatter_plot = alt.Chart(prenoms_epicenes_pop).mark_circle(size=80).encode(
    x=alt.X('nombre_total:Q', 
            scale=alt.Scale(type='log'), 
            title='Popularité totale (cumul H+F) - Échelle Log'),
    
    y=alt.Y('correlation_H_F:Q', 
            title='Corrélation Homme / Femme (-1 à 1)'),
    
    tooltip=[
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('nombre_total:Q', title='Popularité totale'),
        alt.Tooltip('correlation_H_F:Q', title='Corrélation H/F', format='.2f'),
        alt.Tooltip('part_masculine:Q', title='Part masculine', format='.0%')
    ],
    
    # Gradient continu : corail (féminin) -> gris neutre -> vert (masculin)
    color=alt.Color('part_masculine:Q',
                    scale=alt.Scale(
                        domain=[0, 0.5, 1],
                        range=['#e76f51', '#dddddd', '#2a9d8f']
                    ),
                    title='Part masculine'),
    
    # La sélection est marquée par un contour noir épais (pas par la couleur)
    stroke=alt.condition(selection, alt.value('black'), alt.value(None)),
    strokeWidth=alt.condition(selection, alt.value(2.5), alt.value(0)),
    
    opacity=alt.condition(selection, alt.value(1.0), alt.value(0.8))
).add_params(
    selection,
    param_metrique
).properties(
    width=400,
    height=400
)

# Ligne horizontale à corrélation = 0 (sépare évolution conjointe / basculement)
ligne_zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
    color='gray', strokeDash=[4, 4], opacity=0.7
).encode(y='y:Q')

scatter_plot = ligne_zero + scatter_plot

# Graphique de droite : DEUX couches superposées, le dropdown affiche l'une OU l'autre.
# Le dropdown param_metrique est défini plus haut (avant le scatter).

# Couche 1 : PAR PRÉNOM — filtrée par le clic ET par la métrique
couche_prenom = alt.Chart(df_prenom).mark_area(opacity=0.7).encode(
    x=alt.X('annais:Q', title='Année de naissance', axis=alt.Axis(format='d')), 
    y=alt.Y('valeur:Q', 
            title='Valeur selon la métrique (H en +, F en -)',
            axis=alt.Axis(format='.1%')),
    color=alt.Color('sexe:N', 
                    title='Sexe', 
                    scale=alt.Scale(domain=['Homme', 'Femme'], range=['#2a9d8f', '#e76f51']),
                    legend=alt.Legend(orient='top')),
    detail='preusuel:N',
    tooltip=[
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('annais:Q', title='Année', format='d'),
        alt.Tooltip('sexe:N', title='Sexe'),
        alt.Tooltip('valeur_abs:Q', title='Part du sexe', format='.2%')
    ]
).transform_filter(
    selection
).transform_filter(
    param_metrique
)

# Couche 2 : GLOBALE top 1 — filtrée SEULEMENT par la métrique (jamais par le clic).
# Pas de colonne prénom : le clic ne peut pas la tronquer.
couche_top1 = alt.Chart(df_top1).mark_area(opacity=0.7).encode(
    x=alt.X('annais:Q', axis=alt.Axis(format='d')), 
    y=alt.Y('valeur:Q'),
    color=alt.Color('sexe:N', 
                    scale=alt.Scale(domain=['Homme', 'Femme'], range=['#2a9d8f', '#e76f51']),
                    legend=alt.Legend(orient='top')),
    tooltip=[
        alt.Tooltip('annais:Q', title='Année', format='d'),
        alt.Tooltip('sexe:N', title='Sexe'),
        alt.Tooltip('valeur_abs:Q', title='Part du top 1', format='.2%'),
        alt.Tooltip('prenom_top1:N', title='N°1 de l\'année')
    ]
).transform_filter(
    param_metrique
)

courbes = (couche_prenom + couche_top1).properties(
    width=450,
    height=400
)

# Bande vide au-dessus du scatter, même hauteur que la bande de titre de droite,
# pour que les deux colonnes commencent leur zone de tracé à la même ligne.
bande_vide = alt.Chart(pd.DataFrame({'x': [0]})).mark_text().encode().properties(
    width=400, height=40, title='1. Cliquez sur un prénom'
)
scatter_plot = bande_vide & scatter_plot

# Bande de titre dynamique : deux couches, une par métrique.
# Couche titre PAR PRÉNOM (filtrée par le clic + métrique) : affiche le prénom.
titre_prenom = alt.Chart(df_prenom).mark_text(
    align='center', baseline='middle', fontSize=22, fontWeight='bold',
    color='#333', x=225, y=20
).encode(
    text='titre_affiche:N'
).transform_filter(
    selection
).transform_filter(
    param_metrique
).transform_aggregate(
    groupby=['titre_affiche']
)
# Couche titre GLOBALE (filtrée par métrique seule) : libellé neutre.
titre_top1 = alt.Chart(df_top1).mark_text(
    align='center', baseline='middle', fontSize=22, fontWeight='bold',
    color='#333', x=225, y=20
).encode(
    text='titre_affiche:N'
).transform_filter(
    param_metrique
).transform_aggregate(
    groupby=['titre_affiche']
)
titre_dynamique = (titre_prenom + titre_top1).properties(
    width=450,
    height=40,
    title='2. Évolution temporelle'
)

# Empilement vertical : la bande de titre AU-DESSUS de l'aire
courbes = titre_dynamique & courbes

# Alignement des deux colonnes par le haut (sinon la colonne de droite,
# plus haute à cause de la bande de titre, est centrée et décalée)
tableau_de_bord = alt.hconcat(
    scatter_plot, courbes, spacing=40, center=False
).resolve_scale(
    color='independent'
)
file = 'tableau_de_bord_epicenes.html'
tableau_de_bord.save(file)
chemin_complet = 'file://' + os.path.realpath(file)
webbrowser.open(chemin_complet)