import pandas as pd
import numpy as np
import altair as alt
import os
import webbrowser

alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('json')

# ── PARAMÈTRES DE COULEUR ────────────────────────────────────────────────────
COULEUR_HOMME = '#89c4e1'  # bleu pastel
COULEUR_FEMME = '#ffb3d1'  # rose pastel

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

# ============================================================
# MÉTRIQUE GLOBALE 2 : DURÉE DU RÈGNE DU TOP 1
# Pour chaque année, depuis combien d'années consécutives le n°1
# est-il le même. Toute la séquence affiche la durée TOTALE du règne
# (ex : Jean n°1 de 1900 à 1957 -> ces 58 années affichent 58).
# ============================================================
resultats_duree = []
for sexe_val in [1, 2]:
    g = top1_metrique[top1_metrique['sexe'] == sexe_val].sort_values('annais').copy()
    # un nouveau "bloc" commence à chaque changement de prénom n°1
    bloc = (g['prenom_top1'] != g['prenom_top1'].shift()).cumsum()
    # durée du règne = taille du bloc
    g['duree_top1'] = g.groupby(bloc)['prenom_top1'].transform('size')
    resultats_duree.append(g[['annais', 'sexe', 'duree_top1', 'prenom_top1']])
duree_metrique = pd.concat(resultats_duree)

# ============================================================
# MÉTRIQUE GLOBALE 3 : INDICE DE RÉGIONALITÉ (écart-type entre régions)
# Pour chaque (année, sexe) : on prend le top 100 prénoms de l'année,
# on calcule pour chacun la part qu'il représente dans chaque région,
# puis l'écart-type de ces parts entre régions, et la moyenne sur les prénoms.
# Plus l'indice est élevé, plus les prénoms sont inégalement répartis
# géographiquement (= forte régionalité).
# ============================================================
DPT_REGION = {
    '01':'ARA','03':'ARA','07':'ARA','15':'ARA','26':'ARA','38':'ARA','42':'ARA','43':'ARA','63':'ARA','69':'ARA','73':'ARA','74':'ARA',
    '21':'BFC','25':'BFC','39':'BFC','58':'BFC','70':'BFC','71':'BFC','89':'BFC','90':'BFC',
    '22':'BRE','29':'BRE','35':'BRE','56':'BRE',
    '18':'CVL','28':'CVL','36':'CVL','37':'CVL','41':'CVL','45':'CVL',
    '20':'COR',
    '08':'GES','10':'GES','51':'GES','52':'GES','54':'GES','55':'GES','57':'GES','67':'GES','68':'GES','88':'GES',
    '02':'HDF','59':'HDF','60':'HDF','62':'HDF','80':'HDF',
    '75':'IDF','77':'IDF','78':'IDF','91':'IDF','92':'IDF','93':'IDF','94':'IDF','95':'IDF',
    '14':'NOR','27':'NOR','50':'NOR','61':'NOR','76':'NOR',
    '16':'NAQ','17':'NAQ','19':'NAQ','23':'NAQ','24':'NAQ','33':'NAQ','40':'NAQ','47':'NAQ','64':'NAQ','79':'NAQ','86':'NAQ','87':'NAQ',
    '09':'OCC','11':'OCC','12':'OCC','30':'OCC','31':'OCC','32':'OCC','34':'OCC','46':'OCC','48':'OCC','65':'OCC','66':'OCC','81':'OCC','82':'OCC',
    '44':'PDL','49':'PDL','53':'PDL','72':'PDL','85':'PDL',
    '04':'PAC','05':'PAC','06':'PAC','13':'PAC','83':'PAC','84':'PAC',
    '971':'OM','972':'OM','973':'OM','974':'OM',
}
df_regio = df_vrais.copy()
df_regio['region'] = df_regio['dpt'].map(DPT_REGION)

def _indice_regionalite(sub):
    top = sub.groupby('preusuel')['nombre'].sum().nlargest(100).index
    s = sub[sub['preusuel'].isin(top)]
    total_region = s.groupby('region')['nombre'].sum()
    part = s.groupby(['preusuel', 'region'])['nombre'].sum().unstack(fill_value=0)
    part = part.div(total_region, axis=1)
    return part.std(axis=1).mean()

lignes_regio = []
for (annee, sexe_val), sub in df_regio.groupby(['annais', 'sexe']):
    lignes_regio.append({'annais': annee, 'sexe': sexe_val,
                         'indice_regio': _indice_regionalite(sub)})
regio_metrique = pd.DataFrame(lignes_regio)

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

# ====== COUCHE 3 : durée du règne du top 1 (globale, indépendante du prénom) ======
# Même modèle : une ligne par (année, sexe), pas de colonne prénom filtrable.
df_duree = duree_metrique.copy()
df_duree['sexe'] = df_duree['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})
# en miroir : H vers le haut, F vers le bas (en nombre d'années)
df_duree['duree_miroir'] = np.where(
    df_duree['sexe'] == 'Homme', df_duree['duree_top1'], -df_duree['duree_top1']
)
df_duree['metrique'] = 'Durée du règne du top 1 (global)'
df_duree['valeur'] = df_duree['duree_miroir']
df_duree['valeur_abs'] = df_duree['valeur'].abs()
df_duree['titre_affiche'] = 'Durée du règne du prénom n°1'

# ====== COUCHE 4 : indice de régionalité (global, indépendant du prénom) ======
df_regio_layer = regio_metrique.copy()
df_regio_layer['sexe'] = df_regio_layer['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})
df_regio_layer['regio_miroir'] = np.where(
    df_regio_layer['sexe'] == 'Homme', df_regio_layer['indice_regio'], -df_regio_layer['indice_regio']
)
df_regio_layer['metrique'] = 'Indice de régionalité (global)'
df_regio_layer['valeur'] = df_regio_layer['regio_miroir']
df_regio_layer['valeur_abs'] = df_regio_layer['valeur'].abs()
df_regio_layer['titre_affiche'] = 'Régionalité des prénoms (écart-type entre régions)'

# ============================================================
# RANG N-IÈME : rang annuel de chaque prénom par (année, sexe)
# ============================================================
RANG_MAX = 100

_rang_ann = totaux_prenom_an.copy()
_rang_ann['rang'] = _rang_ann.groupby(['annais', 'sexe'])['nombre'].rank(
    ascending=False, method='first'
).astype(int)
_rang_ann = _rang_ann[_rang_ann['rang'] <= RANG_MAX].copy()
_rang_ann = _rang_ann.merge(totaux_annuels, on=['annais', 'sexe'], how='left')
_rang_ann['part_rang'] = _rang_ann['nombre'] / _rang_ann['total_naissances_sexe']

# Durée consécutive au rang N (même logique que la durée du règne du top 1)
_rang_ann = _rang_ann.sort_values(['sexe', 'rang', 'annais'])
_rang_ann['_bloc'] = _rang_ann.groupby(['sexe', 'rang'])['preusuel'].transform(
    lambda x: (x != x.shift()).cumsum()
)
_rang_ann['duree_rang'] = _rang_ann.groupby(
    ['sexe', 'rang', '_bloc']
)['preusuel'].transform('size')
_rang_ann = _rang_ann.drop(columns=['_bloc'])
_rang_ann['sexe'] = _rang_ann['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})
_rang_ann['part_miroir'] = np.where(
    _rang_ann['sexe'] == 'Homme', _rang_ann['part_rang'], -_rang_ann['part_rang']
)
_rang_ann['duree_miroir'] = np.where(
    _rang_ann['sexe'] == 'Homme', _rang_ann['duree_rang'], -_rang_ann['duree_rang']
)
rang_annuel = _rang_ann.reset_index(drop=True)

# Indice de régionalité spécifique au prénom de rang N (écart-type des parts entre régions)
_prenoms_rang = set(rang_annuel['preusuel'].unique())
_df_rp = df_regio[df_regio['preusuel'].isin(_prenoms_rang)].dropna(subset=['region'])

_total_reg_sexe = _df_rp.groupby(
    ['annais', 'sexe', 'region']
)['nombre'].sum().reset_index().rename(columns={'nombre': 'total_reg_sexe'})

_nb_rp = _df_rp.groupby(
    ['annais', 'sexe', 'preusuel', 'region']
)['nombre'].sum().reset_index()
_nb_rp = _nb_rp.merge(_total_reg_sexe, on=['annais', 'sexe', 'region'], how='left')
_nb_rp['part_region'] = _nb_rp['nombre'] / _nb_rp['total_reg_sexe']

_regio_prenom = _nb_rp.groupby(
    ['annais', 'sexe', 'preusuel']
)['part_region'].std().reset_index().rename(columns={'part_region': 'indice_regio'})
_regio_prenom['sexe'] = _regio_prenom['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})

rang_annuel = rang_annuel.merge(_regio_prenom, on=['annais', 'sexe', 'preusuel'], how='left')
rang_annuel['regio_miroir'] = np.where(
    rang_annuel['sexe'] == 'Homme', rang_annuel['indice_regio'], -rang_annuel['indice_regio']
)

# Classement global cumulatif (toutes années confondues) pour le titre dynamique
rang_global_df = totaux_prenom_an.groupby(['sexe', 'preusuel'])['nombre'].sum().reset_index()
rang_global_df['rang_global'] = rang_global_df.groupby('sexe')['nombre'].rank(
    ascending=False, method='first'
).astype(int)
rang_global_df = rang_global_df[rang_global_df['rang_global'] <= RANG_MAX].copy()
rang_global_df['sexe'] = rang_global_df['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})

# CONSTRUCTION DE LA VISU
# -----------------------

prenom_top = prenoms_epicenes_pop.sort_values(by='nombre_total', ascending=False).iloc[0]['preusuel']

selection = alt.selection_point(
    fields=['preusuel'],
    empty=False,
    value=[{'preusuel': prenom_top}]
)

# ── LIGNE 1, GAUCHE : Scatter plot interactif des prénoms épicènes ──────────
scatter_plot = alt.Chart(prenoms_epicenes_pop).mark_circle(size=80).encode(
    x=alt.X('nombre_total:Q',
            scale=alt.Scale(type='log'),
            title='Popularité totale (cumul H+F) — Échelle log'),
    y=alt.Y('correlation_H_F:Q',
            title='Corrélation Homme / Femme (−1 à 1)'),
    tooltip=[
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('nombre_total:Q', title='Popularité totale'),
        alt.Tooltip('correlation_H_F:Q', title='Corrélation H/F', format='.2f'),
        alt.Tooltip('part_masculine:Q', title='Part masculine', format='.0%')
    ],
    color=alt.Color('part_masculine:Q',
                    scale=alt.Scale(domain=[0, 0.5, 1], range=[COULEUR_FEMME, '#dddddd', COULEUR_HOMME]),
                    title='Part masculine'),
    stroke=alt.condition(selection, alt.value('black'), alt.value(None)),
    strokeWidth=alt.condition(selection, alt.value(2.5), alt.value(0)),
    opacity=alt.condition(selection, alt.value(1.0), alt.value(0.8))
).add_params(
    selection
).properties(
    width=400, height=400,
    title=alt.TitleParams('Cliquez sur un prénom', fontSize=14, color='#777', anchor='start')
)

ligne_zero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
    color='gray', strokeDash=[4, 4], opacity=0.7
).encode(y='y:Q')

scatter_plot = ligne_zero + scatter_plot

# ── LIGNE 1, DROITE : Évolution temporelle (fixe, liée à la sélection) ──────
evolution_fixe = alt.Chart(df_prenom).mark_area(opacity=0.85).encode(
    x=alt.X('annais:Q', title='Année de naissance', axis=alt.Axis(format='d')),
    y=alt.Y('valeur:Q', title='Hommes (+) / Femmes (−)'),
    color=alt.Color('sexe:N', title='Sexe',
                    scale=alt.Scale(domain=['Homme', 'Femme'], range=[COULEUR_HOMME, COULEUR_FEMME]),
                    legend=None),
    detail='preusuel:N',
    tooltip=[
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('annais:Q', title='Année', format='d'),
        alt.Tooltip('sexe:N', title='Sexe'),
        alt.Tooltip('valeur_abs:Q', title='Part du sexe', format='.2%')
    ]
).transform_filter(
    selection
).properties(
    width=450, height=400,
    title=alt.TitleParams('Évolution temporelle', fontSize=14, color='#777', anchor='start')
)

# Légende Homme/Femme partagée pour la ligne 1 (placée sous le scatter)
_df_leg = pd.DataFrame({'Sexe': ['Homme', 'Femme'], 'x': [0.25, 0.75]})
_c1 = alt.Chart(_df_leg).mark_circle(size=160).encode(
    x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1]), axis=None, title=None),
    y=alt.value(20),
    color=alt.Color('Sexe:N',
                    scale=alt.Scale(domain=['Homme', 'Femme'], range=[COULEUR_HOMME, COULEUR_FEMME]),
                    legend=None)
)
_t1 = alt.Chart(_df_leg).mark_text(align='left', baseline='middle', fontSize=13, dx=12).encode(
    x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1]), axis=None, title=None),
    y=alt.value(20),
    text='Sexe:N'
)
legende_hf_1 = (_c1 + _t1).properties(width=400, height=40)

# ── LIGNE 1 : assemblage ─────────────────────────────────────────────────────
partie1 = alt.hconcat(
    alt.vconcat(scatter_plot, legende_hf_1, spacing=4),
    evolution_fixe,
    spacing=40, center=False
).properties(
    title=alt.TitleParams(
        'I - Cas particulier des prénoms épicènes',
        fontSize=20, fontWeight='bold', anchor='start', color='#222'
    )
)

# ── LIGNE 2 : sélecteur de rang ──────────────────────────────────────────────
spinner_rang = alt.binding(input='text', name='Rang n° : ')
param_rang = alt.param(name='rang_n', value='1', bind=spinner_rang)

# Titre dynamique : prénom H et F de rang N dans le classement cumulatif
_df_titre_h = rang_global_df[rang_global_df['sexe'] == 'Homme']
_df_titre_f = rang_global_df[rang_global_df['sexe'] == 'Femme']

_nom_h = alt.Chart(_df_titre_h).mark_text(
    align='center', baseline='middle', fontSize=22, fontWeight='bold', color=COULEUR_HOMME
).encode(text='preusuel:N', x=alt.value(225), y=alt.value(22)
).transform_filter('datum.rang_global === toNumber(rang_n)'
).transform_aggregate(groupby=['preusuel'])

_nom_f = alt.Chart(_df_titre_f).mark_text(
    align='center', baseline='middle', fontSize=22, fontWeight='bold', color=COULEUR_FEMME
).encode(text='preusuel:N', x=alt.value(675), y=alt.value(22)
).transform_filter('datum.rang_global === toNumber(rang_n)'
).transform_aggregate(groupby=['preusuel'])

titre_rang_dynamique = (_nom_h + _nom_f).add_params(param_rang).properties(width=900, height=45)

# Graph 1 : Part du prénom occupant le rang N cette année-là
graph_rang_part = alt.Chart(rang_annuel).mark_area(opacity=0.85).encode(
    x=alt.X('annais:Q', title='Année', axis=alt.Axis(format='d')),
    y=alt.Y('part_miroir:Q', title='Hommes (+) / Femmes (−)'),
    color=alt.Color('sexe:N',
                    scale=alt.Scale(domain=['Homme', 'Femme'], range=[COULEUR_HOMME, COULEUR_FEMME]),
                    legend=None),
    tooltip=[
        alt.Tooltip('annais:Q', title='Année', format='d'),
        alt.Tooltip('sexe:N', title='Sexe'),
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('part_rang:Q', title='Part', format='.2%'),
    ]
).transform_filter(
    'datum.rang === toNumber(rang_n)'
).properties(
    width=280, height=300,
    title='Part du prénom de rang N'
)

# Graph 2 : Durée consécutive au rang N
graph_rang_duree = alt.Chart(rang_annuel).mark_area(opacity=0.85).encode(
    x=alt.X('annais:Q', title='Année', axis=alt.Axis(format='d')),
    y=alt.Y('duree_miroir:Q', title='Hommes (+) / Femmes (−)'),
    color=alt.Color('sexe:N',
                    scale=alt.Scale(domain=['Homme', 'Femme'], range=[COULEUR_HOMME, COULEUR_FEMME]),
                    legend=None),
    tooltip=[
        alt.Tooltip('annais:Q', title='Année', format='d'),
        alt.Tooltip('sexe:N', title='Sexe'),
        alt.Tooltip('preusuel:N', title='Prénom au rang N'),
        alt.Tooltip('duree_rang:Q', title='Durée au rang (ans)', format='d'),
    ]
).transform_filter(
    'datum.rang === toNumber(rang_n)'
).properties(
    width=280, height=300,
    title='Durée consécutive au rang N'
)

# Graph 3 : Indice de régionalité du prénom de rang N
graph_regio = alt.Chart(rang_annuel).mark_area(opacity=0.85).encode(
    x=alt.X('annais:Q', title='Année', axis=alt.Axis(format='d')),
    y=alt.Y('regio_miroir:Q', title='Hommes (+) / Femmes (−)'),
    color=alt.Color('sexe:N',
                    scale=alt.Scale(domain=['Homme', 'Femme'], range=[COULEUR_HOMME, COULEUR_FEMME]),
                    legend=None),
    tooltip=[
        alt.Tooltip('annais:Q', title='Année', format='d'),
        alt.Tooltip('sexe:N', title='Sexe'),
        alt.Tooltip('preusuel:N', title='Prénom'),
        alt.Tooltip('indice_regio:Q', title='Indice de régionalité', format='.4f')
    ]
).transform_filter(
    'datum.rang === toNumber(rang_n)'
).properties(
    width=280, height=300,
    title='Régionalité du prénom de rang N'
)

# ── LIGNE 2 : assemblage ─────────────────────────────────────────────────────
partie2 = alt.vconcat(
    titre_rang_dynamique,
    alt.hconcat(
        graph_rang_part,
        graph_rang_duree,
        graph_regio,
        spacing=30, center=False
    ),
    spacing=8
).properties(
    title=alt.TitleParams(
        'II - Différences des tendances au cours du temps pour les autres prénoms',
        fontSize=20, fontWeight='bold', anchor='start', color='#222'
    )
)

# ── ASSEMBLAGE FINAL avec titre principal ─────────────────────────────────────
tableau_de_bord = alt.vconcat(
    partie1, partie2, spacing=60
).resolve_scale(
    color='independent'
).properties(
    title=alt.TitleParams(
        'Prénoms en France (1900–2020)',
        subtitle='Exploration de l\'impact du sexe dans les différences de tendances',
        fontSize=28, fontWeight='bold', anchor='middle', color='#111',
        subtitleFontSize=14, subtitleColor='#555'
    )
)

file = 'tableau_de_bord_prenoms_sexe.html'
tableau_de_bord.save(file)

chemin_complet = 'file://' + os.path.realpath(file)
webbrowser.open(chemin_complet)