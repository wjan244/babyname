import pandas as pd
import numpy as np
import altair as alt
import os
import webbrowser

# enable('json') réinstalle le transformer, donc on désactive la limite APRÈS.
alt.data_transformers.enable('json')
alt.data_transformers.disable_max_rows()

# ── PARAMÈTRES DE COULEUR ────────────────────────────────────────────────────
COULEUR_HOMME = "#1e90c9"  # bleu pastel
COULEUR_FEMME = "#ff6ba6"  # rose pastel


# ── HELPERS GRAPHIQUES PARTAGÉS ──────────────────────────────────────────────
# Tous les graphes utilisent la même échelle de couleur, le même axe des années
# et le même axe vertical en miroir H/F. On les fabrique ici une seule fois.

def echelle_sexe():
    return alt.Scale(domain=['Homme', 'Femme'], range=[COULEUR_HOMME, COULEUR_FEMME])

def couleur_sexe():
    return alt.Color('sexe:N', scale=echelle_sexe(), legend=None)

def axe_annee(titre='Année'):
    return alt.X('annais:Q', title=titre, axis=alt.Axis(format='d'))

def axe_miroir():
    return alt.Y('valeur:Q', title='Hommes (+) / Femmes (−)')


# ── PRÉPARATION D'UNE COUCHE EN MIROIR ───────────────────────────────────────
# Toutes les couches du tableau de bord suivent le même patron :
#   sexe numérique -> 'Homme'/'Femme', puis valeur signée (H vers le haut,
#   F vers le bas) et sa valeur absolue pour les tooltips.
def prepare_couche(df, col_metrique):
    df = df.copy()
    df['sexe'] = df['sexe'].astype(str).replace({'1': 'Homme', '2': 'Femme'})
    df['valeur'] = np.where(df['sexe'] == 'Homme', df[col_metrique], -df[col_metrique])
    df['valeur_abs'] = df['valeur'].abs()
    return df


# ── CHARGEMENT & NETTOYAGE ───────────────────────────────────────────────────
data = pd.read_csv('data/dpt2020.csv', sep=";")
df = data[(data["annais"] != "XXXX") & (data["dpt"] != "XX")]

# Vrais prénoms uniquement (hors agrégats INSEE commençant par '_').
df_vrais = df[~df['preusuel'].str.startswith('_')]

# TOTAUX ANNUELS NATIONAUX PAR SEXE (le dénominateur)
# -------------------------
# Calculé sur TOUT le dataset (tous prénoms), sinon le dénominateur serait faux.
# Donne, pour chaque (année, sexe), le nombre total de naissances en France.
totaux_annuels = (
    df.groupby(['annais', 'sexe'])['nombre'].sum()
    .reset_index()
    .rename(columns={'nombre': 'total_naissances_sexe'})
)

# ============================================================
# MÉTRIQUE GLOBALE 1 : PART DU TOP 1 (par année et par sexe)
# Pour chaque (année, sexe), quel % des naissances représente
# le prénom le plus donné cette année-là.
# ============================================================
totaux_prenom_an = df_vrais.groupby(['annais', 'sexe', 'preusuel'])['nombre'].sum().reset_index()
# Pour chaque (année, sexe), on garde la ligne du prénom max.
idx_max = totaux_prenom_an.groupby(['annais', 'sexe'])['nombre'].idxmax()
top1 = totaux_prenom_an.loc[idx_max].merge(totaux_annuels, on=['annais', 'sexe'], how='left')
top1['part_top1'] = top1['nombre'] / top1['total_naissances_sexe']
top1 = top1.rename(columns={'preusuel': 'prenom_top1'})
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
    # Un nouveau "bloc" commence à chaque changement de prénom n°1.
    bloc = (g['prenom_top1'] != g['prenom_top1'].shift()).cumsum()
    # Durée du règne = taille du bloc.
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

lignes_regio = [
    {'annais': annee, 'sexe': sexe_val, 'indice_regio': _indice_regionalite(sub)}
    for (annee, sexe_val), sub in df_regio.groupby(['annais', 'sexe'])
]
regio_metrique = pd.DataFrame(lignes_regio)


# ── ANALYSE DES PRÉNOMS ÉPICÈNES ─────────────────────────────────────────────

# Répartition H/F cumulée par prénom (colonnes 1 = masculin, 2 = féminin).
repartition_sexe = df.groupby(['preusuel', 'sexe'])['nombre'].sum().unstack(fill_value=0)
repartition_sexe['total'] = repartition_sexe.sum(axis=1)
repartition_sexe['part_minoritaire'] = repartition_sexe[[1, 2]].min(axis=1) / repartition_sexe['total']
# Part masculine globale : 0 = entièrement féminin, 1 = entièrement masculin.
repartition_sexe['part_masculine'] = repartition_sexe[1] / repartition_sexe['total']

# Règle des 10%, pour filtrer le bruit et les erreurs d'annotation (Marie en H).
vrais_epicenes = repartition_sexe[
    (repartition_sexe['part_minoritaire'] >= 0.10) & (repartition_sexe['total'] >= 500)
].index

# On exclut aussi tout préfixe '_' (agrégats INSEE des prénoms rares) : ils restent
# comptés dans totaux_annuels (le dénominateur) mais ne sont pas de vrais prénoms.
df_filtre = df[df['preusuel'].isin(vrais_epicenes) & (~df['preusuel'].str.startswith('_'))]

evolution_annuelle = df_filtre.groupby(['preusuel', 'sexe', 'annais'])['nombre'].sum().reset_index()

# Passage en PARTS : on divise par le total national du sexe pour cette année.
# Neutralise la démographie (plus de naissances en 2000 qu'en 1900).
evolution_annuelle = evolution_annuelle.merge(totaux_annuels, on=['annais', 'sexe'], how='left')
evolution_annuelle['part'] = evolution_annuelle['nombre'] / evolution_annuelle['total_naissances_sexe']

evolution_pivot = evolution_annuelle.pivot_table(
    index=['preusuel', 'annais'], columns='sexe', values='part', fill_value=0
)

# Filtre de robustesse : sous 10 années de données, la corrélation n'a pas de sens.
comptage_annees = evolution_annuelle.groupby('preusuel')['annais'].nunique()
prenoms_solides = comptage_annees[comptage_annees >= 10].index
evolution_pivot_filtre = evolution_pivot.loc[prenoms_solides]

correlations = evolution_pivot_filtre.groupby('preusuel').apply(
    lambda x: x.iloc[:, 0].corr(x.iloc[:, 1], min_periods=5)
).reset_index(name='correlation_H_F')

# Table de synthèse par prénom : popularité, corrélation H/F, part masculine.
prenoms_epicenes_pop = (
    df_filtre.groupby('preusuel')['nombre'].sum().reset_index()
    .merge(correlations, on='preusuel', how='left')
    .merge(repartition_sexe[['part_masculine']], on='preusuel', how='left')
    .rename(columns={'nombre': 'nombre_total'})
    .sort_values(by='correlation_H_F', ascending=True)
)


# ── PRÉPARATION DES COUCHES DE LA VISU ───────────────────────────────────────
# Couche 1 : par prénom (filtrée par le clic). Les colonnes nombre_total /
# correlation_H_F ne sont pas affichées dans ce graphe, donc inutile de les joindre.
df_prenom = prepare_couche(evolution_annuelle, 'part')

# Couches 2 à 4 : métriques globales (une ligne par année/sexe, pas de prénom filtrable).
df_top1 = prepare_couche(top1_metrique, 'part_top1')
df_duree = prepare_couche(duree_metrique, 'duree_top1')
df_regio_layer = prepare_couche(regio_metrique, 'indice_regio')


# ── CONSTRUCTION DE LA VISU ───────────────────────────────────────────────────

prenom_top = prenoms_epicenes_pop.sort_values(by='nombre_total', ascending=False).iloc[0]['preusuel']

selection = alt.selection_point(
    fields=['preusuel'],
    empty=False,
    value=[{'preusuel': prenom_top}]
)

# ── LIGNE 1, GAUCHE : Scatter plot interactif des prénoms épicènes ───────────
scatter_plot = alt.Chart(prenoms_epicenes_pop).mark_circle(size=80).encode(
    x=alt.X('nombre_total:Q',
            scale=alt.Scale(type='log'),
            title='Popularité totale (cumul H+F) — Échelle log'),
    y=alt.Y('correlation_H_F:Q',
            title='Corrélation H/F (−1 = "vases communicants" ; 1 = "tendances similaires")'),
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

# ── LIGNE 1, DROITE : Évolution temporelle (fixe, liée à la sélection) ───────
evolution_fixe = alt.Chart(df_prenom).mark_area(opacity=0.85).encode(
    x=axe_annee('Année de naissance'),
    y=axe_miroir(),
    color=couleur_sexe(),
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

# Légende Homme/Femme partagée pour la ligne 1 (placée sous le scatter).
_df_leg = pd.DataFrame({'Sexe': ['Homme', 'Femme'], 'x': [0.25, 0.75]})
_c1 = alt.Chart(_df_leg).mark_circle(size=160).encode(
    x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1]), axis=None, title=None),
    y=alt.value(20),
    color=alt.Color('Sexe:N', scale=echelle_sexe(), legend=None)
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
        'Cas des prénoms épicènes',
        fontSize=20, fontWeight='bold', anchor='start', color='#222'
    )
)

# ── LIGNE 2 : trois graphes globaux statiques ────────────────────────────────
# Même structure pour les trois : aire en miroir H/F, seuls le titre et le
# tooltip de la valeur changent (et l'affichage ou non du prénom n°1).
def graphe_global(data, titre, abs_titre, abs_format, avec_prenom=False):
    tooltip = [
        alt.Tooltip('annais:Q', title='Année', format='d'),
        alt.Tooltip('sexe:N', title='Sexe'),
        alt.Tooltip('valeur_abs:Q', title=abs_titre, format=abs_format),
    ]
    if avec_prenom:
        tooltip.append(alt.Tooltip('prenom_top1:N', title="N°1 de l'année"))
    return alt.Chart(data).mark_area(opacity=0.85).encode(
        x=axe_annee('Année'),
        y=axe_miroir(),
        color=couleur_sexe(),
        tooltip=tooltip
    ).properties(width=280, height=300, title=titre)

graph_top1 = graphe_global(
    df_top1, 'Poids du prénom n°1', 'Part du top 1', '.2%', avec_prenom=True
)
graph_duree = graphe_global(
    df_duree, 'Durée du règne du prénom n°1', 'Durée du règne (ans)', 'd', avec_prenom=True
)
graph_regio = graphe_global(
    df_regio_layer, 'Régionalité des prénoms', 'Indice de régionalité', '.4f'
)

# ── LIGNE 2 : assemblage ─────────────────────────────────────────────────────
partie2 = alt.hconcat(
    graph_top1, graph_duree, graph_regio, spacing=30, center=False
).properties(
    title=alt.TitleParams(
        "Différences des tendances au cours du temps pour l'ensemble des prénoms",
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
        subtitle="Exploration de l'impact du sexe dans les différences de tendances",
        fontSize=28, fontWeight='bold', anchor='middle', color='#111',
        subtitleFontSize=14, subtitleColor='#555'
    )
)

file = 'tableau_de_bord_prenoms_sexe.html'
tableau_de_bord.save(file)

chemin_complet = 'file://' + os.path.realpath(file)
webbrowser.open(chemin_complet)