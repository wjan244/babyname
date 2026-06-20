import pandas as pd
import numpy as np
import altair as alt
import os
import webbrowser
import pywt

# config
alt.data_transformers.disable_max_rows()
alt.data_transformers.enable('json')

WIDTH = 600
GOLDEN = (1 + 5 ** 0.5) / 2
WINDOW = 5  # years to average over


# clean data
data = pd.read_csv('data/dpt2020.csv',sep=";")
df = data[(data["annais"] != "XXXX") & (data["dpt"] != "XX")]
df_region = df


# initial df
df_year = (df_region.groupby(["preusuel", "sexe", "annais"])["nombre"].sum().reset_index()) # remove department information
df_year['part_de_naissance'] = df_year['nombre'] / df_year.groupby('annais')['nombre'].transform('sum')
df_year["popularite_annuelle"] = (
    df_year.groupby(["annais", "sexe"])["nombre"]
    .rank(ascending=False, method="min")
    .astype(int)
)


# extra rows df_year
df_name = (
    df_year.groupby(["preusuel", "sexe"])
    .agg(nombre_total_naissance=("nombre", "sum"),
         part_naisssance_moyenne=("part_de_naissance", "mean"),
         part_naisssance_min=("part_de_naissance", "min"),
         part_naisssance_max=("part_de_naissance", "max"))
    .reset_index()
    .sort_values("nombre_total_naissance", ascending=False)
)

df_name["Popularité Cumulée"] = (
    df_name["nombre_total_naissance"]
    .rank(ascending=False, method="min")
    .astype(int)
)

delta_popularite = (
    df_year.sort_values(["preusuel", "sexe", "annais"])
    .groupby(["preusuel","sexe"])["popularite_annuelle"]
    .agg(lambda x: (x.max() - x.min()))
    .rename("delta_popularite")
    .reset_index()
)

df_name = df_name.merge(delta_popularite, on=["preusuel","sexe"], how="left")

top10_count = (
    df_year[df_year["popularite_annuelle"] <= 10]
    .groupby(["preusuel", "sexe"])
    .size()
    .rename("Années top 10")
    .reset_index()
)

df_name = df_name.merge(top10_count, on=["preusuel", "sexe"], how="left")
df_name["Années top 10"] = df_name["Années top 10"].fillna(0).astype(int)

top100_count = (
    df_year[df_year["popularite_annuelle"] <= 100]
    .groupby(["preusuel", "sexe"])
    .size()
    .rename("Années top 100")
    .reset_index()
)

df_name = df_name.merge(top100_count, on=["preusuel", "sexe"], how="left")
df_name["Années top 100"] = df_name["Années top 100"].fillna(0).astype(int)


EPS = 1

df_year = df_year.sort_values(["preusuel", "sexe", "annais"])
df_year["delta_pdm"] = (
    df_year.groupby(["preusuel", "sexe"])["part_de_naissance"]
    .transform(lambda x: x.diff() / (x.shift(1) + EPS))
)

delta_pdm_max = (
    df_year.groupby(["preusuel", "sexe"])["delta_pdm"]
    .apply(lambda x: x.max())
    .rename("delta_pdm_max")
    .reset_index()
)

delta_pdm_min = (
    df_year.groupby(["preusuel", "sexe"])["delta_pdm"]
    .apply(lambda x: x.min())
    .rename("delta_pdm_min")
    .reset_index()
)

df_name = df_name.merge(delta_pdm_max, on=["preusuel", "sexe"], how="left")
df_name = df_name.merge(delta_pdm_min, on=["preusuel", "sexe"], how="left")

# Chapeau mexicain

# 1. Pivoter d'abord (les années manquantes restent des NaN pour l'instant)
df_large = df_year.pivot(
    index=["preusuel", "sexe"], columns="annais", values="part_de_naissance"
)

# 2. Normalisation par ligne (chaque prénom fait sa propre somme à 1)
row_sums = df_large.sum(axis=1)
df_large_normalized = df_large.div(row_sums, axis=0)

# 3. Remplacement des années sans aucune naissance par 0
df_large_normalized = df_large_normalized.fillna(0)

# 4. Extraction de la matrice et CWT
signal_matrix = df_large_normalized.values
scales = np.arange(1, 16)
coefficients, frequencies = pywt.cwt(signal_matrix, scales, "mexh", axis=-1)

# 5. Calcul du score
scores = np.max(np.abs(coefficients), axis=(0, 2))

# Ajout du score au DataFrame df_name
df_name = df_name.merge(
    pd.DataFrame({"preusuel": df_large.index.get_level_values(0), "sexe": df_large.index.get_level_values(1), "cwt_sharpness_score": scores}),
    on=["preusuel", "sexe"],
    how="left"
)

# prenom genré
df_name["prenom_genre"] = df_name["preusuel"] + df_name["sexe"].map(
    {1: "_M", 2: "_F"}
)
df_year["prenom_genre"] = df_year["preusuel"] + df_year["sexe"].map(
    {1: "_M", 2: "_F"}
)

idx_max_pdm = df_year.sort_values(
    by="part_de_naissance", ascending=False
).drop_duplicates(subset=["prenom_genre"])

df_annee_max = idx_max_pdm[["prenom_genre", "annais"]].rename(
    columns={"annais": "annee_max_pdm"}
)

df_name = df_name.merge(df_annee_max, on="prenom_genre", how="left")

# plot


# 2. Extraction des top et bottom prénoms
top_names_genres = (
    df_name.head(500)["prenom_genre"].tolist()
    + df_name.tail(500)["prenom_genre"].tolist()
)
df_year_top = df_year[df_year["prenom_genre"].isin(top_names_genres)]

df_name = df_name.rename(columns={
    "annee_max_pdm":"Année de popularité max",
    "nombre_total_naissance": "Naissances totales",
    "part_naisssance_moyenne": "Part de naissance moyenne",
    "part_naisssance_min": "Part de naissance minimale",
    "part_naisssance_max": "Part de naissance maximale",
    "delta_popularite": "Delta popularité",
    "delta_pdm_max": "Max différentielle relative part de naissance",
    "delta_pdm_min": "Min différentielle relative part de naissance",
    "cwt_sharpness_score": "Indice de popularité éclair",
})
df_year_top = df_year_top.rename(columns={"part_de_naissance": "Part de naissance"})
df_name["Sexe"] = df_name["sexe"].map({1: "M", 2: "F"})
df_year_top["Sexe"] = df_year_top["sexe"].map({1: "M", 2: "F"})

columns = [
    "Année de popularité max",
    "Indice de popularité éclair",
    "Naissances totales",
    "Popularité Cumulée",
    "Part de naissance moyenne",
    "Part de naissance minimale",
    "Part de naissance maximale",
    "Delta popularité",
    "Années top 10",
    "Années top 100",
    "Max différentielle relative part de naissance",
    "Min différentielle relative part de naissance",
]

# Groupe labels for the toggle
top_500 = df_name.head(500).copy(); top_500["groupe"] = "Les 500 plus populaires"
bot_500 = df_name.tail(500).copy(); bot_500["groupe"] = "Les 500 moins populaires"
df_scatter = pd.concat([top_500, bot_500], ignore_index=True)

# Enrich df_year_top with scatter features + groupe so the brush can cross-filter
df_year_top = df_year_top.merge(
    df_scatter[["prenom_genre", "groupe"] + columns], on="prenom_genre", how="left"
)

x_param = alt.param(
    name="x_param",
    value="Année de popularité max",
    bind=alt.binding_select(options=columns, name="X: "),
)
y_param = alt.param(
    name="y_param",
    value="Indice de popularité éclair",
    bind=alt.binding_select(options=columns, name="Y: "),
)
log_x_param = alt.param(
    name="log_x_param",
    value=False,
    bind=alt.binding_checkbox(name="Log scale X: "),
)
log_y_param = alt.param(
    name="log_y_param",
    value=False,
    bind=alt.binding_checkbox(name="Log scale Y: "),
)

groupe_param = alt.param(
    name="groupe_param",
    value="Les 500 plus populaires",
    bind=alt.binding_radio(
        options=["Les 500 plus populaires", "Les 500 moins populaires"],
        name="Groupe : ",
    ),
)

# ── Vues prédéfinies ──────────────────────────────────────────────────────────

columns = [
    "Année de popularité max",
    "Indice de popularité éclair",
    "Naissances totales",
    "Popularité Cumulée",
    "Part de naissance moyenne",
    "Part de naissance minimale",
    "Part de naissance maximale",
    "Delta popularité",
    "Années top 10",
    "Années top 100",
    "Max différentielle relative part de naissance",
    "Min différentielle relative part de naissance",
]


POP = "Les 500 plus populaires"
RARE = "Les 500 moins populaires"

# (x_field, y_field, [prenom_genre preselected], groupe)
PRESETS = {
    "Personnalisé":                                 (None,                                   None,                                  [],                                                                                                                                                                                                                             None),
    "Les plus populaire":                           ("Année de popularité max",               "Part de naissance maximale",          ["MARIE_F","JEAN_M","PHILIPPE_M","SYLVIE_F","NATHALIE_F","STEPHANIE_F","SÉBASTIEN_M","NICOLAS_M","JULIEN_M","KEVIN_M","THOMAS_M","LÉA_F","LUCAS_M","ENZO_M","GABRIEL_M","ADAM_M","LÉO_M","RAPHAËL_M"],                       POP),
    # "Descente progressive":                         ("Max différentielle part de naissance",  "Min différentielle part de naissance", ["HÉLÈNE_F","ODETTE_F","SUZANNE_F","INÈS_F","ZOÉ_F"],                                                                                                                                                                      POP),
    "Prénoms intemporels":                          ("Part de naissance minimale",            "Années top 100",                      ["PAUL_M","PIERRE_M","CHARLES_M"],                                                                                                                                                                                           POP),
    "Pic de popularité bref":                       ("Année de popularité max",               "Indice de popularité éclair",         ["AUGUSTINE_F","GINETTE_F","SIMONNE_F","JEANNINE_F","DANIELLE_F","MARTINE_F","PASCALE_F","VALÉRIE_F","CORINNE_F","SEVERINE_F","JENNIFER_F","PASCALE_F","GEOFFREY_M","DYLAN_M","MATTEO_M","NOA_M","MAELYS_F","TIMEO_M"],       POP),
    "Prénoms soudainement populaires":              ("Indice de popularité éclair",           "Max différentielle relative part de naissance",["ALBERT_M","NATHALIE_F","STEPHANIE_F","EMILIE_F","NICOLAS_M","CHRISTOPHE_M"],                                                                                                                                               POP),
    "Prénoms avec perte popularité temporaire":     ("Année de popularité max",           "Indice de popularité éclair",["ROSE_F","VICTOR_M","SAMUEL_M","HÉLÈNE_F","REMY_F"],  POP),
    "Prénoms impopulaires":                         ("Année de popularité max",           "Popularité Cumulée",[],  RARE),
}

def _preset_expr(axis_idx, fallback):
    expr = fallback
    for label, fields in reversed(list(PRESETS.items())):
        if fields[0] is not None:
            expr = f"preset_param == '{label}' ? datum['{fields[axis_idx]}'] : {expr}"
    return expr

def _preset_names_filter():
    parts = []
    for label, fields in PRESETS.items():
        names = fields[2]
        if names:
            vega_list = "['" + "','".join(names) + "']"
            parts.append(f"(preset_param == '{label}' && indexof({vega_list}, datum.prenom_genre) >= 0)")
        else:
            # empty list = show all names for this preset
            parts.append(f"preset_param == '{label}'")
    return " || ".join(parts)

def _preset_groupe_filter():
    """Vega expression for the effective group: preset overrides the radio when set."""
    expr = "groupe_param"
    for label, fields in reversed(list(PRESETS.items())):
        groupe = fields[3]
        if groupe is not None:
            expr = f"preset_param == '{label}' ? '{groupe}' : {expr}"
    return f"datum.groupe == ({expr})"

x_val_expr = _preset_expr(0, "log_x_param ? log(max(datum[x_param], 1e-10)) / log(10) : datum[x_param]")
y_val_expr = _preset_expr(1, "log_y_param ? log(max(datum[y_param], 1e-10)) / log(10) : datum[y_param]")

preset_param = alt.param(
    name="preset_param",
    value="Personnalisé",
    bind=alt.binding_radio(options=list(PRESETS.keys()), name="Vue : "),
)

brush_interval = alt.selection_interval(encodings=["x", "y"], empty=True)
brush_point = alt.selection_point(fields=["prenom_genre"], toggle=True, empty=True)

# 3. Le Scatter Plot (Maître)
scatter = (
    alt.Chart(df_scatter)
    .mark_point(size=10)
    .encode(
        x=alt.X("x_val:Q", title="", scale=alt.Scale(zero=False),
                axis=alt.Axis(labelExpr="(log_x_param && preset_param == 'Personnalisé') ? format(pow(10, datum.value), '~g') : format(datum.value, '~g')")),
        y=alt.Y("y_val:Q", title="", scale=alt.Scale(zero=False),
                axis=alt.Axis(minExtent=60, labelExpr="(log_y_param && preset_param == 'Personnalisé') ? format(pow(10, datum.value), '~g') : format(datum.value, '~g')")),
        color=alt.Color("prenom_genre:N", legend=None),
        shape=alt.Shape("Sexe:N", scale=alt.Scale(domain=["M", "F"], range=["square", "circle"]),
                        legend=alt.Legend(title="Sexe", symbolSize=60, orient="top-left")),
        opacity=alt.condition(brush_interval & brush_point, alt.value(1.0), alt.value(0.15)),
        tooltip=["preusuel", "Sexe", "Naissances totales"],
    )
    .transform_filter(_preset_groupe_filter())
    .transform_calculate(x_val=x_val_expr, y_val=y_val_expr)
    .add_params(x_param, y_param, log_y_param, log_x_param, brush_interval, brush_point, groupe_param, preset_param)
    .properties(width=WIDTH, height=WIDTH/GOLDEN, title="Prénoms en fonctions de caractéristiques sélectionnables")
)

# 4. Le graphique Temporel (Esclave)
lines = (
    alt.Chart(df_year_top)
    .mark_line()
    .encode(
        x=alt.X(
            "annais:Q",
            title="Année",
            scale=alt.Scale(domain=[1900, 2020], clamp=True),
            axis=alt.Axis(format="d", tickCount=25),
        ),
        y=alt.Y("Part de naissance:Q", title="Part de naissance"),
        detail="prenom_genre:N",
        color=alt.Color("prenom_genre:N", legend=None),
        strokeDash=alt.StrokeDash("Sexe:N", scale=alt.Scale(domain=["F", "M"], range=[[], [4, 2]]),
                                  legend=alt.Legend(title="Sexe", symbolStrokeWidth=2, orient="none", legendX=2*WIDTH, legendY=10)),
        tooltip=["preusuel", "Sexe", "annais", "nombre"],
    )
    .transform_filter(_preset_groupe_filter())
    .transform_filter(_preset_names_filter())
    .transform_filter(brush_interval & brush_point)
    .properties(width=WIDTH, height=WIDTH/GOLDEN, title="Évolution temporelle en part de naissance des prénoms sélectionnés")
)

labels = (
    alt.Chart(df_year_top)
    .mark_text(align="left", dx=4, fontSize=9, clip=False)
    .encode(
        x=alt.X("annais:Q"),
        y=alt.Y("Part de naissance:Q"),
        color=alt.Color("prenom_genre:N", legend=None),
        text="preusuel:N",
    )
    .transform_filter(_preset_groupe_filter())
    .transform_filter(_preset_names_filter())
    .transform_filter(brush_interval & brush_point)
    .transform_joinaggregate(max_year="max(annais)", groupby=["prenom_genre"])
    .transform_filter(alt.datum.annais == alt.datum.max_year)
)

# Affichage côte à côte
dashboard = (
    alt.hconcat(scatter, lines + labels)
    .properties(title=alt.TitleParams(
        "Tendances des prénoms français (1900-2020) \n \n",
        subtitle=[" ",
                  "Ce tableau de bord interactif vous permet de sélectionner des caractéristiques afin de remarquer des tendances dans les prénoms sur le diagramme de dispersion.",
                  "En selectionnant un ou plusieurs prénoms en cliquant sur les points, où une zone, l'évolution temporelle des prénom est visible sur le diagramme de droite ce qui permet de comparer les tendances.",
                  " "," "],
        anchor="middle",
        fontSize=18,
        subtitleFontSize=12,
        subtitleColor="gray",
    ))
    .configure_axisY(minExtent=20)
)


# ----
file = 'viz1.html'
dashboard.save(file)

description = """
<div style="max-width:1300px; margin:30px auto 40px; padding:0 20px;
            font-family:sans-serif; font-size:12px; color:#555; line-height:1.7;">
  <b style="font-size:13px; color:#333;">Glossaire des métriques</b>
  <ul style="margin-top:8px; padding-left:18px;">
    <li><b>Naissances totales</b> : Nombre cumulé de naissances portant ce prénom sur l'ensemble de la période 1900-2020.</li>
    <li><b>Popularité Cumulée</b> : Rang du prénom (1 = le plus donné sur toute la période), calculé séparément par sexe.</li>
    <li><b>Part de naissance moyenne / minimale / maximale</b> : Part de naissance annuelle (moyenne / minimale / maximale sur la période) du prénom.</li>
    <li><b>Année de popularité max</b> : Année où la part de naissance du prénom a atteint son maximum historique.</li>
    <li><b>Delta popularité</b> : Différence entre le rang le plus bas et le plus haut atteints sur toute la période.</li>
    <li><b>Années top 10 / top 100</b> : Nombre d'années où le prénom figurait parmi les 10 (resp. 100) prénoms les plus donnés, au sein de son sexe.</li>
    <li><b>Indice de popularité éclair</b> : Score issu d'une transformée en ondelettes continue (chapeau mexicain) appliquée à la courbe de part de naissance normalisée.
        Il mesure la netteté des pics de popularité : un score élevé signale une montée et decente rapide tandis qu'un score négatif indique une chute temporaire de la popularité.</li>
    <li><b>Max/Min différentielle relative part de naissance</b> : Maximum/Minimum de la différence relative entre la part de naissance d'un prénom d'une année à l'autre.</li>
  </ul>
</div>
"""

with open(file, "r", encoding="utf-8") as f:
    html = f.read()
html = html.replace("</body>", description + "</body>")
with open(file, "w", encoding="utf-8") as f:
    f.write(html)

chemin_complet = 'file://' + os.path.realpath(file)
webbrowser.open(chemin_complet)