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


# 2. Extraction des top prénoms mis à jour
top_names_genres = df_name.head(500)["prenom_genre"].tolist()
df_year_top = df_year[df_year["prenom_genre"].isin(top_names_genres)]

df_name = df_name.rename(columns={
    "annee_max_pdm":"Année de popularité max",
    "nombre_total_naissance": "Naissances totales",
    "part_naisssance_moyenne": "Part de naissance moyenne",
    "part_naisssance_min": "Part de naissance minimale",
    "part_naisssance_max": "Part de naissance maximale",
    "delta_popularite": "Delta popularité",
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
]

# Enrich df_year_top with scatter features so the interval brush can cross-filter
df_year_top = df_year_top.merge(
    df_name[["prenom_genre"] + columns], on="prenom_genre", how="left"
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
log_y_param = alt.param(
    name="log_y_param",
    value=False,
    bind=alt.binding_checkbox(name="Log scale Y: "),
)
log_x_param = alt.param(
    name="log_x_param",
    value=False,
    bind=alt.binding_checkbox(name="Log scale X: "),
)

brush_interval = alt.selection_interval(encodings=["x", "y"], empty=True)
brush_point = alt.selection_point(fields=["prenom_genre"], toggle=True, empty=True)

# 3. Le Scatter Plot (Maître)
scatter = (
    alt.Chart(df_name.head(500))
    .mark_point(size=10)
    .encode(
        x=alt.X("x_val:Q", title="", scale=alt.Scale(zero=False)),
        y=alt.Y("y_val:Q", title="", scale=alt.Scale(zero=False)),
        color=alt.Color("prenom_genre:N", legend=None),
        shape=alt.Shape("Sexe:N", scale=alt.Scale(domain=["M", "F"], range=["square", "circle"]), legend=None),
        opacity=alt.condition(brush_interval & brush_point, alt.value(1.0), alt.value(0.15)),
        tooltip=["preusuel", "Sexe", "Naissances totales"],
    )
    .transform_calculate(x_val="log_x_param ? log(datum[x_param]) : datum[x_param]", y_val="log_y_param ? log(datum[y_param]) : datum[y_param]")
    .add_params(x_param, y_param, log_y_param, log_x_param, brush_interval, brush_point)
    .properties(width=WIDTH, height=WIDTH/GOLDEN)
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
    .transform_calculate(x_val="log_x_param ? log(datum[x_param]) : datum[x_param]", y_val="log_y_param ? log(datum[y_param]) : datum[y_param]")
    .transform_filter(brush_interval & brush_point)
    .properties(width=WIDTH, height=WIDTH/GOLDEN)
)

labels = (
    alt.Chart(df_year_top)
    .mark_text(align="left", dx=4, fontSize=9)
    .encode(
        x=alt.X("annais:Q"),
        y=alt.Y("Part de naissance:Q"),
        color=alt.Color("prenom_genre:N", legend=None),
        text="preusuel:N",
    )
    .transform_calculate(x_val="log_x_param ? log(datum[x_param]) : datum[x_param]", y_val="log_y_param ? log(datum[y_param]) : datum[y_param]")
    .transform_filter(brush_interval & brush_point)
    .transform_joinaggregate(max_year="max(annais)", groupby=["prenom_genre"])
    .transform_filter(alt.datum.annais == alt.datum.max_year)
)

# Affichage côte à côte
dashboard = (scatter | (lines + labels)).configure_axisY(minExtent=20)


# ----
file = 'viz1.html'
dashboard.save(file)
chemin_complet = 'file://' + os.path.realpath(file)
webbrowser.open(chemin_complet)