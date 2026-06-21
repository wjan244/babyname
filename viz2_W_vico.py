import pandas as pd
import numpy as np
import altair as alt
import json
import os
import urllib.request
import webbrowser

# ── Configuration ──────────────────────────────────────────────────────────────
alt.data_transformers.enable("default")

WIDTH = 600
GOLDEN = (1 + 5 ** 0.5) / 2
PERIOD = 20
N_NAMES = 10

GEO_PATH = "data/regions.geojson"

# ── Chargement ─────────────────────────────────────────────────────────────────
data = pd.read_csv("data/dpt2020.csv", sep=";")

DOM_CODES = {"971", "972", "973", "974", "976"}

# Nettoyage : on retire les années et départements inconnus, et les DOM
df = data[
    (data["annais"] != "XXXX") &
    (data["dpt"] != "XX") &
    (~data["dpt"].isin(DOM_CODES))
].copy()

# ── Nomenclature INSEE 2016 : département (code) -> région administrative ──────
DPT_TO_REGION = {
    # Auvergne-Rhône-Alpes (84)
    "01": "Auvergne-Rhône-Alpes", "03": "Auvergne-Rhône-Alpes", "07": "Auvergne-Rhône-Alpes",
    "15": "Auvergne-Rhône-Alpes", "26": "Auvergne-Rhône-Alpes", "38": "Auvergne-Rhône-Alpes",
    "42": "Auvergne-Rhône-Alpes", "43": "Auvergne-Rhône-Alpes", "63": "Auvergne-Rhône-Alpes",
    "69": "Auvergne-Rhône-Alpes", "73": "Auvergne-Rhône-Alpes", "74": "Auvergne-Rhône-Alpes",
    # Bourgogne-Franche-Comté (27)
    "21": "Bourgogne-Franche-Comté", "25": "Bourgogne-Franche-Comté", "39": "Bourgogne-Franche-Comté",
    "58": "Bourgogne-Franche-Comté", "70": "Bourgogne-Franche-Comté", "71": "Bourgogne-Franche-Comté",
    "89": "Bourgogne-Franche-Comté", "90": "Bourgogne-Franche-Comté",
    # Bretagne (53)
    "22": "Bretagne", "29": "Bretagne", "35": "Bretagne", "56": "Bretagne",
    # Centre-Val de Loire (24)
    "18": "Centre-Val de Loire", "28": "Centre-Val de Loire", "36": "Centre-Val de Loire",
    "37": "Centre-Val de Loire", "41": "Centre-Val de Loire", "45": "Centre-Val de Loire",
    # Corse (94)
    "20": "Corse",
    # Grand Est (44)
    "08": "Grand Est", "10": "Grand Est", "51": "Grand Est", "52": "Grand Est",
    "54": "Grand Est", "55": "Grand Est", "57": "Grand Est", "67": "Grand Est",
    "68": "Grand Est", "88": "Grand Est",
    # Hauts-de-France (32)
    "02": "Hauts-de-France", "59": "Hauts-de-France", "60": "Hauts-de-France",
    "62": "Hauts-de-France", "80": "Hauts-de-France",
    # Île-de-France (11)
    "75": "Île-de-France", "77": "Île-de-France", "78": "Île-de-France", "91": "Île-de-France",
    "92": "Île-de-France", "93": "Île-de-France", "94": "Île-de-France", "95": "Île-de-France",
    # Normandie (28)
    "14": "Normandie", "27": "Normandie", "50": "Normandie", "61": "Normandie", "76": "Normandie",
    # Nouvelle-Aquitaine (75)
    "16": "Nouvelle-Aquitaine", "17": "Nouvelle-Aquitaine", "19": "Nouvelle-Aquitaine",
    "23": "Nouvelle-Aquitaine", "24": "Nouvelle-Aquitaine", "33": "Nouvelle-Aquitaine",
    "40": "Nouvelle-Aquitaine", "47": "Nouvelle-Aquitaine", "64": "Nouvelle-Aquitaine",
    "79": "Nouvelle-Aquitaine", "86": "Nouvelle-Aquitaine", "87": "Nouvelle-Aquitaine",
    # Occitanie (76)
    "09": "Occitanie", "11": "Occitanie", "12": "Occitanie", "30": "Occitanie",
    "31": "Occitanie", "32": "Occitanie", "34": "Occitanie", "46": "Occitanie",
    "48": "Occitanie", "65": "Occitanie", "66": "Occitanie", "81": "Occitanie", "82": "Occitanie",
    # Pays de la Loire (52)
    "44": "Pays de la Loire", "49": "Pays de la Loire", "53": "Pays de la Loire",
    "72": "Pays de la Loire", "85": "Pays de la Loire",
    # Provence-Alpes-Côte d'Azur (93)
    "04": "Provence-Alpes-Côte d'Azur", "05": "Provence-Alpes-Côte d'Azur",
    "06": "Provence-Alpes-Côte d'Azur", "13": "Provence-Alpes-Côte d'Azur",
    "83": "Provence-Alpes-Côte d'Azur", "84": "Provence-Alpes-Côte d'Azur",
    # DOM (conservé dans la table de référence mais filtré en amont)
    "971": "Guadeloupe", "972": "Martinique", "973": "Guyane",
    "974": "La Réunion", "976": "Mayotte",
}

# ── Agrégation par (prénom, période, région) ───────────────────────────────────
# On repart du df nettoyé (sans XXXX / XX / DOM), tous sexes confondus
df_reg = df.copy()
df_reg["region"] = df_reg["dpt"].map(DPT_TO_REGION)
df_reg["annais"] = df_reg["annais"].astype(int)

# Période de 5 ans : 1900-1904 -> "1900", etc. (borne basse comme étiquette)
df_reg["periode"] = (df_reg["annais"] // PERIOD) * PERIOD

# Agrégation tous sexes confondus : nombre par (prénom, période, région)
counts = (
    df_reg.groupby(["preusuel", "periode", "region"])["nombre"]
    .sum()
    .reset_index()
)

# Total des naissances par (période, région) -- _PRENOMS_RARES INCLUS au dénominateur
total_pr = (
    df_reg.groupby(["periode", "region"])["nombre"]
    .sum()
    .rename("total_region_periode")
    .reset_index()
)

counts = counts.merge(total_pr, on=["periode", "region"], how="left")
counts["part"] = counts["nombre"] / counts["total_region_periode"]

# On exclut _PRENOMS_RARES de l'AFFICHAGE (mais il a compté dans le total ci-dessus)
counts = counts[counts["preusuel"] != "_PRENOMS_RARES"]

# ── Sélection des N prénoms les plus représentatifs par période ────────────────
REGIONS = sorted(set(DPT_TO_REGION.values()))

def select_names_for_period(sub):
    """sub : counts restreint à une période. Renvoie la liste des N prénoms retenus."""
    # rang du prénom DANS chaque région (1 = le plus donné), par part décroissante
    sub = sub.copy()
    sub["rang_region"] = (
        sub.groupby("region")["part"].rank(ascending=False, method="first")
    )
    # popularité nationale du prénom sur la période (pour départager à la troncature)
    pop_nat = sub.groupby("preusuel")["nombre"].sum()

    retenus = []          # ordre d'ajout
    rang = 1
    n_regions = sub["region"].nunique()
    while len(set(retenus)) < N_NAMES and rang <= n_regions * 5:  # garde-fou
        # prénoms qui sont au rang `rang` dans au moins une région
        candidats = sub.loc[sub["rang_region"] == rang, "preusuel"].unique().tolist()
        # on ne garde que les nouveaux
        nouveaux = [n for n in candidats if n not in set(retenus)]
        # on les trie par popularité nationale décroissante pour que la troncature
        # éventuelle retire bien les moins populaires de ce rang
        nouveaux = sorted(nouveaux, key=lambda n: pop_nat.get(n, 0), reverse=True)
        retenus.extend(nouveaux)
        rang += 1

    return retenus[:N_NAMES]

# Application période par période
selected = {}
for periode, sub in counts.groupby("periode"):
    selected[periode] = select_names_for_period(sub)

# ── Calcul du TOP NATIONAL par période (prénom #1 national) ─────────────────────
top_national_by_periode = {}
for periode, sub in counts.groupby("periode"):
    top_national = sub.groupby("preusuel")["nombre"].sum().idxmax()
    top_national_by_periode[periode] = top_national

# ── Construction de prof : long format avec parts normalisées et centrées ───────
# Long format des couples (période, prénom retenu)
paires = pd.DataFrame(
    [(p, n) for p, noms in selected.items() for n in noms],
    columns=["periode", "preusuel"]
)

# On récupère les parts, en RÉINTRODUISANT les régions à 0
# (produit cartésien période×prénom retenu × toutes les régions présentes cette période-là)
regions_par_periode = counts[["periode", "region"]].drop_duplicates()
grille = paires.merge(regions_par_periode, on="periode", how="left")

prof = grille.merge(
    counts[["periode", "preusuel", "region", "part"]],
    on=["periode", "preusuel", "region"], how="left"
)
prof["part"] = prof["part"].fillna(0.0)   # région sans ce prénom -> part 0
prof["part_norm"] = prof["part"].fillna(0.0) / prof.groupby(["periode", "preusuel"])["part"].transform("sum")   # région sans ce prénom -> part 0

# Centrage : moyenne NON pondérée des parts régionales, par (période, prénom)
prof["part_moy"] = prof.groupby(["periode", "preusuel"])["part_norm"].transform("mean")
prof["part_std"] = prof.groupby(["periode", "preusuel"])["part_norm"].transform("std")
prof["part_centree"] = prof["part_norm"] - prof["part_moy"]
# prof["part_centree_abs"] = abs(prof["part"] - prof["part_moy"])


# Application de la racine carrée signée pour étirer les petites variations et tasser la Corse
prof["part_z"] = np.sign(prof["part_centree"]) * np.sqrt(prof["part_centree"].abs())
# prof["part_z"] = prof["part_centree"] / prof.groupby(["periode", "preusuel"])["part_centree_abs"].transform("sum")
# prof["part_z_nostd"] = prof["part_centree"] #/ prof["part_std"]

# garde-fou : si part_std == 0 (prénom à part identique partout, ou présent dans 1 seule région
# après réintroduction des zéros -> std non nul en fait), on met z = 0
prof["part_z"] = prof["part_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

prof["signe"] = np.where(prof["part_z"] >= 0, "+", "-")

# Ajouter l'indicateur "top national"
prof["is_top_national"] = prof.apply(
    lambda row: row["preusuel"] == top_national_by_periode.get(row["periode"]),
    axis=1
)

# ── Ordre des régions et des prénoms dans chaque case ─────────────────────────

# --- Ordre des régions DANS chaque case : part décroissante (rang propre au couple)
prof["rang_region_case"] = (
    prof.groupby(["periode", "preusuel"])["part"]
    .rank(ascending=False, method="first")
    .astype(int)
)

# --- Ordre des prénoms DANS chaque ligne : par disparité inter-régions décroissante.
# Mesure de disparité : entropie de Shannon des parts (normalisées en distribution).
# Faible entropie = concentré sur peu de régions = forte disparité -> à GAUCHE.
def entropie(parts):
    p = np.asarray(parts, dtype=float)
    s = p.sum()
    if s <= 0:
        return np.nan
    p = p[p > 0] / s
    return -(p * np.log(p)).sum()

disp = (
    prof.groupby(["periode", "preusuel"])["part"]
    .apply(lambda s: entropie(s.values))
    .rename("entropie")
    .reset_index()
)
# rang du prénom dans la ligne : entropie croissante -> position 1 (gauche) = plus disparate
disp["rang_prenom"] = (
    disp.groupby("periode")["entropie"].rank(ascending=True, method="first").astype(int)
)

prof = prof.merge(disp, on=["periode", "preusuel"], how="left")

# ── GeoJSON des régions françaises ────────────────────────────────────────────
URL = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"

if not os.path.exists(GEO_PATH):
    urllib.request.urlretrieve(URL, GEO_PATH)
    print("Téléchargé :", GEO_PATH)
else:
    print("Déjà présent :", GEO_PATH)

with open(GEO_PATH, encoding="utf-8") as f:
    geojson = json.load(f)

# ── Données carte : counts filtré aux seuls prénoms retenus ───────────────────
# On ne passe au frontend que les lignes utiles (≈ N_NAMES × n_periodes × n_regions)
selected_pairs = prof[["preusuel", "periode"]].drop_duplicates()
counts_map = counts.merge(selected_pairs, on=["preusuel", "periode"])
counts_map["periode"] = counts_map["periode"].astype(int)
counts_map["preusuel"] = counts_map["preusuel"].astype(str)

# Compléter avec les régions manquantes (mettre part=1 pour celles sans naissance)
regions_list = list(set(counts["region"]))  # toutes les régions du dataset
preusuel_periode_pairs = counts_map[["preusuel", "periode"]].drop_duplicates()

# Créer toutes les combinaisons manquantes
missing_rows = []
for _, row in preusuel_periode_pairs.iterrows():
    preusuel = row["preusuel"]
    periode = row["periode"]
    existing_regions = set(counts_map[(counts_map["preusuel"] == preusuel) & 
                                       (counts_map["periode"] == periode)]["region"])
    for region in regions_list:
        if region not in existing_regions:
            missing_rows.append({
                "preusuel": preusuel,
                "periode": periode,
                "region": region,
                "part": 1  # valeur par défaut pour régions sans naissance
            })

if missing_rows:
    missing_df = pd.DataFrame(missing_rows)
    counts_map = pd.concat([counts_map, missing_df], ignore_index=True)

# Calculer le total national de naissances par prénom par période
# (somme sur toutes les régions)
total_national_by_prenom_periode = counts.groupby(["preusuel", "periode"])["nombre"].sum().reset_index()
total_national_by_prenom_periode = total_national_by_prenom_periode.rename(columns={"nombre": "total_national"})

# Merger avec counts_map pour avoir le total national
counts_map = counts_map.merge(total_national_by_prenom_periode, on=["preusuel", "periode"], how="left")

# Calculer le max du total national par période
max_national_by_periode = total_national_by_prenom_periode.groupby("periode")["total_national"].max().to_dict()
counts_map["max_national_periode"] = counts_map["periode"].map(max_national_by_periode)

# Normaliser par le max national de la période
# Pour les régions manquantes (part=1), on utilise total_national=1 (valeur par défaut)
counts_map["total_national"] = counts_map["total_national"].fillna(1)
counts_map["part_normalized"] = counts_map["total_national"] / counts_map["max_national_periode"]

# ── Sélection interactive : menus déroulants pour choisir le prénom et la période ─
# Valeur par défaut : prénom le plus populaire de la période la plus récente
default_periode = int(max(selected.keys()))
default_prenom = selected[default_periode][0]

all_prenoms = sorted(prof["preusuel"].unique().tolist())
all_periodes = sorted(prof["periode"].unique().tolist())

prenom_param = alt.param(
    name="selected_prenom",
    value=default_prenom,
    bind=alt.binding_select(options=all_prenoms, name="Prénom : "),
)
periode_param = alt.param(
    name="selected_periode",
    value=default_periode,
    bind=alt.binding_select(options=all_periodes, name="Période : "),
)

# ── Facette (panneau gauche) ───────────────────────────────────────────────────
yabs = prof["part_z"].abs().max()

bars = alt.Chart(prof).mark_bar().encode(
    x=alt.X("rang_region_case:O", title=None, axis=None),
    y=alt.Y("part_z:Q",
            axis=alt.Axis(labels=False, ticks=False, domain=False),
            title=""),#"Écart à la moyenne (z-score)",
            # scale=alt.Scale(domain=[-yabs, yabs])),
            color=alt.Color("signe:N",
                    scale=alt.Scale(domain=["+", "-"], range=["#d8642f", "#3a7ca5"]),
                    legend=None),
    tooltip=[alt.Tooltip("periode:O", title="Période"),
             alt.Tooltip("preusuel:N", title="Prénom"),
             alt.Tooltip("region:N", title="Région"),
             alt.Tooltip("part:Q", format=".2%", title="Part des naissances")],
)

# petit label du prénom, un seul par case (on prend la 1re ligne du groupe)
labels = alt.Chart(prof).mark_text(
    baseline="top", fontSize=8, dy=2
).encode(
    x=alt.value(40),          # centré horizontalement dans une case de largeur 80
    y=alt.value(0),           # tout en haut de la case
    text=alt.Text("preusuel:N"),
)

case = alt.layer(bars, labels).properties(width=80, height=70)

facet_chart = case.facet(
    row=alt.Row("periode:O", title="Disparité par rapport à la moyenne nationale (par période)", sort="descending"),
    column=alt.Column("rang_prenom:O",
                      title="Prénoms (gauche = plus disparate)",
                      sort="ascending",
                      header=alt.Header(labelOrient="bottom")),
).resolve_scale(x="independent")

# ── Carte (panneau droit) ──────────────────────────────────────────────────────
# Approche « GeoJSON enrichi » : on intègre les données statistiques directement
# dans les properties de chaque feature GeoJSON.  mark_geoshape fonctionne de façon
# fiable uniquement avec des features GeoJSON valides (type + geometry + properties).
# Le transform_filter filtre ensuite sur datum.properties.* côté Vega-Lite.

geo_by_nom = {f["properties"]["nom"]: f["geometry"] for f in geojson["features"]}

# Un feature par (preusuel, periode, region) — la géométrie est répétée mais c'est léger
features_enrichis = [
    {
        "type": "Feature",
        "geometry": geo_by_nom[row["region"]],
        "properties": {
            "nom": row["region"],
            "preusuel": str(row["preusuel"]),
            "periode": int(row["periode"]),
            "part": float(row["part"]),
            "part_normalized": float(row["part_normalized"]),
            "is_top_national": row["preusuel"] == top_national_by_periode.get(int(row["periode"])),
        },
    }
    for _, row in counts_map.iterrows()
    if row["region"] in geo_by_nom
]

map_chart = (
    alt.Chart(alt.InlineData(
        values={"type": "FeatureCollection", "features": features_enrichis},
        format=alt.DataFormat(property="features"),
    ))
    .mark_geoshape(stroke="white", strokeWidth=0.5)
    .transform_filter(
        "datum.properties.preusuel == selected_prenom && datum.properties.periode == selected_periode"
    )
    .encode(
        color=alt.Color("properties.part_normalized:Q", scale=alt.Scale(scheme="purples", domain=[0, 1]), title="Popularité nationale"),
        tooltip=[alt.Tooltip("properties.nom:N", title="Région"),
                 alt.Tooltip("properties.part:Q", format=".2%", title="Part des naissances")],
    )
    .project(type="mercator")
    .properties(width=450, height=450, title="Part par région")
    .add_params(prenom_param, periode_param)
)

# ── Dashboard ─────────────────────────────────────────────────────────────────

dashboard = (facet_chart | map_chart).properties(
    title=alt.TitleParams(
        "Disparité géographique des prénoms français (1900-2021) \n \n",
        subtitle=[" ",
                  "Visualisation de l'évolution des prénoms français par région et période.",
                  "L'indice de disparité mesure la concentration régionale : plus la couleur est foncée, plus le prénom est concentré dans certaines régions.",
                  " "," "],
        anchor="middle",
        fontSize=18,
        subtitleFontSize=12,
        subtitleColor="gray",
    )
)

# ── Sauvegarde et ouverture ────────────────────────────────────────────────────
file = "viz2.html"
dashboard.save(file)
chemin_complet = "file://" + os.path.realpath(file)
webbrowser.open(chemin_complet)