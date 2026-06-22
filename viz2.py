import pandas as pd
import numpy as np
import altair as alt
import json
import os
import urllib.request
import webbrowser

# ── Configuration ──────────────────────────────────────────────────────────────
alt.data_transformers.enable("default")

PERIOD = 20

# Éligibilité grille de droite : un prénom doit figurer dans le top-TOP_REGION
# (par effectif) d'au moins une région pour être candidat. Plus petit = plus strict.
TOP_REGION = 50

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

# ── Sélection des prénoms : 5 populaires + 5 régionaux (hors populaires) ─────────
N_POP = 5         # nombre de prénoms populaires (grille de gauche)
N_REG = 5         # nombre de prénoms régionalement marqués (grille de droite)


def shannon_entropy(parts):
    """Entropie de Shannon des parts régionales d'un prénom.
    Faible = concentré sur quelques régions ; élevée = réparti uniformément."""
    p = np.asarray(parts, dtype=float)
    s = p.sum()
    if s <= 0:
        return np.nan
    p = p[p > 0] / s
    return -(p * np.log(p)).sum()


def select_national(sub, k):
    """Les k prénoms les plus donnés au niveau NATIONAL sur la période."""
    pop_nat = sub.groupby("preusuel")["nombre"].sum()
    return pop_nat.nlargest(k).index.tolist()


def select_regional(sub, k, exclure):
    """Sélection en k tours, sur un ensemble de régions qui rétrécit.

    À chaque tour :
      1. on calcule l'entropie de Shannon de chaque prénom éligible sur les
         régions ENCORE ACTIVES (parts renormalisées sur ces régions) ;
      2. on retient le prénom à plus faible entropie (= le plus concentré) ;
      3. on repère SA région de plus forte part et on l'écarte globalement
         pour tous les tours suivants ;
      4. on retire ce prénom du pool, AINSI QUE tout prénom dont l'éligibilité
         (top-TOP_REGION) ne tenait qu'à la région qu'on vient d'écarter —
         c'est-à-dire les prénoms présents dans le top de cette seule région et
         d'aucune autre. Ces prénoms ont une faible entropie pour la seule raison
         qu'ils sont quasi absents ailleurs (artefact de rareté), pas parce
         qu'ils sont caractéristiques d'une autre région.

    Effet : une région structurellement « capturante » (ex. Corse) tombe au
    premier tour, et les prénoms qui ne doivent leur concentration qu'à elle
    disparaissent avec elle.

    Éligibilité : figurer dans le top-TOP_REGION (par effectif) d'au moins une
    région, hors prénoms déjà retenus comme populaires.
    """
    sub = sub.copy()
    exclure = set(exclure)

    # Éligibilité : top-TOP_REGION par effectif dans au moins une région
    sub["rang_eff_region"] = (
        sub.groupby("region")["nombre"].rank(ascending=False, method="first")
    )
    est_top = sub["rang_eff_region"] <= TOP_REGION

    # Pour chaque prénom : ensemble des régions où il est dans le top-TOP_REGION
    top_regions_par_prenom = (
        sub.loc[est_top]
        .groupby("preusuel")["region"]
        .agg(set)
        .to_dict()
    )

    eligibles = set(sub.loc[est_top, "preusuel"].unique()) - exclure

    regions_actives = set(sub["region"].unique())
    retenus = []

    for _ in range(k):
        if not eligibles or not regions_actives:
            break

        sub_act = sub[
            sub["preusuel"].isin(eligibles) & sub["region"].isin(regions_actives)
        ]
        if sub_act.empty:
            break

        # Entropie de chaque prénom éligible sur les régions actives
        entropie_par_prenom = (
            sub_act.groupby("preusuel")["part"]
            .apply(lambda s: shannon_entropy(s.values))
            .dropna()
        )
        if entropie_par_prenom.empty:
            break

        # Prénom le plus concentré sur les régions restantes
        prenom = entropie_par_prenom.idxmin()
        retenus.append(prenom)

        # Sa région de plus forte part (parmi les régions actives) -> écartée
        lignes_prenom = sub_act[sub_act["preusuel"] == prenom]
        region_dominante = lignes_prenom.loc[lignes_prenom["part"].idxmax(), "region"]

        regions_actives.discard(region_dominante)
        eligibles.discard(prenom)

        # Éviction des prénoms dont le seul top régional était cette région :
        # leur top_regions ∩ regions_actives est désormais vide.
        a_evincer = {
            n for n in eligibles
            if not (top_regions_par_prenom.get(n, set()) & regions_actives)
        }
        eligibles -= a_evincer

    return retenus

# Application période par période, pour les DEUX grilles
selected_par_ref = {"national": {}, "regional": {}}
for periode, sub in counts.groupby("periode"):
    pops = select_national(sub, N_POP)
    selected_par_ref["national"][periode] = pops
    selected_par_ref["regional"][periode] = select_regional(sub, N_REG, exclure=pops)

# ── Construction de prof : fonction réutilisable pour CHAQUE référentiel ─────────
def construire_prof(selected):
    """Prend un dict {periode: [prénoms]} et construit le dataframe 'prof'
    (long format avec parts centrées, racine signée, ordres) pour ces prénoms."""
    paires = pd.DataFrame(
        [(p, n) for p, noms in selected.items() for n in noms],
        columns=["periode", "preusuel"]
    )
    regions_par_periode = counts[["periode", "region"]].drop_duplicates()
    grille = paires.merge(regions_par_periode, on="periode", how="left")

    prof = grille.merge(
        counts[["periode", "preusuel", "region", "part"]],
        on=["periode", "preusuel", "region"], how="left"
    )
    prof["part"] = prof["part"].fillna(0.0)
    prof["part_norm"] = prof["part"] / prof.groupby(["periode", "preusuel"])["part"].transform("sum")

    prof["part_moy"] = prof.groupby(["periode", "preusuel"])["part_norm"].transform("mean")
    prof["part_centree"] = prof["part_norm"] - prof["part_moy"]

    # Racine signée de l'écart à la moyenne : compresse les grands écarts pour
    # la hauteur des barres. Ce n'est PAS un z-score (pas de division par l'écart-type).
    prof["ecart_racine_signe"] = np.sign(prof["part_centree"]) * np.sqrt(prof["part_centree"].abs())
    prof["ecart_racine_signe"] = prof["ecart_racine_signe"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prof["signe"] = np.where(prof["ecart_racine_signe"] >= 0, "+", "-")

    prof["rang_region_case"] = (
        prof.groupby(["periode", "preusuel"])["part"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    disp = (
        prof.groupby(["periode", "preusuel"])["part"]
        .apply(lambda s: shannon_entropy(s.values))
        .rename("entropie")
        .reset_index()
    )
    disp["rang_prenom"] = (
        disp.groupby("periode")["entropie"].rank(ascending=True, method="first").astype(int)
    )
    prof = prof.merge(disp, on=["periode", "preusuel"], how="left")
    return prof

# Construire prof pour les DEUX référentiels et les empiler avec une colonne 'referentiel'
prof_national = construire_prof(selected_par_ref["national"])
prof_national["referentiel"] = "national"
prof_regional = construire_prof(selected_par_ref["regional"])
prof_regional["referentiel"] = "regional"
prof = pd.concat([prof_national, prof_regional], ignore_index=True)

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
# On ne passe au frontend que les lignes utiles (≈ (N_POP+N_REG) × n_periodes × n_regions)
selected_pairs = prof[["preusuel", "periode"]].drop_duplicates()
counts_map = counts.merge(selected_pairs, on=["preusuel", "periode"])
counts_map["periode"] = counts_map["periode"].astype(int)
counts_map["preusuel"] = counts_map["preusuel"].astype(str)

# Compléter avec les régions manquantes : pour chaque (prénom, période) retenu,
# on veut une ligne par région. Les régions sans naissance reçoivent le sentinel
# part=1, neutralisé en 0 plus bas pour la couleur (part_carte).
regions_list = sorted(counts["region"].unique())
paires = counts_map[["preusuel", "periode"]].drop_duplicates()

# Grille complète (prénom, période) × région via produit cartésien
grille_complete = paires.merge(pd.DataFrame({"region": regions_list}), how="cross")

# Lignes réellement observées, pour repérer les manquantes
observees = counts_map[["preusuel", "periode", "region"]]
manquantes = grille_complete.merge(
    observees, on=["preusuel", "periode", "region"], how="left", indicator=True
)
manquantes = manquantes[manquantes["_merge"] == "left_only"].drop(columns="_merge")
manquantes["part"] = 1  # sentinel pour « région sans ce prénom »

counts_map = pd.concat([counts_map, manquantes], ignore_index=True)

# ── Couleur de carte : part de naissance, normalisée par le MAX de la période ──
# Les régions sans ce prénom (sentinel part=1) doivent valoir 0 (= blanc), pas 1.
counts_map["part_carte"] = counts_map["part"].where(counts_map["part"] != 1, 0.0)
# Le sentinel a joué son rôle : on l'efface de `part` pour qu'il ne fuie pas dans
# le tooltip (sinon une région sans le prénom afficherait « 100.00% »).
counts_map["part"] = counts_map["part_carte"]
# max de part observé dans chaque période (sur tous les prénoms retenus et régions)
max_part_by_periode = counts_map.groupby("periode")["part_carte"].max().to_dict()
counts_map["max_part_periode"] = counts_map["periode"].map(max_part_by_periode)
# part rapportée au max de la période -> 0 (blanc) à 1 (violet foncé), échelle linéaire fixe
counts_map["part_sur_max_annee"] = counts_map["part_carte"] / counts_map["max_part_periode"]

# ── Sélection interactive : menus déroulants pour choisir le prénom et la période ─
# Valeur par défaut : prénom le plus populaire de la période la plus récente
default_periode = int(max(selected_par_ref["national"].keys()))
default_prenom = selected_par_ref["national"][default_periode][0]

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

# ── Deux facettes : "populaires" (national) et "régionaux" (régional) ───────────
def faire_facette(prof_sous, titre_colonnes):
    """Construit une grille de facettes à partir d'un sous-ensemble de prof."""
    bars = alt.Chart(prof_sous).mark_bar().encode(
        x=alt.X("rang_region_case:O", title=None, axis=None),
        y=alt.Y("ecart_racine_signe:Q",
                axis=alt.Axis(labels=False, ticks=False, domain=False), title=""),
        color=alt.Color("signe:N",
                scale=alt.Scale(domain=["+", "-"], range=["#d8642f", "#3a7ca5"]),
                legend=None),
        tooltip=[alt.Tooltip("periode:O", title="Période"),
                 alt.Tooltip("preusuel:N", title="Prénom"),
                 alt.Tooltip("region:N", title="Région"),
                 alt.Tooltip("part:Q", format=".2%", title="Part des naissances")],
    )
    labels = alt.Chart(prof_sous).mark_text(
        baseline="top", fontSize=8, dy=2
    ).encode(
        x=alt.value(40), y=alt.value(0), text=alt.Text("preusuel:N"),
    )
    case = alt.layer(bars, labels).properties(width=80, height=70)
    return case.facet(
        row=alt.Row("periode:O", title="Disparité par rapport à la moyenne (par période)", sort="descending"),
        column=alt.Column("rang_prenom:O", title=titre_colonnes, sort="ascending",
                          header=alt.Header(labelOrient="bottom")),
    ).resolve_scale(x="independent")

facet_national = faire_facette(
    prof[prof["referentiel"] == "national"],
    "Prénoms populaires (gauche = plus disparate)"
)
facet_regional = faire_facette(
    prof[prof["referentiel"] == "regional"],
    "Prénoms régionalement marqués (gauche = plus disparate)"
)

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
            "part_sur_max_annee": float(row["part_sur_max_annee"]),
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
        color=alt.Color("properties.part_sur_max_annee:Q",
                        scale=alt.Scale(scheme="purples", domain=[0, 1]),
                        title="Part / max de l'année"),
        tooltip=[alt.Tooltip("properties.nom:N", title="Région"),
                 alt.Tooltip("properties.part:Q", format=".2%", title="Part des naissances")],
    )
    .project(type="mercator")
    .properties(width=450, height=450, title="Part par région")
    .add_params(prenom_param, periode_param)
)

# ── Dashboard ─────────────────────────────────────────────────────────────────

dashboard = (facet_national | facet_regional | map_chart).properties(
    title=alt.TitleParams(
        "Disparité géographique des prénoms français (1900-2021) \n \n",
        subtitle=[" ",
                  "Visualisation des prénoms français par région et période.",
                  "L'indice de disparité mesure l'atypicité régionale : plus la barre est haute, plus le prénom est populaire dans cette région.",
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