import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation

# ==========================================
# 1. CONFIGURATION
# ==========================================

FPS = 60

# Une frame représente une minute fictive
MINUTES_PER_FRAME = 1

TOTAL_FRAMES = 24 * 60

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================

df = pd.read_csv(
    "data.csv",
    usecols=[
        "End_Lat",
        "End_Lng",
        "County",
        "Start_Time",
        "End_Time"
    ]
)

# Filtrer Manhattan
df = df[df["County"] == "New York"].copy()

# Nettoyer les coordonnées
df["End_Lat"] = pd.to_numeric(
    df["End_Lat"],
    errors="coerce"
)

df["End_Lng"] = pd.to_numeric(
    df["End_Lng"],
    errors="coerce"
)

df = df.dropna(
    subset=["End_Lat", "End_Lng"]
)

# Convertir les dates
df["Start_Time"] = pd.to_datetime(
    df["Start_Time"],
    errors="coerce",
    utc=True
)

df["End_Time"] = pd.to_datetime(
    df["End_Time"],
    errors="coerce",
    utc=True
)

df = df.dropna(
    subset=["Start_Time", "End_Time"]
)

# Supprimer les durées incohérentes
df = df[
    df["End_Time"] >= df["Start_Time"]
].copy()

# ==========================================
# 3. CONSTRUIRE LA JOURNÉE MOYENNE
# ==========================================

# Heure de début en minutes depuis minuit
df["start_minute"] = (
    df["Start_Time"].dt.hour * 60
    + df["Start_Time"].dt.minute
    + df["Start_Time"].dt.second / 60
)

# Durée réelle en minutes
df["duration"] = (
    df["End_Time"] - df["Start_Time"]
).dt.total_seconds() / 60

# Heure de fin fictive
df["end_fictional"] = (
    df["start_minute"] + df["duration"]
)

# ==========================================
# 4. CRÉER LES POINTS GÉOGRAPHIQUES
# ==========================================

accidents = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["End_Lng"],
        df["End_Lat"]
    ),
    crs="EPSG:4326"
)

# Charger les rues de Manhattan
roads = gpd.read_file(
    "rues_manhattan.gpkg",
    layer="rues"
)

# Même projection
accidents = accidents.to_crs(roads.crs)

# ==========================================
# 5. EXTRAIRE LES COORDONNÉES
# ==========================================

x = accidents.geometry.x.to_numpy()
y = accidents.geometry.y.to_numpy()

start_times = df["start_minute"].to_numpy()
end_times = df["end_fictional"].to_numpy()

# ==========================================
# 6. PRÉ-CALCUL DES ACCIDENTS ACTIFS
# ==========================================

active_by_minute = []

for minute in range(TOTAL_FRAMES):

    # Accidents ayant commencé avant ou pendant
    # la minute actuelle et toujours actifs
    active = np.flatnonzero(
        (start_times <= minute)
        &
        (end_times > minute)
    )

    # Accidents commencés la veille
    # et toujours actifs après minuit
    previous_day = np.flatnonzero(
        (end_times >= 1440 + minute)
    )

    # Regrouper les deux catégories
    active = np.unique(
        np.concatenate((
            active,
            previous_day
        ))
    )

    active_by_minute.append(active)

# ==========================================
# 7. PRÉPARER LA CARTE
# ==========================================

fig, ax = plt.subplots(
    figsize=(14, 10)
)

# Afficher le réseau routier
roads.plot(
    ax=ax,
    color="lightgray",
    linewidth=0.3,
    zorder=1
)

# Limites de la carte
bounds = roads.total_bounds

ax.set_xlim(
    bounds[0],
    bounds[2]
)

ax.set_ylim(
    bounds[1],
    bounds[3]
)

ax.set_aspect("equal")
ax.set_axis_off()

# Initialiser les points
scatter = ax.scatter(
    [],
    [],
    color="red",
    s=3,
    alpha=0.5,
    zorder=2
)

# Horloge
clock = ax.text(
    0.5,
    0.95,
    "",
    transform=ax.transAxes,
    fontsize=24,
    color="black",
    ha="center",
    fontweight="bold"
)

# Compteur
counter = ax.text(
    0.5,
    0.90,
    "",
    transform=ax.transAxes,
    fontsize=14,
    color="black",
    ha="center"
)

ax.set_title(
    "Évolution temporelle des accidents à Manhattan",
    fontsize=18,
    pad=20
)

# ==========================================
# 8. FONCTION D'ANIMATION
# ==========================================

def update(frame):

    current_minute = (
        frame * MINUTES_PER_FRAME
    )

    # Accidents actifs
    active = active_by_minute[frame]

    # Coordonnées
    positions = np.column_stack((
        x[active],
        y[active]
    ))

    scatter.set_offsets(positions)

    # Horloge
    hours = int(current_minute // 60)
    minutes = int(current_minute % 60)

    clock.set_text(
        f"{hours:02d}:{minutes:02d}"
    )

    # Compteur
    counter.set_text(
        f"Accidents actifs : {len(active):,}"
    )

    return scatter, clock, counter

# ==========================================
# 9. LANCER L'ANIMATION
# ==========================================

animation = FuncAnimation(
    fig,
    update,
    frames=TOTAL_FRAMES,
    interval=1000 / FPS,
    blit=False,
    repeat=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
