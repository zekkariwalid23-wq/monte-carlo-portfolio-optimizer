# monte-carlo-portfolio-optimizer
!pip install yfinance
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. PARAMÈTRES (modifiez-les pour expérimenter)
# ------------------------------------------------------------
TICKER = "TSLA"          # Action à étudier : "TSLA", "MSFT", "MC.PA"...
DATE_DEBUT = "2020-01-01"
ANNEES = 1               # Durée de la simulation en années
JOURS_BOURSE = 252       # Nombre de jours de bourse par an
N_SIMS = 10000      # Nombre de futurs simulés

np.random.seed(42)       # Même hasard à chaque exécution (retirez pour varier)

# ------------------------------------------------------------
# 1. RÉCUPÉRER LES VRAIS PRIX
# ------------------------------------------------------------
data = yf.download(TICKER, start=DATE_DEBUT, progress=False)["Close"].squeeze()
data = data.dropna()
print(f"{len(data)} jours de données téléchargés pour {TICKER}")

# ------------------------------------------------------------
# 2. CALCULER LES RENDEMENTS ET LES PARAMÈTRES
# ------------------------------------------------------------
# Rendement logarithmique quotidien = variation de chaque jour
log_returns = np.log(data / data.shift(1)).dropna()

# Tendance (mu) et volatilité (sigma), annualisées
mu = log_returns.mean() * JOURS_BOURSE
sigma = log_returns.std() * np.sqrt(JOURS_BOURSE)

print(f"Tendance annuelle (mu)     : {mu:.2%}")
print(f"Volatilité annuelle (sigma): {sigma:.2%}")

# ------------------------------------------------------------
# 3. SIMULATION
# ------------------------------------------------------------
S0 = float(data.iloc[-1])             # Dernier prix connu
steps = ANNEES * JOURS_BOURSE         # Nombre de jours simulés
dt = 1 / JOURS_BOURSE                 # Un jour exprimé en années

# Tableau de hasard : une ligne par jour, une colonne par simulation
Z = np.random.standard_normal((steps, N_SIMS))

# Variation de chaque jour = tendance + hasard
increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

# On additionne les variations, puis on convertit en prix
paths = S0 * np.exp(np.cumsum(increments, axis=0))

# On ajoute le prix de départ en première ligne (jour 0)
paths = np.vstack([np.full(N_SIMS, S0), paths])

# ------------------------------------------------------------
# 4. GRAPHIQUES
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4a. Quelques trajectoires possibles
axes[0].plot(paths[:, :100], alpha=0.3, linewidth=0.8)
axes[0].axhline(S0, color="black", linestyle="--", label=f"Prix actuel : {S0:.2f}")
axes[0].set_title(f"{TICKER} : 100 trajectoires simulées")
axes[0].set_xlabel("Jours de bourse")
axes[0].set_ylabel("Prix")
axes[0].legend()

# 4b. Distribution des prix finaux
final = paths[-1]
p5, p50, p95 = np.percentile(final, [5, 50, 95])

axes[1].hist(final, bins=60, color="steelblue", alpha=0.8)
axes[1].axvline(S0, color="black", linestyle="--", label="Prix actuel")
axes[1].axvline(p5, color="red", linestyle="--", label=f"5% pires cas : {p5:.2f}")
axes[1].axvline(p95, color="green", linestyle="--", label=f"5% meilleurs cas : {p95:.2f}")
axes[1].set_title("Distribution des prix finaux")
axes[1].set_xlabel("Prix à la fin de la période")
axes[1].set_ylabel("Nombre de simulations")
axes[1].legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. STATISTIQUES DE RISQUE
# ------------------------------------------------------------
print("\n===== RÉSULTATS =====")
print(f"Prix actuel                  : {S0:.2f}")
print(f"Prix moyen simulé            : {final.mean():.2f}")
print(f"Prix médian simulé           : {p50:.2f}")
print(f"5e percentile (pire cas)     : {p5:.2f}")
print(f"95e percentile (meilleur cas): {p95:.2f}")
print(f"VaR 95% (perte max probable) : {S0 - p5:.2f} par action")
print(f"Probabilité de finir en perte: {(final < S0).mean():.1%}")
print(f"Probabilité de gagner +20%   : {(final > S0 * 1.2).mean():.1%}")
resume = pd.DataFrame({
    "Indicateur": ["Prix actuel", "Prix moyen", "Prix médian", "Pire cas (5%)", "Meilleur cas (95%)", "VaR 95% ($)", "Probabilité de perte"],
    "Valeur": [S0, final.mean(), p50, p5, p95, S0 - p5, (final < S0).mean()]
})
resume.to_excel("resultats_montecarlo.xlsx", index=False)

