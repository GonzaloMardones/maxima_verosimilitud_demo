"""
Máxima Verosimilitud aplicada a no-show de pasajeros
Autor: Gonzalo Mardones
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize
from scipy.stats import binom, norm


rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 11
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'

# Colores"
C_PRIMARY = '#0B3D91'     # azul profundo
C_ACCENT  = '#E63946'     # rojo señal
C_SOFT    = '#A8C0E0'     # azul claro
C_GREEN   = '#2A9D8F'     # verde dato
C_GRAY    = '#5C677D'

RNG = np.random.default_rng(42)
OUT = '/home/claude/mle_post'

# =============================================================
# 1. DATOS SINTÉTICOS REALISTAS
# =============================================================
# Supuesto: Ruta doméstica de alta frecuencia.
# Tenemos el histórico de 180 vuelos. Cada vuelo tiene N reservas y
# observamos cuántos pasajeros NO se presentaron (no-shows).
# El "no-show rate" verdadero (desconocido en la realidad) lo fijamos
# en p_true = 0.08 para poder evaluar qué tan bien lo recupera MLE.

p_true = 0.08
n_flights = 180
bookings_per_flight = RNG.integers(low=140, high=180, size=n_flights)

# Pequeña variación por vuelo (heterogeneidad leve)
# El modelo asume p constante: a ver cómo se comporta.
no_shows = np.array([
    RNG.binomial(n=n, p=p_true) for n in bookings_per_flight
])

df = pd.DataFrame({
    'flight_id':   [f'LA{1000+i}' for i in range(n_flights)],
    'bookings':    bookings_per_flight,
    'no_shows':    no_shows,
    'no_show_rate': no_shows / bookings_per_flight
})
df.to_csv(f'{OUT}/dataset_no_shows.csv', index=False)
print(df.head())
print(f"\nTotal reservas: {df['bookings'].sum()}, total no-shows: {df['no_shows'].sum()}")

# =============================================================
# 2. FUNCIÓN DE VEROSIMILITUD Y LOG-VEROSIMILITUD
# =============================================================
# Modelo: cada vuelo i tiene k_i no-shows de n_i reservas, k_i ~ Binomial(n_i, p)
# L(p | datos) = Π_i C(n_i, k_i) · p^{k_i} · (1-p)^{n_i - k_i}
# log L(p | datos) = Σ_i [ k_i·log(p) + (n_i - k_i)·log(1-p) ]   (constantes fuera)

def log_likelihood(p, k_arr, n_arr):
    if p <= 0 or p >= 1:
        return -np.inf
    return np.sum(k_arr * np.log(p) + (n_arr - k_arr) * np.log(1 - p))

# Negativa para minimizar
def neg_ll(p, k_arr, n_arr):
    return -log_likelihood(p, k_arr, n_arr)

k = df['no_shows'].values
n = df['bookings'].values

# Optimización numérica
res = minimize(neg_ll, x0=[0.1], args=(k, n), bounds=[(1e-6, 1-1e-6)])
p_mle_numeric = res.x[0]

# Solución (sabemos que el MLE binomial es Σk / Σn)
p_mle_analytic = k.sum() / n.sum()

# Error estándar via información de Fisher
# I(p) = Σn / (p(1-p))   →   SE = sqrt(p(1-p)/Σn)
se = np.sqrt(p_mle_analytic * (1 - p_mle_analytic) / n.sum())
ci_low, ci_high = p_mle_analytic - 1.96*se, p_mle_analytic + 1.96*se

print(f"\np_MLE numérico  = {p_mle_numeric:.5f}")
print(f"p_MLE analítico = {p_mle_analytic:.5f}")
print(f"IC 95%: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"p verdadero (oculto en la realidad) = {p_true}")

# =============================================================
# GRÁFICO 1 — Histograma de no-show rate observado
# =============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(df['no_show_rate'], bins=22, color=C_SOFT, edgecolor=C_PRIMARY, alpha=0.85)
ax.axvline(p_mle_analytic, color=C_ACCENT, linewidth=2.5,
           label=f'p estimado (MLE) = {p_mle_analytic:.3f}')
ax.axvline(p_true, color=C_GREEN, linewidth=2, linestyle='--',
           label=f'p verdadero (oculto) = {p_true:.3f}')
ax.set_xlabel('Tasa de no-show por vuelo')
ax.set_ylabel('Cantidad de vuelos')
ax.set_title('Lo que observamos: 180 vuelos, mucha variabilidad vuelo a vuelo')
ax.legend(frameon=False, loc='upper right')
ax.text(0.02, 0.97,
        f'n = {n_flights} vuelos\n{n.sum():,} reservas\n{k.sum():,} no-shows totales',
        transform=ax.transAxes, va='top', fontsize=10, color=C_GRAY,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=C_GRAY, alpha=0.9))
plt.tight_layout()
plt.savefig(f'{OUT}/01_histograma_observado.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# GRÁFICO 2 — Curva de log-verosimilitud
# =============================================================
p_grid = np.linspace(0.04, 0.13, 400)
ll_grid = np.array([log_likelihood(p, k, n) for p in p_grid])

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(p_grid, ll_grid, color=C_PRIMARY, linewidth=2.5)
ax.axvline(p_mle_analytic, color=C_ACCENT, linewidth=2,
           label=f'Máximo en p = {p_mle_analytic:.4f}')
ax.scatter([p_mle_analytic], [log_likelihood(p_mle_analytic, k, n)],
           color=C_ACCENT, s=120, zorder=5, edgecolor='white', linewidth=2)

# Marcar algunos puntos para mostrar que "otros p" son menos verosímiles
for p_test in [0.05, 0.10, 0.12]:
    ll_t = log_likelihood(p_test, k, n)
    ax.scatter([p_test], [ll_t], color=C_GRAY, s=50, zorder=4)
    ax.annotate(f'p={p_test}', (p_test, ll_t),
                textcoords='offset points', xytext=(8, -4),
                fontsize=9, color=C_GRAY)

ax.set_xlabel('Valor candidato de p (tasa de no-show)')
ax.set_ylabel('log-verosimilitud  log L(p | datos)')
ax.set_title('La pregunta de MLE: ¿qué valor de p hace MÁS PROBABLES los datos que vi?')
ax.legend(frameon=False, loc='lower center')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/02_log_verosimilitud.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# GRÁFICO 3 — Convergencia: cuántos vuelos necesitas para que MLE sea confiable
# =============================================================
sample_sizes = np.arange(5, n_flights+1, 1)
estimates = []
ci_low_arr = []
ci_high_arr = []
for m in sample_sizes:
    k_m = k[:m]
    n_m = n[:m]
    p_hat = k_m.sum() / n_m.sum()
    se_m = np.sqrt(p_hat*(1-p_hat)/n_m.sum())
    estimates.append(p_hat)
    ci_low_arr.append(p_hat - 1.96*se_m)
    ci_high_arr.append(p_hat + 1.96*se_m)

estimates = np.array(estimates)
ci_low_arr = np.array(ci_low_arr)
ci_high_arr = np.array(ci_high_arr)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(sample_sizes, ci_low_arr, ci_high_arr,
                color=C_SOFT, alpha=0.5, label='IC 95%')
ax.plot(sample_sizes, estimates, color=C_PRIMARY, linewidth=2,
        label='Estimación MLE acumulada')
ax.axhline(p_true, color=C_GREEN, linestyle='--', linewidth=1.8,
           label=f'p verdadero = {p_true}')
ax.set_xlabel('Cantidad de vuelos observados')
ax.set_ylabel('p estimado')
ax.set_title('Más datos → estimación más precisa (intervalo de confianza se cierra)')
ax.legend(frameon=False, loc='upper right')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/03_convergencia.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# GRÁFICO 4 — Aplicación de negocio: política de overbooking
# =============================================================
# Si sabemos que p_no_show ≈ 8%, ¿cuántos boletos extra podemos vender
# para un avión de 180 asientos sin pasar de un riesgo de denial > X%?
#
# Modelo: vendemos N reservas. Show-ups ~ Binomial(N, 1-p).
# Denial = max(0, show_ups - 180). Calculamos P(denial >= 1) para distintos N.

capacity = 180
p_show = 1 - p_mle_analytic
N_range = np.arange(180, 210)

risk = []
expected_denials = []
for N in N_range:
    # P(show_ups > capacity)
    risk.append(1 - binom.cdf(capacity, N, p_show))
    # E[max(0, show_ups - capacity)]
    ks = np.arange(0, N+1)
    pmf = binom.pmf(ks, N, p_show)
    expected_denials.append(np.sum(np.maximum(0, ks - capacity) * pmf))

risk = np.array(risk)
expected_denials = np.array(expected_denials)

# Encontrar overbooking que mantiene riesgo de denial <= 5%
threshold = 0.05
N_safe = N_range[risk <= threshold].max()
extra_seats = N_safe - capacity

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.plot(N_range, risk*100, color=C_PRIMARY, linewidth=2.5, marker='o', markersize=5)
ax.axhline(threshold*100, color=C_ACCENT, linestyle='--', linewidth=1.5,
           label=f'Tolerancia 5%')
ax.axvline(N_safe, color=C_GREEN, linestyle=':', linewidth=2,
           label=f'Máx. N seguro = {N_safe} ({extra_seats} extra)')
ax.set_xlabel('Reservas vendidas (capacidad = 180)')
ax.set_ylabel('Probabilidad de denegar embarque (%)')
ax.set_title('Riesgo operacional vs. cantidad de overbooking')
ax.legend(frameon=False, loc='upper left')
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.bar(N_range, expected_denials, color=C_SOFT, edgecolor=C_PRIMARY, alpha=0.85)
ax.axvline(N_safe, color=C_GREEN, linestyle=':', linewidth=2)
ax.set_xlabel('Reservas vendidas')
ax.set_ylabel('Pasajeros denegados esperados (E[denials])')
ax.set_title('Costo esperado: cuántos pax denegamos en promedio')
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/04_overbooking.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================
# GRÁFICO 5 — Diagrama conceptual
# =============================================================
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

boxes = [
    (1.0, 4.0, 'DATOS\nobservados\n(no-shows por vuelo)', C_SOFT),
    (4.0, 4.0, 'MODELO\nBinomial(n, p)\np es desconocido', '#FFE5B4'),
    (7.0, 4.0, 'FUNCIÓN DE\nVEROSIMILITUD\nL(p | datos)', '#D4F1D4'),
    (4.0, 1.0, 'OPTIMIZACIÓN\nbusca p que\nmaximiza L', '#FFD6D6'),
    (8.5, 1.0, 'p̂ MLE\n+ IC 95%\n+ decisión', C_GREEN),
]

for x, y, text, color in boxes:
    box = plt.Rectangle((x-0.9, y-0.6), 1.8, 1.2,
                        facecolor=color, edgecolor=C_PRIMARY,
                        linewidth=1.8, alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')

# Flechas
arrows = [
    ((1.9, 4.0), (3.1, 4.0)),
    ((4.9, 4.0), (6.1, 4.0)),
    ((7.0, 3.4), (4.5, 1.65)),
    ((4.9, 1.0), (7.6, 1.0)),
]
for (x1, y1), (x2, y2) in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color=C_PRIMARY))

ax.text(5, 5.6, 'Máxima Verosimilitud — el ciclo',
        ha='center', fontsize=15, weight='bold', color=C_PRIMARY)
ax.text(5, 0.1,
        'La intuición: "asumo una historia (modelo), y busco los parámetros que hacen mis datos\nlo más esperables posible bajo esa historia."',
        ha='center', fontsize=10, style='italic', color=C_GRAY)

plt.tight_layout()
plt.savefig(f'{OUT}/05_diagrama_ciclo.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✓ 5 gráficos generados en", OUT)
import os
for f in sorted(os.listdir(OUT)):
    print(' -', f)
