# test_simulation.py
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

print("=" * 70)
print("🎲 SIMULACIÓN MONTE CARLO - PRIMERA VERSIÓN OPERACIONAL")
print("=" * 70)

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Cargar parámetros
print("\n📥 CARGANDO PARÁMETROS...")
params_path = "data/processed/parameters.json"

if not os.path.exists(params_path):
    print(f"❌ No se encontró {params_path}")
    print("Ejecuta primero: python calculate_params.py")
    exit(1)

with open(params_path, 'r') as f:
    params = json.load(f)

print(f"✅ Parámetros cargados ({len(params['assets'])} activos)")

# 2. Seleccionar activos para simulación (BTC y ETH primero)
print("\n🎯 SELECCIONANDO ACTIVOS PARA SIMULACIÓN...")
target_symbols = ['BTC', 'ETH']  # Empezamos con estos 2

selected_params = {}
for symbol in target_symbols:
    if symbol in params['assets']:
        selected_params[symbol] = params['assets'][symbol]
        print(f"   ✅ {symbol}: μ={selected_params[symbol]['mu']:.6f}, σ={selected_params[symbol]['sigma']:.6f}")
    else:
        print(f"   ❌ {symbol} no encontrado en parámetros")

if len(selected_params) < 2:
    print("⚠️  No se encontraron BTC y ETH, usando primeros 2 activos disponibles...")
    available_symbols = list(params['assets'].keys())[:2]
    for symbol in available_symbols:
        selected_params[symbol] = params['assets'][symbol]

# 3. Configurar simulación
print("\n⚙️  CONFIGURANDO SIMULACIÓN...")
n_simulations = 10000  # Número de trayectorias
n_days = 252  # 1 año de trading (252 días)
dt = 1 / 252  # Paso de tiempo diario

# Precios iniciales (usamos los precios finales de los datos históricos)
initial_prices = {}
for symbol in selected_params:
    initial_prices[symbol] = selected_params[symbol]['final_price']

print(f"   Simulaciones: {n_simulations:,}")
print(f"   Horizonte: {n_days} días ({n_days / 252:.1f} años)")
print(f"   Precios iniciales: {initial_prices}")

# 4. Obtener matriz de correlación
print("\n🔗 CONFIGURANDO CORRELACIONES...")
corr_matrix = None

# Intentar obtener correlación específica BTC-ETH
if 'btc_eth_correlation' in params:
    btc_eth_corr = params['btc_eth_correlation']
    print(f"   Correlación BTC-ETH específica: {btc_eth_corr:.4f}")
    corr_matrix = np.array([[1.0, btc_eth_corr], [btc_eth_corr, 1.0]])
elif 'correlation_matrix' in params and len(selected_params) >= 2:
    symbols = list(selected_params.keys())
    try:
        corr_dict = params['correlation_matrix']
        corr_matrix = np.array([[corr_dict[s1][s2] for s2 in symbols] for s1 in symbols])
        print(f"   Matriz de correlación cargada")
    except:
        corr_matrix = np.eye(len(symbols))
        print(f"   ⚠️  Usando matriz identidad (sin correlación)")
else:
    corr_matrix = np.eye(len(selected_params))
    print(f"   ⚠️  Usando matriz identidad (sin correlación)")

print(f"\n📊 MATRIZ DE CORRELACIÓN:")
print(corr_matrix)

# 5. Función de simulación GBM correlacionado
print("\n🧮 EJECUTANDO SIMULACIÓN MONTE CARLO...")


def simulate_gbm_correlated(initial_prices, mus, sigmas, corr_matrix, n_days, n_simulations, dt=1 / 252):
    """
    Simulación GBM multivariada con correlación usando Cholesky
    """
    n_assets = len(initial_prices)

    # Descomposición de Cholesky de la matriz de correlación
    L = np.linalg.cholesky(corr_matrix)

    # Array para almacenar resultados
    # Forma: (n_assets, n_days+1, n_simulations)
    paths = np.zeros((n_assets, n_days + 1, n_simulations))

    # Inicializar con precios iniciales
    for i in range(n_assets):
        paths[i, 0, :] = initial_prices[i]

    # Generar ruido correlacionado
    np.random.seed(42)  # Para reproducibilidad
    Z = np.random.normal(0, 1, (n_assets, n_days, n_simulations))
    correlated_Z = np.einsum('ij,jkl->ikl', L, Z)

    # Simular trayectorias
    for t in range(1, n_days + 1):
        for i in range(n_assets):
            drift = (mus[i] - 0.5 * sigmas[i] ** 2) * dt
            diffusion = sigmas[i] * np.sqrt(dt) * correlated_Z[i, t - 1, :]
            paths[i, t, :] = paths[i, t - 1, :] * np.exp(drift + diffusion)

    return paths


# Preparar arrays para la simulación
symbols = list(selected_params.keys())
initial_array = [initial_prices[s] for s in symbols]
mus_array = [selected_params[s]['mu'] for s in symbols]
sigmas_array = [selected_params[s]['sigma'] for s in symbols]

# Ejecutar simulación
print("   Simulando... Esto puede tomar unos segundos...")
paths = simulate_gbm_correlated(
    initial_array, mus_array, sigmas_array,
    corr_matrix, n_days, n_simulations, dt
)

print(f"   ✅ Simulación completada: {paths.shape}")

# 6. Calcular métricas de riesgo
print("\n📊 CALCULANDO MÉTRICAS DE RIESGO...")

# Precios finales de las simulaciones
final_prices = paths[:, -1, :]

# Retornos totales del período
returns = (final_prices / np.array(initial_array).reshape(-1, 1)) - 1


# Funciones para métricas de riesgo
def calculate_var(returns_array, confidence=0.95):
    """Value at Risk (VaR)"""
    return np.percentile(returns_array, (1 - confidence) * 100)


def calculate_cvar(returns_array, confidence=0.95):
    """Conditional Value at Risk (CVaR) / Expected Shortfall"""
    var = calculate_var(returns_array, confidence)
    return returns_array[returns_array <= var].mean()


def calculate_probability_of_loss(returns_array, threshold=-0.20):
    """Probabilidad de pérdida mayor a cierto umbral"""
    return np.mean(returns_array < threshold) * 100


# Calcular métricas para cada activo
risk_metrics = {}
for i, symbol in enumerate(symbols):
    asset_returns = returns[i, :]

    risk_metrics[symbol] = {
        'mean_return': np.mean(asset_returns) * 100,
        'std_return': np.std(asset_returns) * 100,
        'var_95': calculate_var(asset_returns, 0.95) * 100,
        'var_99': calculate_var(asset_returns, 0.99) * 100,
        'cvar_95': calculate_cvar(asset_returns, 0.95) * 100,
        'cvar_99': calculate_cvar(asset_returns, 0.99) * 100,
        'prob_loss_10': calculate_probability_of_loss(asset_returns, -0.10),
        'prob_loss_20': calculate_probability_of_loss(asset_returns, -0.20),
        'prob_loss_30': calculate_probability_of_loss(asset_returns, -0.30),
        'best_5_percent': np.percentile(asset_returns, 95) * 100,
        'worst_5_percent': np.percentile(asset_returns, 5) * 100,
        'median_return': np.median(asset_returns) * 100
    }

# 7. Mostrar resultados
print("\n" + "=" * 70)
print("📈 RESULTADOS DE SIMULACIÓN (1 AÑO)")
print("=" * 70)

for symbol in symbols:
    print(f"\n💰 {symbol}:")
    metrics = risk_metrics[symbol]
    print(f"   Retorno esperado: {metrics['mean_return']:.2f}%")
    print(f"   Volatilidad esperada: {metrics['std_return']:.2f}%")
    print(f"   Retorno mediano: {metrics['median_return']:.2f}%")
    print(f"   Mejor 5%: +{metrics['best_5_percent']:.2f}%")
    print(f"   Peor 5%: {metrics['worst_5_percent']:.2f}%")
    print(f"\n   📉 RIESGO:")
    print(f"   VaR 95% (pérdida máxima en 95% casos): {metrics['var_95']:.2f}%")
    print(f"   VaR 99% (pérdida máxima en 99% casos): {metrics['var_99']:.2f}%")
    print(f"   CVaR 95% (pérdida promedio en peor 5%): {metrics['cvar_95']:.2f}%")
    print(f"   CVaR 99% (pérdida promedio en peor 1%): {metrics['cvar_99']:.2f}%")
    print(f"\n   🎲 PROBABILIDADES:")
    print(f"   Pérdida >10%: {metrics['prob_loss_10']:.1f}%")
    print(f"   Pérdida >20%: {metrics['prob_loss_20']:.1f}%")
    print(f"   Pérdida >30%: {metrics['prob_loss_30']:.1f}%")

# 8. Crear visualizaciones
print("\n🎨 CREANDO VISUALIZACIONES...")
os.makedirs("../outputs", exist_ok=True)

# Figura 1: Trayectorias simuladas
fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
fig1.suptitle('Simulación Monte Carlo - BTC y ETH (1 año)', fontsize=16, fontweight='bold')

for idx, symbol in enumerate(symbols[:2]):  # Mostrar solo BTC y ETH
    ax1 = axes[idx, 0]
    ax2 = axes[idx, 1]

    # Gráfico de trayectorias (primeras 100)
    for sim in range(min(100, n_simulations)):
        ax1.plot(paths[idx, :, sim], alpha=0.1, linewidth=0.5)

    ax1.axhline(y=initial_array[idx], color='red', linestyle='--', alpha=0.7,
                label=f'Precio inicial: ${initial_array[idx]:,.2f}')
    ax1.set_title(f'{symbol} - 100 Trayectorias Simuladas')
    ax1.set_xlabel('Días')
    ax1.set_ylabel('Precio (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histograma de retornos finales
    ax2.hist(returns[idx, :] * 100, bins=50, alpha=0.7, edgecolor='black', density=True)

    # Añadir líneas de VaR
    var_95 = risk_metrics[symbol]['var_95']
    var_99 = risk_metrics[symbol]['var_99']

    ax2.axvline(x=var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
    ax2.axvline(x=var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.2f}%')
    ax2.axvline(x=0, color='green', linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_title(f'{symbol} - Distribución de Retornos (1 año)')
    ax2.set_xlabel('Retorno %')
    ax2.set_ylabel('Densidad')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig('outputs/simulated_paths.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Figura 1 guardada: outputs/simulated_paths.png")

# Figura 2: Comparación de riesgo
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Comparación de Riesgo entre Activos', fontsize=16, fontweight='bold')

# Gráfico de barras: VaR y CVaR
metrics_to_plot = ['var_95', 'cvar_95']
x = np.arange(len(symbols))
width = 0.35

for i, metric in enumerate(metrics_to_plot):
    values = [risk_metrics[s][metric] for s in symbols]
    ax1.bar(x + i * width, values, width, label=metric.replace('_', ' ').upper())

ax1.set_xlabel('Activo')
ax1.set_ylabel('Pérdida %')
ax1.set_title('VaR 95% vs CVaR 95%')
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(symbols)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico de probabilidades de pérdida
prob_metrics = ['prob_loss_10', 'prob_loss_20', 'prob_loss_30']
x = np.arange(len(symbols))
width = 0.25

for i, metric in enumerate(prob_metrics):
    values = [risk_metrics[s][metric] for s in symbols]
    ax2.bar(x + i * width, values, width, label=metric.replace('_', ' ').upper())

ax2.set_xlabel('Activo')
ax2.set_ylabel('Probabilidad %')
ax2.set_title('Probabilidades de Pérdida')
ax2.set_xticks(x + width)
ax2.set_xticklabels(symbols)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('outputs/risk_comparison.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Figura 2 guardada: outputs/risk_comparison.png")

# 9. Guardar resultados en JSON
print("\n💾 GUARDANDO RESULTADOS...")
results = {
    'simulation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'parameters': {
        'n_simulations': n_simulations,
        'n_days': n_days,
        'dt': dt,
        'initial_prices': initial_prices,
        'correlation_matrix': corr_matrix.tolist()
    },
    'risk_metrics': risk_metrics,
    'assets': symbols
}

results_path = 'data/processed/simulation_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"   ✅ Resultados guardados: {results_path}")

# 10. Conclusiones prácticas
print("\n" + "=" * 70)
print("💡 CONCLUSIONES PRÁCTICAS")
print("=" * 70)

print("\n📌 RESUMEN EJECUTIVO:")
print(f"Simulación de {n_simulations:,} escenarios para {len(symbols)} activos")
print(f"Horizonte: 1 año ({n_days} días de trading)")

for symbol in symbols:
    metrics = risk_metrics[symbol]
    print(f"\n📊 {symbol}:")
    print(f"  • Hay {metrics['prob_loss_20']:.1f}% probabilidad de perder más del 20% en 1 año")
    print(f"  • En el peor 5% de casos, la pérdida promedio es {abs(metrics['cvar_95']):.2f}%")
    print(f"  • El escenario esperado es un retorno de {metrics['mean_return']:.2f}%")
    print(
        f"  📈 Recomendación: {'ALTO RIESGO' if metrics['prob_loss_20'] > 30 else 'RIESGO MODERADO' if metrics['prob_loss_20'] > 20 else 'RIESGO ACEPTABLE'}")

print("\n" + "=" * 70)
print("🎯 PRÓXIMOS PASOS RECOMENDADOS:")
print("1. Analizar portafolio 50/50 BTC-ETH")
print("2. Probar diferentes horizontes temporales")
print("3. Incorporar modelo con saltos (extremos)")
print("4. Añadir más activos (SOL, ADA, BNB)")
print("=" * 70)