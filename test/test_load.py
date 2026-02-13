# test_load.py
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

print("=" * 60)
print("🚀 PRUEBA DE CARGA DE DATOS PARQUET (VERSIÓN CORREGIDA)")
print("=" * 60)

# Configurar rutas
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"\n📁 Buscando archivos en: {DATA_DIR}")

# 1. Buscar TODOS los archivos .parquet
parquet_files = list(glob.glob(os.path.join(DATA_DIR, "*.parquet")))

if not parquet_files:
    print("❌ No se encontraron archivos .parquet")
    print(f"Por favor, copia tus archivos a: {os.path.abspath(DATA_DIR)}")
    exit(1)

print(f"\n✅ Encontrados {len(parquet_files)} archivos:")
for i, file_path in enumerate(parquet_files, 1):
    file_name = os.path.basename(file_path)
    print(f"  {i}. {file_name}")


# 2. Función para extraer símbolo del nombre
def extract_symbol(filename):
    """Extrae el símbolo del nombre del archivo"""
    # Ejemplo: "BTCUSDT_1m_2020-01-01_to_2025-08-31.parquet" -> "BTC"
    parts = filename.split('_')
    symbol = parts[0].replace('USDT', '')  # Quita USDT
    return symbol


# 3. Cargar todos los archivos
dataframes = {}
for file_path in parquet_files:
    file_name = os.path.basename(file_path)
    symbol = extract_symbol(file_name)

    print(f"\n📊 CARGANDO {symbol}...")
    print(f"   Archivo: {file_name}")

    try:
        # Cargar el archivo
        df = pd.read_parquet(file_path)

        # Verificar estructura
        print(f"   ✅ Cargado: {len(df):,} registros")
        print(f"   📊 Columnas: {list(df.columns)}")

        # IMPORTANTE: Asegurar que 'open_time' sea datetime y sea el índice
        if 'open_time' in df.columns:
            print(f"   🔧 Convirtiendo 'open_time' a datetime...")
            df['open_time'] = pd.to_datetime(df['open_time'])
            df.set_index('open_time', inplace=True)
            print(f"   ✅ Índice establecido como datetime")

        # Mostrar rango temporal
        if isinstance(df.index, pd.DatetimeIndex):
            print(f"   📅 Rango: {df.index.min()} -> {df.index.max()}")
        else:
            print(f"   ⚠️  Índice no es datetime, es tipo: {type(df.index).__name__}")
            print(f"   🔧 Intentando convertir el índice a datetime...")
            try:
                df.index = pd.to_datetime(df.index)
                print(f"   ✅ Índice convertido a datetime")
                print(f"   📅 Rango: {df.index.min()} -> {df.index.max()}")
            except:
                print(f"   ❌ No se pudo convertir el índice")

        # Mostrar precio más reciente
        if 'close' in df.columns:
            print(f"   💰 Precio más reciente: ${df['close'].iloc[-1]:,.2f}")
            print(f"   📈 Precio inicial: ${df['close'].iloc[0]:,.2f}")
            print(f"   🎯 Retorno total: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")

        # Mostrar estadísticas básicas
        if 'close' in df.columns:
            print(f"   📊 Volumen total: {df['volume'].sum():,.0f}")
            print(f"   📊 Precio promedio: ${df['close'].mean():,.2f}")

        # Guardar en diccionario
        dataframes[symbol] = df

    except Exception as e:
        print(f"   ❌ Error cargando {file_name}: {e}")
        import traceback

        traceback.print_exc()

# 4. Resumen de datos cargados
print("\n" + "=" * 60)
print("🔍 RESUMEN DE DATOS CARGADOS")
print("=" * 60)

for symbol, df in dataframes.items():
    print(f"\n{symbol}:")
    print(f"  Registros: {len(df):,}")
    print(f"  Columnas: {df.columns.tolist()}")

    # Manejar diferentes tipos de índice
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"  Periodo: {df.index.min().date()} al {df.index.max().date()}")
    else:
        print(f"  Periodo: No disponible (índice no datetime)")

    if 'close' in df.columns:
        print(f"  Precio inicial: ${df['close'].iloc[0]:,.2f}")
        print(f"  Precio final: ${df['close'].iloc[-1]:,.2f}")
        ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        print(f"  Retorno total: {ret:.2f}%")

# 5. Guardar en feather para procesamiento rápido
print("\n" + "=" * 60)
print("💾 GUARDANDO EN FORMATO FEATHER...")
print("=" * 60)

os.makedirs("data/processed", exist_ok=True)

for symbol, df in dataframes.items():
    try:
        # Guardar datos de 1 minuto
        feather_path_1m = f"data/processed/{symbol.lower()}_1m.feather"
        df.reset_index().to_feather(feather_path_1m)  # reset_index para guardar el datetime como columna
        print(f"  ✅ {symbol} guardado como: {symbol.lower()}_1m.feather")

        # También guardar como CSV (opcional, por si acaso)
        csv_path = f"data/processed/{symbol.lower()}_1m.csv"
        df.to_csv(csv_path)
        print(f"  ✅ {symbol} guardado como: {symbol.lower()}_1m.csv")

    except Exception as e:
        print(f"  ❌ Error guardando {symbol}: {e}")

# 6. Información adicional
print("\n" + "=" * 60)
print("📊 ESTADÍSTICAS COMPARATIVAS")
print("=" * 60)

print("\n💰 PRECIOS FINALES:")
for symbol, df in sorted(dataframes.items()):
    if 'close' in df.columns:
        price = df['close'].iloc[-1]
        print(f"  {symbol}: ${price:,.2f}")

print("\n📈 RETORNOS TOTALES (desde inicio de datos):")
for symbol, df in sorted(dataframes.items()):
    if 'close' in df.columns and len(df) > 1:
        ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        print(f"  {symbol}: {ret:+.2f}%")

print("\n📊 VOLATILIDAD (std de retornos diarios aproximados):")
for symbol, df in sorted(dataframes.items()):
    if 'close' in df.columns and len(df) > 100:
        # Calcular retornos logarítmicos (aproximados)
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        if len(returns) > 0:
            vol_daily = returns.std()
            vol_annual = vol_daily * np.sqrt(365 * 24 * 60)  # Aproximación para datos de 1 minuto
            print(f"  {symbol}: {vol_annual * 100:.2f}% anual")

print("\n" + "=" * 60)
print("🎉 CARGA COMPLETADA - LISTO PARA EL SIGUIENTE PASO")
print("=" * 60)
print("\n📋 PRÓXIMOS PASOS:")
print("1. Ejecutar: python resample_to_daily.py")
print("2. Ejecutar: python calculate_params.py")
print("3. Ejecutar: python test_simulation.py")
print("\n📁 Archivos guardados en: data/processed/")
print("   - btc_1m.feather, eth_1m.feather, etc.")
print("   - btc_1m.csv, eth_1m.csv, etc.")