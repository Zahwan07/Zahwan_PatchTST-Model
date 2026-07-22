import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

# Path configurations
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "historical_environment.csv")
OUT_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    print(f"Loading dataset from: {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run fetch_openmeteo_historical.py first.")
    df = pd.read_csv(DATA_PATH)
    # Convert time to datetime format
    df['time'] = pd.to_datetime(df['time'])
    return df

def analyze_data_quality(df):
    print("\n=== 1. ANALISIS KUALITAS DATA ===")
    total_rows = len(df)
    print(f"Total baris data: {total_rows} jam")
    
    # Check missing values
    missing = df.isnull().sum()
    print("Jumlah missing values per kolom:")
    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count} ({count/total_rows*100:.2f}%)")
        else:
            print(f"  - {col}: 0 (0.00%)")
            
    # Check basic stats for outliers/extreme values
    stats = df[['suhu', 'humidity', 'light_intensity', 'ph', 'precipitation']].describe()
    print("\nStatistik Deskriptif Variabel Utama:")
    print(stats)
    
    # Save a boxplot of variables to check extremes
    plt.figure(figsize=(12, 6))
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    variables = ['suhu', 'humidity', 'light_intensity', 'ph', 'precipitation']
    colors = ['#ff6b6b', '#4dadf7', '#ffd43b', '#51cf66', '#a9e34b']
    
    for i, var in enumerate(variables):
        sns.boxplot(y=df[var], ax=axes[i], color=colors[i])
        axes[i].set_title(f"Boxplot {var.capitalize()}")
        axes[i].set_ylabel("")
        
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "distribusi_dan_nilai_ekstrem.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot distribusi dan nilai ekstrem disimpan ke: {plot_path}")

def plot_diurnal_patterns(df):
    print("\n=== 2. ANALISIS POLA TEMPORAL HARIAN (DIURNAL) ===")
    # Extract hour of day
    df['hour'] = df['time'].dt.hour
    
    # Compute hourly averages
    hourly_avg = df.groupby('hour')[['suhu', 'light_intensity', 'humidity']].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Temperature
    color = '#e74c3c'
    ax1.set_xlabel('Jam (00:00 - 23:00)')
    ax1.set_ylabel('Suhu Rata-rata (°C)', color=color)
    line1 = ax1.plot(hourly_avg['hour'], hourly_avg['suhu'], color=color, linewidth=2.5, marker='o', label='Suhu (°C)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(0, 24, 2))
    
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = '#f1c40f'
    ax2.set_ylabel('Intensitas Cahaya Rata-rata (W/m²)', color=color)
    line2 = ax2.plot(hourly_avg['hour'], hourly_avg['light_intensity'], color=color, linewidth=2.5, marker='^', linestyle='--', label='Intensitas Cahaya')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Title and Legend
    plt.title('Pola Temporal Harian (Diurnal) Suhu dan Intensitas Cahaya')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "pola_temporal_diurnal.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot pola diurnal disimpan ke: {plot_path}")

def plot_seasonal_patterns(df):
    print("\n=== 3. ANALISIS TREN TEMPORAL KESELURUHAN DATA (2020 - 2026) ===")
    
    # Resample to monthly average to show the clean trend from 2020 to 2026
    df_resampled = df.set_index('time').resample('ME')[['suhu', 'precipitation']].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Precipitation bar
    color = '#2ecc71'
    ax1.set_xlabel('Tahun (Periode Bulanan)')
    ax1.set_ylabel('Curah Hujan Rata-rata Bulanan (mm/jam)', color=color)
    bar = ax1.bar(df_resampled['time'], df_resampled['precipitation'], width=20, color=color, alpha=0.6, label='Curah Hujan (mm)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Suhu line
    ax2 = ax1.twinx()
    color = '#e74c3c'
    ax2.set_ylabel('Suhu Rata-rata Bulanan (°C)', color=color)
    line = ax2.plot(df_resampled['time'], df_resampled['suhu'], color=color, linewidth=2.0, marker='o', markersize=4, label='Suhu (°C)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Tren Temporal Bulanan dari Januari 2020 - Mei 2026')
    
    # Legend
    lines = [bar] + line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "pola_temporal_musiman.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot tren temporal keseluruhan disimpan ke: {plot_path}")

def plot_precipitation_sparsity(df):
    print("\n=== 4. ANALISIS KETIDAKSEIMBANGAN & SPARSITAS PRESIPITASI ===")
    total_hours = len(df)
    dry_hours = len(df[df['precipitation'] == 0])
    wet_hours = total_hours - dry_hours
    
    print(f"Total jam kering (tanpa hujan): {dry_hours} jam ({dry_hours/total_hours*100:.2f}%)")
    print(f"Total jam basah (hujan): {wet_hours} jam ({wet_hours/total_hours*100:.2f}%)")
    
    # Pie chart representation
    plt.figure(figsize=(8, 6))
    labels = ['Kering (0.0 mm)', 'Hujan (> 0.0 mm)']
    sizes = [dry_hours, wet_hours]
    colors = ['#ffdd59', '#3c6382']
    explode = (0, 0.1)  # explode the 2nd slice (wet hours)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12})
    plt.title('Sparsitas & Imbalance Data Presipitasi (Curah Hujan)')
    plt.axis('equal')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "sparsitas_presipitasi.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Plot sparsitas presipitasi disimpan ke: {plot_path}")

def main():
    try:
        df = load_data()
        analyze_data_quality(df)
        plot_diurnal_patterns(df)
        plot_seasonal_patterns(df)
        plot_precipitation_sparsity(df)
        print("\n[SUKSES] Seluruh visualisasi tren data berhasil dibuat di folder 'visualizations/'.")
    except Exception as e:
        print(f"\n[ERROR] Gagal membuat visualisasi data: {e}")

if __name__ == "__main__":
    main()
