# ==============================================================================
# MICROSOFT FABRIC NOTEBOOK CELL: 3-JOURNEY BATTERY COMPARISON PLOT
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1. READ LAKEHOUSE DELTA TABLE
# Replace 'iot_telemetry' with your Lakehouse table name
df_raw = spark.read.table("iot_telemetry")

# Clean & filter base telemetry
df_clean = (
    df_raw
    .filter(
        (F.col("battery_voltage").between(2.0, 4.5)) &
        F.col("event_time").isNotNull() &
        F.col("asset_id").isNotNull() &
        F.col("device_id").isNotNull()
    )
    .withColumn("event_time", F.to_timestamp("event_time"))
)

# 2. IDENTIFY COMPLETED JOURNEYS (LOFU -> DIFU per asset_id)
w_asset = Window.partitionBy("asset_id").orderBy("event_time")

df_milestones = (
    df_clean
    .filter(F.col("event_name").isin("LOFU", "DIFU"))
    .withColumn("next_event", F.lead("event_name").over(w_asset))
    .withColumn("next_time", F.lead("event_time").over(w_asset))
    .withColumn("next_device_id", F.lead("device_id").over(w_asset))
    .filter(
        (F.col("event_name") == "LOFU") & 
        (F.col("next_event") == "DIFU") &
        (F.col("device_id") == F.col("next_device_id")) # Ensure same device was on the asset
    )
    .select(
        F.col("asset_id"),
        F.col("device_id"),
        F.col("event_time").alias("lofu_time"),
        F.col("next_time").alias("difu_time"),
        ((F.col("next_time").cast("long") - F.col("event_time").cast("long")) / 86400.0).alias("transit_days")
    )
    # Filter valid ocean journeys (>= 1 day to filter test pings; <= 90 days to eliminate stranded assets)
    .filter((F.col("transit_days") >= 1.0) & (F.col("transit_days") <= 90.0))
)

# 3. SELECT LONGEST, MEDIAN, AND SHORTEST JOURNEYS
pdf_journeys = df_milestones.orderBy("transit_days").toPandas()

if len(pdf_journeys) < 3:
    raise ValueError("Not enough completed LOFU-to-DIFU journeys found in the dataset.")

longest_row = pdf_journeys.iloc[-1]
median_row = pdf_journeys.iloc[len(pdf_journeys) // 2]
shortest_row = pdf_journeys.iloc[0]

target_journeys = [
    {"type": "Longest", "row": longest_row, "color": "#1565c0"},
    {"type": "Median",  "row": median_row,  "color": "#2e7d32"},
    {"type": "Shortest","row": shortest_row,"color": "#c62828"}
]

# 4. FETCH FULL TELEMETRY PINGS FOR THE 3 SELECTED JOURNEYS
telemetry_slices = []

for item in target_journeys:
    r = item["row"]
    pings_df = (
        df_clean
        .filter(
            (F.col("asset_id") == r["asset_id"]) &
            (F.col("device_id") == r["device_id"]) &
            (F.col("event_time") >= r["lofu_time"]) &
            (F.col("event_time") <= r["difu_time"])
        )
        .orderBy("event_time")
        .select("event_time", "battery_voltage", "event_name", "lat", "long")
        .toPandas()
    )
    
    # Calculate elapsed days from LOFU
    t0 = pings_df["event_time"].min()
    pings_df["elapsed_days"] = (pings_df["event_time"] - t0).dt.total_seconds() / 86400.0
    
    item["telemetry"] = pings_df
    telemetry_slices.append(item)

# 5. RENDER DIAGNOSTIC PLOTS
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharey=True)

# Define operational battery bands
bands = [
    (3.8, 4.2, "#e8f5e9", "Normal (>3.8V)"),
    (3.5, 3.8, "#fffde7", "Backup (3.5V - 3.8V)"),
    (3.3, 3.5, "#fff3e0", "Tertiary (3.3V - 3.5V)"),
    (2.9, 3.3, "#ffebee", "Reserve (2.9V - 3.3V)"),
    (2.6, 2.9, "#ffcdd2", "Hibernation (<2.9V)")
]

for idx, item in enumerate(telemetry_slices):
    ax = axes[idx]
    df_p = item["telemetry"]
    r = item["row"]
    
    # Fill operational band zones
    for y_low, y_high, bg_color, label in bands:
        ax.axhspan(y_low, y_high, color=bg_color, alpha=0.6, zorder=0)
        
    # Plot voltage line and pings
    ax.plot(df_p["elapsed_days"], df_p["battery_voltage"], color=item["color"], lw=1.8, zorder=3)
    ax.scatter(df_p["elapsed_days"], df_p["battery_voltage"], color=item["color"], s=18, alpha=0.75, zorder=4)
    
    # Mark LOFU (start) and DIFU (end) milestones
    ax.scatter(df_p["elapsed_days"].iloc[0], df_p["battery_voltage"].iloc[0], 
               color="#004d40", s=110, marker="^", zorder=5)
    ax.scatter(df_p["elapsed_days"].iloc[-1], df_p["battery_voltage"].iloc[-1], 
               color="#b71c1c", s=110, marker="v", zorder=5)
    
    # Compute summary stats
    v_start = df_p["battery_voltage"].iloc[0]
    v_end = df_p["battery_voltage"].iloc[-1]
    total_dv = (v_start - v_end) * 1000.0
    drain_day = total_dv / r["transit_days"]
    
    stat_box = (f"Start: {v_start:.2f}V  |  End: {v_end:.2f}V\n"
                f"Total ΔV: -{total_dv:.1f} mV\n"
                f"Avg Drain: {drain_day:.2f} mV/day")
    
    ax.text(0.985, 0.92, stat_box, transform=ax.transAxes,
            fontsize=9.5, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#b0bec5', alpha=0.92),
            zorder=6)
    
    title_str = f"{item['type']} Journey — Asset: {r['asset_id']} | Device: {r['device_id']} ({r['transit_days']:.1f} Days)"
    ax.set_title(title_str, fontsize=11, fontweight='bold', pad=6, loc='left')
    ax.set_ylabel("Voltage (V)", fontsize=10.5)
    ax.set_ylim(2.8, 4.05)
    ax.grid(True, linestyle='--', alpha=0.5)

axes[2].set_xlabel("Elapsed Ocean Transit Time (Days from LOFU to DIFU)", fontsize=11, fontweight='bold')

# Shared Figure Legend
legend_elements = [
    plt.Line2D([0], [0], color='#37474f', lw=2, label='Telemetry Voltage'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#004d40', markersize=10, label='LOFU (Voyage Start)'),
    plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#b71c1c', markersize=10, label='DIFU (Discharge End)'),
    Patch(facecolor="#e8f5e9", edgecolor="#81c784", label="Normal (>3.8V)"),
    Patch(facecolor="#fffde7", edgecolor="#fff59d", label="Backup (3.5 - 3.8V)"),
    Patch(facecolor="#fff3e0", edgecolor="#ffcc80", label="Tertiary (3.3 - 3.5V)"),
    Patch(facecolor="#ffebee", edgecolor="#ef9a9a", label="Reserve (2.9 - 3.3V)")
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.985),
           ncol=4, fontsize=9.5, frameon=True, facecolor='white', edgecolor='#b0bec5')

plt.subplots_adjust(top=0.92, bottom=0.06, hspace=0.28)
plt.show()