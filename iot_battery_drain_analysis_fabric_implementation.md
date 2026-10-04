# IoT Asset Tracker Battery Life & Drain Analysis
## Architecture, Electrochemical Mechanics, and Microsoft Fabric Implementation

---

## 1. Executive Summary & Problem Formulation

Predicting the Remaining Useful Life (RUL) of battery-powered supply chain IoT asset trackers is a non-linear problem. Trackers deployed on intermodal shipping containers typically operate on primary Lithium-thionyl chloride ($\text{Li-SOCl}_2$) or advanced Lithium-Manganese Dioxide ($\text{Li-MnO}_2$) chemistries. 

These cells exhibit a uniquely flat discharge plateau across $80\%$ to $90\%$ of their service life, followed by a sharp precipice ("the voltage cliff"). Consequently, calculating remaining capacity via a single linear conversion of raw battery voltage produces severe estimation errors. 

Furthermore, raw voltage readings fluctuate in the field due to:
1. **Operating load:** Cellular transmit bursts ($+23\text{ dBm}$) versus deep sleep.
2. **Ambient temperature:** Electrochemical impedance spikes causing voltage sag.
3. **RF search cycles:** Ocean maritime dead zones causing continuous high-power radio ping attempts.

This document compiles the domain research, operational band specifications, and production-ready Microsoft Fabric PySpark code to answer:
- **Question 1:** Battery voltage drain variance across Latitude and Longitude.
- **Question 2:** Battery voltage drain patterns across Seasonality ($\text{MM-YY}$) and Hemisphere.
- **Operational Milestone:** Battery consumption across the primary ocean transit leg ($\text{LOFU} \to \text{DIFU}$).
- **Predictive Engine:** Non-linear Days-to-Hibernation prediction per device.

---

## 2. Theoretical Analysis of the Core Questions

### 2.1 Battery Voltage Drain by Latitude and Longitude

Geographic coordinates govern battery drain through two primary physical drivers: **thermal impedance** and **RF connectivity / cell-hunting behavior**.

#### A. Thermal Mechanics (Latitude)
* **High Latitudes ($> 45^\circ\text{N/S}$):** Sub-zero winter sea lanes (e.g., North Pacific, North Atlantic) subject containers to temperatures between $-10^\circ\text{C}$ and $-40^\circ\text{C}$. Electrolyte viscosity increases and internal cell resistance ($R_{\text{int}}$) rises sharply. When the cellular modem wakes and pulls pulse currents of $1.0\text{A} - 2.0\text{A}$, the cell experiences significant **transient voltage sag**:
  $$V_{\text{terminal}} = V_{\text{OCV}} - I_{\text{pulse}} \cdot R_{\text{int}}(T)$$
  This can cause premature trips into lower operational bands, even when chemical capacity remains high.
* **Equatorial Latitudes ($0^\circ - 20^\circ$):** Persistent ambient heat ($> 35^\circ\text{C}$, reaching $> 55^\circ\text{C}$ inside steel container walls under direct sun) reduces internal impedance, giving an artificially high voltage reading under load. However, the Arrhenius relationship accelerates chemical self-discharge, permanently stripping $2\% - 4\%$ of nominal capacity per year.

#### B. RF Environment & Maritime Dead Zones (Longitude & Coastlines)
* **Open Ocean Dead Zones:** Between departure and arrival ports, terrestrial cellular coverage (LTE-M, NB-IoT, 2G) vanishes. If tracker firmware lacks adaptive exponential backoff, modems enter repeated high-power cell registration loops, burning up to $50\times$ more energy per hour than stationary sleep.
* **Faraday Cage Effects in Vessel Stows:** Containers stowed below deck inside cargo holds are shielded by thick steel hull plates. Both GPS satellite reception and cellular signals are blocked. Devices continuously hit acquisition timeouts (e.g., $120\text{s} - 180\text{s}$ searching for satellites vs. the typical $15\text{s}$ time-to-first-fix), rapidly draining the battery.

---

### 2.2 Battery Voltage Drain by Season ($\text{MM-YY}$)

Seasonal evaluation requires aligning calendar months with geographic hemisphere:

| Parameter | Northern Hemisphere Winter / Southern Summer (`12-yy` to `02-yy`) | Northern Hemisphere Summer / Southern Winter (`06-yy` to `08-yy`) |
|---|---|---|
| **Voltage Behavior (North)** | Severe transient sag under modem load | Apparent voltage "bounce" (higher open-circuit voltage) |
| **Operational Impact (North)** | Slower ship voyages, port weather congestion, extended dwell | Smooth maritime crossings, faster port velocity |
| **Voltage Behavior (South)** | Warm ocean lanes, normal voltage curves | Cold southern route exposure (e.g., Cape of Good Hope sag) |

#### Voltage "Bounce" vs. Real Capacity
In spring (`03-yy` to `05-yy` in Northern latitudes), ambient warming causes devices that previously reported degraded voltage (e.g., $3.4\text{V}$) to rebound back to $3.6\text{V}$ without any recharging. Linear life predictors will falsely record negative drain (apparent capacity gain). Models must account for this phenomenon using monotonic smoothing and moving averages.

---

### 2.3 Operational Battery Voltage Bands

The tracker hardware operates across five distinct functional states:

```
+-------------------------------------------------------------------------+
| > 3.8V          | NORMAL OPERATION  | Full sensor sampling & regular pings |
+-------------------------------------------------------------------------+
| 3.5V - 3.8V     | BACKUP BAND       | Standard tracking, slight throttling|
+-------------------------------------------------------------------------+
| 3.3V - 3.5V     | TERTIARY BAND     | Reduced ping rate, shorter GPS caps |
+-------------------------------------------------------------------------+
| 2.9V - 3.3V     | RESERVE BAND      | Emergency mode, cell-ID only, cliff |
+-------------------------------------------------------------------------+
| < 2.9V          | HIBERNATION       | Modem shutdown; brownout protection |
+-------------------------------------------------------------------------+
```

1. **Normal Operation ($> 3.8\text{V}$):** Nominal operating zone. Active reporting cadence (e.g., every 2 to 4 hours). Voltage stays stable for extended periods.
2. **Backup Band ($3.5\text{V} - 3.8\text{V}$):** Safe operating zone with minimal power degradation. Standard tracking continues.
3. **Tertiary Band ($3.3\text{V} - 3.5\text{V}$):** The transition knee. Cold weather snaps will routinely push cells from here into the reserve band. Firmware typically cuts sampling frequency in half.
4. **Reserve Band ($2.9\text{V} - 3.3\text{V}$):** Emergency phase. The battery has entered its terminal discharge slope; remaining capacity is $< 10\%$. Pings should be throttled to 1 per day.
5. **Hibernation ($< 2.9\text{V}$):** The modem is unpowered to prevent brownouts. The device is considered operationally dead until hardware replacement.

---

### 2.4 Supply Chain Journey Context

The dataset tracks six milestone events defining the container lifecycle:
* **`GOMT` (Gate Out Empty):** Empty container departs depot for shipper loading.
* **`GIFU` (Gate In Full):** Loaded container enters the origin port.
* **`LOFU` (Loaded Full):** Container is hoisted and locked onto the cargo vessel.
* **`DIFU` (Discharged Full):** Container is unloaded from the vessel at the destination port.
* **`GOFU` (Gate Out Full):** Full container leaves destination port for customer delivery.
* **`GIMT` (Gate In Empty):** Empty container returned to destination depot.

**The Core Journey is defined as `LOFU` $\to$ `DIFU`**, isolating the maritime ocean leg where devices face RF signal isolation and severe temperature extremes.

---

## 3. Microsoft Fabric Architecture & Implementation

Below is the complete PySpark pipeline for execution in a **Microsoft Fabric Notebook** backed by a Lakehouse Delta table.

```
Lakehouse Source (`iot_telemetry`)
        │
        ▼
[Step 1: Ingestion, Banding & Ocean Leg Sessionization]
        │
        ▼
[Step 2: Rolling Window Daily Drain Rate (mV/Day)]
        │
        ├───────────────────────┬───────────────────────┬───────────────────────┐
        ▼                       ▼                       ▼                       ▼
[Step 3: Geo Aggs]     [Step 4: Seasonal Aggs] [Step 5: Voyage Diffs]  [Step 6: Live RUL Engine]
(`agg_battery_geo`)    (`agg_battery_season`) (`fact_voyage_drain`)   (`fact_device_live_rul`)
        │                       │                       │                       │
        └───────────────────────┴───────────┬───────────┴───────────────────────┘
                                            ▼
                                [Step 7: Power BI Model]
```

### Step 1: Ingestion, Voltage Banding, and Journey Sessionization

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1. Read from Fabric Lakehouse
# Replace table path with your Lakehouse delta table name
df_raw = spark.read.table("iot_telemetry")

# 2. Filter noise and apply voltage bands
df_base = (
    df_raw
    .filter(
        (F.col("battery_voltage").between(2.0, 4.5)) &
        (F.col("lat").between(-90.0, 90.0)) &
        (F.col("long").between(-180.0, 180.0)) &
        F.col("event_time").isNotNull()
    )
    .withColumn("event_time", F.to_timestamp("event_time"))
    .withColumn(
        "battery_band",
        F.when(F.col("battery_voltage") > 3.8, "1_Normal")
         .when((F.col("battery_voltage") <= 3.8) & (F.col("battery_voltage") > 3.5), "2_Backup")
         .when((F.col("battery_voltage") <= 3.5) & (F.col("battery_voltage") > 3.3), "3_Tertiary")
         .when((F.col("battery_voltage") <= 3.3) & (F.col("battery_voltage") >= 2.9), "4_Reserve")
         .otherwise("5_Hibernation")
    )
)

# 3. Sessionize Ocean Transit (LOFU -> DIFU) per asset_id
asset_order_window = Window.partitionBy("asset_id").orderBy("event_time")

df_journey = (
    df_base
    .withColumn("is_lofu", F.when(F.col("event_name") == "LOFU", 1).otherwise(0))
    .withColumn("is_difu", F.when(F.col("event_name") == "DIFU", 1).otherwise(0))
    .withColumn("lofu_count", F.sum("is_lofu").over(asset_order_window))
    .withColumn("difu_count", F.sum("is_difu").over(asset_order_window))
    .withColumn(
        "is_ocean_voyage",
        F.when((F.col("lofu_count") > F.col("difu_count")) | (F.col("event_name") == "DIFU"), True)
        .otherwise(False)
    )
    .drop("lofu_count", "difu_count", "is_lofu", "is_difu")
)
```

---

### Step 2: Compute Rolling Daily Voltage Drain Rate ($\text{mV}/\text{Day}$)

```python
# Use lagging window across individual device telemetry
device_time_window = Window.partitionBy("device_id").orderBy("event_time")

df_telemetry = (
    df_journey
    .withColumn("prev_event_time", F.lag("event_time", 1).over(device_time_window))
    .withColumn("prev_voltage", F.lag("battery_voltage", 1).over(device_time_window))
    .withColumn(
        "delta_hours", 
        (F.col("event_time").cast("long") - F.col("prev_event_time").cast("long")) / 3600.0
    )
    .withColumn(
        "delta_v", 
        F.col("prev_voltage") - F.col("battery_voltage")  # Positive = voltage consumed
    )
    # Require at least 2 hours between pings to suppress measurement noise; cap at 7 days
    .filter((F.col("delta_hours") >= 2.0) & (F.col("delta_hours") <= 168.0))
    .withColumn(
        "drain_rate_mv_per_day", 
        (F.col("delta_v") * 1000.0) / (F.col("delta_hours") / 24.0)
    )
)
```

---

### Step 3: Question 1 — Geographic Analysis (Lat/Long Bins)

```python
# Bin coordinates into 2° x 2° spatial buckets (~220 km tiles)
df_geo_analysis = (
    df_telemetry
    .withColumn("lat_bin", F.floor(F.col("lat") / 2.0) * 2.0)
    .withColumn("long_bin", F.floor(F.col("long") / 2.0) * 2.0)
    .groupBy("lat_bin", "long_bin", "is_ocean_voyage")
    .agg(
        F.countDistinct("device_id").alias("unique_devices"),
        F.count("event_time").alias("total_pings"),
        F.round(F.percentile_approx("battery_voltage", 0.5), 3).alias("median_voltage"),
        F.round(F.percentile_approx("drain_rate_mv_per_day", 0.5), 2).alias("median_drain_mv_day"),
        F.round(F.percentile_approx("drain_rate_mv_per_day", 0.9), 2).alias("p90_drain_mv_day"),
        # Band distribution percentages
        F.round(F.mean(F.when(F.col("battery_band") == "1_Normal", 1).otherwise(0)) * 100, 1).alias("pct_normal"),
        F.round(F.mean(F.when(F.col("battery_band") == "2_Backup", 1).otherwise(0)) * 100, 1).alias("pct_backup"),
        F.round(F.mean(F.when(F.col("battery_band").isin("3_Tertiary", "4_Reserve", "5_Hibernation"), 1).otherwise(0)) * 100, 1).alias("pct_critical_sub_3_5v")
    )
    .filter(F.col("unique_devices") >= 5)  # Suppress single-device noise
    .orderBy(F.desc("median_drain_mv_day"))
)

# Write to Fabric Lakehouse
df_geo_analysis.write.mode("overwrite").format("delta").saveAsTable("agg_battery_drain_by_geography")
```

---

### Step 4: Question 2 — Seasonal Analysis ($\text{MM-YY}$ & Hemisphere)

```python
df_seasonal_analysis = (
    df_telemetry
    .withColumn("year_month", F.date_format("event_time", "yyyy-MM"))
    .withColumn("month", F.month("event_time"))
    .withColumn("hemisphere", F.when(F.col("lat") >= 0, "Northern").otherwise("Southern"))
    .withColumn(
        "season",
        F.when(
            (F.col("month").isin(12, 1, 2) & (F.col("hemisphere") == "Northern")) |
            (F.col("month").isin(6, 7, 8) & (F.col("hemisphere") == "Southern")),
            "Winter"
        ).when(
            (F.col("month").isin(6, 7, 8) & (F.col("hemisphere") == "Northern")) |
            (F.col("month").isin(12, 1, 2) & (F.col("hemisphere") == "Southern")),
            "Summer"
        ).when(
            (F.col("month").isin(3, 4, 5) & (F.col("hemisphere") == "Northern")) |
            (F.col("month").isin(9, 10, 11) & (F.col("hemisphere") == "Southern")),
            "Spring"
        ).otherwise("Autumn")
    )
    .groupBy("year_month", "season", "hemisphere", "is_ocean_voyage")
    .agg(
        F.countDistinct("device_id").alias("unique_devices"),
        F.count("event_time").alias("total_pings"),
        F.round(F.percentile_approx("battery_voltage", 0.5), 3).alias("median_voltage"),
        F.round(F.percentile_approx("drain_rate_mv_per_day", 0.5), 2).alias("median_drain_mv_day"),
        F.round(F.percentile_approx("drain_rate_mv_per_day", 0.9), 2).alias("p90_drain_mv_day"),
        F.round(F.mean(F.when(F.col("battery_band") == "1_Normal", 1).otherwise(0)) * 100, 1).alias("pct_normal"),
        F.round(F.mean(F.when(F.col("battery_band").isin("3_Tertiary", "4_Reserve", "5_Hibernation"), 1).otherwise(0)) * 100, 1).alias("pct_critical")
    )
    .orderBy("year_month", "hemisphere", "is_ocean_voyage")
)

# Write to Fabric Lakehouse
df_seasonal_analysis.write.mode("overwrite").format("delta").saveAsTable("agg_battery_drain_by_season")
```

---

### Step 5: Core Journey Leg Analysis ($\text{LOFU} \to \text{DIFU}$)

This step measures total voltage loss between the loading event (`LOFU`) and discharge event (`DIFU`) for each maritime voyage.

```python
w_voyage = Window.partitionBy("asset_id", "device_id").orderBy("event_time")

df_voyage_summary = (
    df_base
    .filter(F.col("event_name").isin("LOFU", "DIFU"))
    .withColumn("prev_event_name", F.lag("event_name", 1).over(w_voyage))
    .withColumn("lofu_time", F.lag("event_time", 1).over(w_voyage))
    .withColumn("lofu_voltage", F.lag("battery_voltage", 1).over(w_voyage))
    .withColumn("lofu_lat", F.lag("lat", 1).over(w_voyage))
    .withColumn("lofu_long", F.lag("long", 1).over(w_voyage))
    # Keep completed LOFU -> DIFU segments
    .filter((F.col("event_name") == "DIFU") & (F.col("prev_event_name") == "LOFU"))
    .select(
        F.col("device_id"),
        F.col("asset_id"),
        F.col("lofu_time"),
        F.col("event_time").alias("difu_time"),
        F.round((F.col("event_time").cast("long") - F.col("lofu_time").cast("long")) / 86400.0, 1).alias("voyage_duration_days"),
        F.col("lofu_voltage"),
        F.col("battery_voltage").alias("difu_voltage"),
        F.round(F.col("lofu_voltage") - F.col("battery_voltage"), 3).alias("total_voltage_drop"),
        F.date_format("lofu_time", "yyyy-MM").alias("voyage_year_month"),
        F.col("lofu_lat"),
        F.col("lofu_long"),
        F.col("lat").alias("difu_lat"),
        F.col("long").alias("difu_long")
    )
    .withColumn(
        "daily_voyage_drain_mv",
        F.when(
            F.col("voyage_duration_days") > 0, 
            F.round((F.col("total_voltage_drop") * 1000.0) / F.col("voyage_duration_days"), 2)
        ).otherwise(0.0)
    )
)

# Write to Fabric Lakehouse
df_voyage_summary.write.mode("overwrite").format("delta").saveAsTable("fact_lofu_difu_voyage_drain")
```

---

### Step 6: Dynamic Non-Linear RUL Prediction Engine

This script outputs the real-time health and predicted days remaining before the device enters hibernation ($< 2.9\text{V}$). 

The model weights drain acceleration based on current operating band:
- **Normal ($> 3.8\text{V}$):** Linear capacity headroom down to $3.8\text{V}$ plus lower band buffers.
- **Backup ($3.5\text{V} - 3.8\text{V}$):** Normal drain to $3.5\text{V}$ plus lower band buffers.
- **Tertiary ($3.3\text{V} - 3.5\text{V}$):** Accelerated drain factor ($1.5\times$) down to $3.3\text{V}$ plus reserve buffer.
- **Reserve ($2.9\text{V} - 3.3\text{V}$):** Rapid cliff drop ($3.0\times$ acceleration) down to cutoff ($2.9\text{V}$).

```python
w_latest = Window.partitionBy("device_id").orderBy(F.desc("event_time"))

df_live_rul = (
    df_telemetry
    .withColumn("rn", F.row_number().over(w_latest))
    .filter(F.col("rn") == 1)
    .select(
        "device_id",
        "asset_id",
        "event_time",
        "lat",
        "long",
        "battery_voltage",
        "battery_band",
        "is_ocean_voyage",
        "drain_rate_mv_per_day"
    )
    # Apply baseline floor: 1.5 mV/day on land, 3.5 mV/day if at sea in signal dead zones
    .withColumn(
        "effective_drain_mv_day",
        F.when(F.col("drain_rate_mv_per_day") > 0.5, F.col("drain_rate_mv_per_day"))
         .when(F.col("is_ocean_voyage") == True, 3.5)
         .otherwise(1.5)
    )
    .withColumn(
        "estimated_days_to_hibernation",
        F.when(F.col("battery_band") == "5_Hibernation", 0.0)
         .when(
             # Reserve Band: 3.0x accelerated drain
             F.col("battery_band") == "4_Reserve",
             F.round(((F.col("battery_voltage") - 2.9) * 1000.0) / (F.col("effective_drain_mv_day") * 3.0), 1)
         )
         .when(
             # Tertiary Band: 1.5x drain to 3.3V + reserve buffer
             F.col("battery_band") == "3_Tertiary",
             F.round(
                 (((F.col("battery_voltage") - 3.3) * 1000.0) / (F.col("effective_drain_mv_day") * 1.5)) +
                 ((0.4 * 1000.0) / (F.col("effective_drain_mv_day") * 3.0)), 1
             )
         )
         .when(
             # Backup Band: 1.0x drain to 3.5V + tertiary buffer + reserve buffer
             F.col("battery_band") == "2_Backup",
             F.round(
                 (((F.col("battery_voltage") - 3.5) * 1000.0) / F.col("effective_drain_mv_day")) +
                 ((0.2 * 1000.0) / (F.col("effective_drain_mv_day") * 1.5)) +
                 ((0.4 * 1000.0) / (F.col("effective_drain_mv_day") * 3.0)), 1
             )
         )
         .otherwise(
             # Normal Band: full headroom to 3.8V + backup + tertiary + reserve buffers
             F.round(
                 (((F.col("battery_voltage") - 3.8) * 1000.0) / F.col("effective_drain_mv_day")) +
                 ((0.3 * 1000.0) / F.col("effective_drain_mv_day")) +
                 ((0.2 * 1000.0) / (F.col("effective_drain_mv_day") * 1.5)) +
                 ((0.4 * 1000.0) / (F.col("effective_drain_mv_day") * 3.0)), 1
             )
         )
    )
    .withColumn(
        "operational_risk_flag",
        F.when(F.col("estimated_days_to_hibernation") <= 30.0, "HIGH_RISK_REPLACE_SOON")
         .when(F.col("estimated_days_to_hibernation") <= 90.0, "MEDIUM_RISK_MONITOR")
         .otherwise("HEALTHY")
    )
)

# Write to Fabric Lakehouse
df_live_rul.write.mode("overwrite").format("delta").saveAsTable("fact_device_live_rul")
```

---

## 4. Power BI & Microsoft Fabric Semantic Model

To deliver insights to operational teams, connect the Delta tables inside Microsoft Fabric to a Direct Lake semantic model:

```
                      +---------------------------------------+
                      |       fact_device_live_rul            |
                      | (device_id, battery_voltage, RUL days)|
                      +---------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   ▼                                             ▼
+---------------------------------------+     +---------------------------------------+
|     agg_battery_drain_by_geography    |     |      fact_lofu_difu_voyage_drain      |
|  (lat_bin, long_bin, median_drain)    |     | (asset_id, duration_days, drop_v)     |
+---------------------------------------+     +---------------------------------------+
                   |
                   ▼
+---------------------------------------+
|      agg_battery_drain_by_season      |
| (year_month, season, pct_critical)    |
+---------------------------------------+
```

### Key DAX Measures for Dashboard Visuals:

```dax
// 1. Percentage of Fleet in Critical Band
% In Degraded Bands = 
DIVIDE(
    CALCULATE(COUNTROWS(fact_device_live_rul), fact_device_live_rul[battery_band] IN {"3_Tertiary", "4_Reserve", "5_Hibernation"}),
    COUNTROWS(fact_device_live_rul)
)

// 2. Average Ocean Transit Voltage Drop (mV)
Avg Voyage Drop mV = 
AVERAGEX(
    fact_lofu_difu_voyage_drain, 
    fact_lofu_difu_voyage_drain[total_voltage_drop] * 1000
)

// 3. Count of Devices with < 45 Days Remaining Life
Devices At Immediate Risk = 
CALCULATE(
    DISTINCTCOUNT(fact_device_live_rul[device_id]),
    fact_device_live_rul[estimated_days_to_hibernation] < 45
)
```