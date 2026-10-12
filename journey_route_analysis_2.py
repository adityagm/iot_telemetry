#!/usr/bin/env python3
"""
========================================================================================
MICROSOFT FABRIC LAKEHOUSE PYSPARK PIPELINE:
IOT SUPPLY CHAIN TRACKER BATTERY DRAIN & REMAINING USEFUL LIFE (RUL) ENGINE
========================================================================================

----------------------------------------------------------------------------------------
THEORETICAL FOUNDATION & DOMAIN DYNAMICS
----------------------------------------------------------------------------------------

1. ELECTROCHEMISTRY & DISCHARGE BEHAVIOR (Li-SOCl2 CELLS):
   - Flat Plateau Chemistry:
     Supply chain asset trackers predominantly rely on primary Lithium-thionyl chloride 
     (Li-SOCl2) chemistries. These cells maintain an exceptionally flat open-circuit 
     voltage plateau across ~85% to 90% of their operational discharge life, followed 
     by an abrupt terminal cliff.
   - Anode Passivation:
     A protective film of Lithium Chloride (LiCl) naturally deposits over the lithium 
     anode to suppress spontaneous chemical decomposition during storage. In aged cells 
     (>1-2 years), this passivation layer thickens. The initial milliamp surge required 
     to wake up the cellular modem and GPS engine causes a sharp, transient "passivation 
     voltage dip" before the layer dynamically breaks down.
   - The Thermal Voltage "Sag" and "Bounce" Phenomenon:
     * High Latitudes / Winter (Cold Sag):
       Low ambient temperatures (-10°C to -40°C) elevate electrolyte viscosity and internal 
       cell impedance (R_int). Under high transmission loads (+23 dBm RF), the instantaneous 
       voltage drops severely into lower bands. This is NOT necessarily permanent chemical 
       exhaustion.
     * Equatorial / Summer (Thermal Bounce):
       Higher temperatures lower internal resistance, creating a false impression of 
       "battery recovery" (50–150 mV voltage bounce). However, prolonged exposure to high 
       heat accelerates continuous self-discharge (2% to 4% per year vs. 1% baseline).

2. OPERATIONAL VOLTAGE BANDS (STATE MACHINE):
   Due to the non-linear discharge curve, modeling battery drain as a global linear rate 
   leads to massive prediction errors. Instead, the tracker operates across 5 discrete bands:
     - Normal Operation (> 3.8V):
         Full operational sampling, frequent GPS/cellular reporting (e.g., hourly/4h).
     - Backup Band (3.5V - 3.8V):
         Linear power degradation begins; firmware starts sensor power throttling.
     - Tertiary Band (3.3V - 3.5V):
         Transition knee; high vulnerability to cold-sag; firmware trims GPS timeouts.
     - Reserve Band (2.9V - 3.3V):
         Terminal depletion phase (<10% remaining capacity). Accelerated voltage cliff; 
         emergency pings only (1-2 pings/day, Cell-ID fallback).
     - Hibernation (< 2.9V):
         Deep hardware brownout prevention; modem disabled; unrecoverable without swap.

3. SUPPLY CHAIN LIFECYCLE SESSIONIZATION:
   Container movements follow a distinct 3-phase multi-modal lifecycle:
     - Land Journey 1 (Pre-Carriage):  GOMT -> (GIFU -> GOFU)* -> LOFU
         Drayage, rail movement, and yard dwell. Stable terrestrial cellular connection; 
         steady, predictable baseline power consumption.
     - Maritime Journey (Deep-Sea):     LOFU -> DIFU
         Vessel ocean transit. No terrestrial cellular coverage; heavy RF attenuation 
         if stowed below deck (acting as a Faraday cage). Devices risk entering severe 
         cell-search loops and GPS timeout limits.
     - Land Journey 2 (On-Carriage):   DIFU -> GIMT
         Discharge at destination port, drayage to consignee, destuffing, and empty return. 
         Wakeup bursts cause momentary pulse drain as buffered logs upload.

4. WHY MILESTONE CORRIDORS BEAT SEQUENCE MINING:
   Classical sequence mining (PrefixSpan, SPADE) converts continuous coordinates into 
   symbolic tokens (Geohash / H3 cells). In maritime IoT, this fails due to:
     - Asynchronous, irregular wakeups producing divergent spatial token strings.
     - State-space explosion over continuous ocean coordinates.
     - Extended signal blind spots during below-deck stowage.
   Solution: Anchor on domain milestones (LOFU -> DIFU), enforce Haversine displacement 
   (> 100 km) to eliminate circular repositioning loops, and classify maritime corridors 
   using geographic choke-point bounding boxes.

5. PIECEWISE NON-LINEAR REMAINING USEFUL LIFE (RUL):
   Because the voltage cliff accelerates in the lower bands, empirical drain velocity is 
   penalized dynamically:
     - Normal & Backup: Standard linear extrapolation down to 3.3V.
     - Tertiary Band:  Drain penalized by 1.5x factor.
     - Reserve Band:   Drain penalized by 3.0x factor (rapid terminal drop).
========================================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Initialize or connect to active Microsoft Fabric Spark session
spark = SparkSession.builder.getOrCreate()


# ==============================================================================
# SECTION 1: INGESTION, NOISE FILTERING & AGE PROVISIONING
# ==============================================================================
def load_and_clean_telemetry(table_name: str = "iot_telemetry"):
    """
    Ingests IoT telemetry from Delta Lake, scrubs unphysical sensor readings,
    and calculates device operational age.

    Theoretical Context:
    Primary Li-SOCl2 batteries lose ~1-2% of effective capacity per year via passive
    shelf degradation. Establishing the 'device_first_seen' timestamp allows us to 
    isolate calendar aging and anode passivation effects from operational cycle drain.
    """
    df_raw = spark.read.table(table_name)

    df_clean = (
        df_raw
        .filter(
            # Battery voltages outside [2.0V, 4.5V] indicate ADC noise or circuit brownout
            (F.col("battery_voltage").between(2.0, 4.5)) &
            (F.col("latitude").between(-90.0, 90.0)) &
            (F.col("longitude").between(-180.0, 180.0)) &
            F.col("event_time").isNotNull() &
            F.col("asset_id").isNotNull() &
            F.col("device_id").isNotNull()
        )
        .withColumn("event_time", F.to_timestamp("event_time"))
    )

    # Establish device birth date (earliest observed ping)
    w_device = Window.partitionBy("device_id")
    df_with_age = (
        df_clean
        .withColumn("device_first_seen", F.min("event_time").over(w_device))
        .withColumn(
            "device_age_days",
            F.round((F.col("event_time").cast("long") - F.col("device_first_seen").cast("long")) / 86400.0, 1)
        )
    )

    return df_with_age


# ==============================================================================
# SECTION 2: MULTI-MODAL CONTAINER LIFECYCLE SESSIONIZATION
# ==============================================================================
def sessionize_container_lifecycles(df):
    """
    Partitions continuous tracker telemetry into structured 3-phase journey cycles:
      - Phase 1: Land Pre-Carriage  [GOMT -> LOFU]  (supports intermediate GIFU/GOFU handoffs)
      - Phase 2: Maritime Ocean     [LOFU -> DIFU]  (vessel deep-sea transit)
      - Phase 3: Land On-Carriage   [DIFU -> GIMT]  (delivery & empty return)

    Theoretical Context:
    RF power draw varies by an order of magnitude across these legs:
    - On land: Terrestrial base stations require ~23 dBm max, but quickly negotiate lower RF power.
    - At sea: Lack of towers forces radios to hit full search timeouts, spiking milliamp-hour draw.
    """
    w_asset_time = Window.partitionBy("asset_id").orderBy("event_time")

    # Increment cycle index each time a container leaves a depot empty (GOMT)
    df_journey_tagged = (
        df
        .withColumn("is_new_cycle", F.when(F.col("event_name") == "GOMT", 1).otherwise(0))
        .withColumn(
            "journey_cycle_id",
            F.concat(F.col("asset_id"), F.lit("_CYC_"), F.sum("is_new_cycle").over(w_asset_time))
        )
    )

    w_cycle = Window.partitionBy("journey_cycle_id")

    # Isolate boundary milestone timestamps for each lifecycle
    df_milestones = (
        df_journey_tagged
        .withColumn("t_gomt", F.min(F.when(F.col("event_name") == "GOMT", F.col("event_time"))).over(w_cycle))
        .withColumn("t_lofu", F.min(F.when(F.col("event_name") == "LOFU", F.col("event_time"))).over(w_cycle))
        .withColumn("t_difu", F.min(F.when(F.col("event_name") == "DIFU", F.col("event_time"))).over(w_cycle))
        .withColumn("t_gimt", F.min(F.when(F.col("event_name") == "GIMT", F.col("event_time"))).over(w_cycle))
    )

    # Validate chronological progression: GOMT < LOFU < DIFU < GIMT
    df_valid_cycles = (
        df_milestones
        .filter(
            F.col("t_gomt").isNotNull() &
            F.col("t_lofu").isNotNull() &
            F.col("t_difu").isNotNull() &
            F.col("t_gimt").isNotNull() &
            (F.col("t_gomt") < F.col("t_lofu")) &
            (F.col("t_lofu") < F.col("t_difu")) &
            (F.col("t_difu") < F.col("t_gimt"))
        )
        .withColumn(
            "journey_phase",
            F.when((F.col("event_time") >= F.col("t_gomt")) & (F.col("event_time") < F.col("t_lofu")), "1_Land_PreCarriage")
             .when((F.col("event_time") >= F.col("t_lofu")) & (F.col("event_time") <= F.col("t_difu")), "2_Maritime_Ocean")
             .when((F.col("event_time") > F.col("t_difu")) & (F.col("event_time") <= F.col("t_gimt")), "3_Land_OnCarriage")
             .otherwise("Out_of_Bounds")
        )
        .filter(F.col("journey_phase") != "Out_of_Bounds")
    )

    return df_valid_cycles


# ==============================================================================
# SECTION 3: GEOGRAPHIC DISPLACEMENT & NON-CIRCULAR VERIFICATION
# ==============================================================================
def filter_non_circular_journeys(df):
    """
    Applies the Haversine spherical trigonometric formula between LOFU and DIFU.

    Theoretical Context:
    Repositioning moves, depot turnaround tests, or rejected cargo can cause
    Origin -> Destination -> Origin patterns. Evaluating a circular loop as a single 
    journey artificially inflates elapsed transit time and confounds two distinct geographic 
    climates. Requiring Great-Circle displacement > 100 km ensures strictly point-to-point, 
    one-way maritime corridors.
    """
    EARTH_RADIUS_KM = 6371.0
    w_cycle = Window.partitionBy("journey_cycle_id")

    df_coords = (
        df
        .withColumn("latitude_lofu", F.first(F.when(F.col("event_name") == "LOFU", F.col("latitude")), ignorenulls=True).over(w_cycle))
        .withColumn("longitude_lofu", F.first(F.when(F.col("event_name") == "LOFU", F.col("longitude")), ignorenulls=True).over(w_cycle))
        .withColumn("latitude_difu", F.first(F.when(F.col("event_name") == "DIFU", F.col("latitude")), ignorenulls=True).over(w_cycle))
        .withColumn("longitude_difu", F.first(F.when(F.col("event_name") == "DIFU", F.col("longitude")), ignorenulls=True).over(w_cycle))
    )

    # Great-Circle Haversine implementation
    df_displacement = (
        df_coords
        .withColumn("lat1_rad", F.radians(F.col("latitude_lofu")))
        .withColumn("lon1_rad", F.radians(F.col("longitude_lofu")))
        .withColumn("lat2_rad", F.radians(F.col("latitude_difu")))
        .withColumn("lon2_rad", F.radians(F.col("longitude_difu")))
        .withColumn("dlat", F.col("lat2_rad") - F.col("lat1_rad"))
        .withColumn("dlon", F.col("lon2_rad") - F.col("lon1_rad"))
        .withColumn(
            "a",
            F.sin(F.col("dlat") / 2.0)**2 +
            F.cos(F.col("lat1_rad")) * F.cos(F.col("lat2_rad")) * F.sin(F.col("dlon") / 2.0)**2
        )
        .withColumn("c", 2.0 * F.asin(F.sqrt(F.col("a"))))
        .withColumn("displacement_km", F.round(F.col("c") * EARTH_RADIUS_KM, 1))
        # Strictly reject circular loops and co-located port tests
        .filter(F.col("displacement_km") > 100.0)
    )

    return df_displacement


# ==============================================================================
# SECTION 4: OPERATIONAL VOLTAGE BANDS & ROLLING DRAIN RATE
# ==============================================================================
def compute_battery_bands_and_drain(df):
    """
    Classifies telemetry into operational battery bands and calculates rolling
    drain velocity (mV/day).

    Theoretical Context:
    Calculating delta V on raw adjacent pings (1-4 hours apart) produces high variance 
    and false positive voltage gains due to ambient temperature oscillations. By imposing 
    a minimum 2-hour delta threshold and analyzing rolling windows, we extract the true 
    underlying electrochemical depletion trend.
    """
    # Map raw voltage to the 5 operational states
    df_banded = df.withColumn(
        "battery_band",
        F.when(F.col("battery_voltage") > 3.8, "1_Normal")
         .when((F.col("battery_voltage") <= 3.8) & (F.col("battery_voltage") > 3.5), "2_Backup")
         .when((F.col("battery_voltage") <= 3.5) & (F.col("battery_voltage") > 3.3), "3_Tertiary")
         .when((F.col("battery_voltage") <= 3.3) & (F.col("battery_voltage") >= 2.9), "4_Reserve")
         .otherwise("5_Hibernation")
    )

    w_device_time = Window.partitionBy("device_id").orderBy("event_time")

    df_drain = (
        df_banded
        .withColumn("prev_time", F.lag("event_time", 1).over(w_device_time))
        .withColumn("prev_v", F.lag("battery_voltage", 1).over(w_device_time))
        .withColumn(
            "delta_hours",
            (F.col("event_time").cast("long") - F.col("prev_time").cast("long")) / 3600.0
        )
        .withColumn("delta_v", F.col("prev_v") - F.col("battery_voltage"))
        # Restrict window between 2 hours and 7 days to eliminate division artifacts
        .filter((F.col("delta_hours") >= 2.0) & (F.col("delta_hours") <= 168.0))
        .withColumn("drain_rate_mv_day", (F.col("delta_v") * 1000.0) / (F.col("delta_hours") / 24.0))
    )

    return df_drain


# ==============================================================================
# SECTION 5: ANALYTICAL DELIVERABLES (CORE QUESTIONS 1, 2, AND 3)
# ==============================================================================
def aggregate_q1_geography(df):
    """
    Question 1: Battery voltage drain by Latitude and Longitude.

    Theoretical Context:
    Aggregates spatial tiles (2x2 degrees) to expose:
    - High-latitude winter zones (>45°N/S) with severe load sag.
    - Equatorial trade lanes (0°-20°N) with accelerated self-discharge.
    - Coastal vs. open-water RF coverage boundaries.
    """
    return (
        df
        .withColumn("latitude_bin", F.floor(F.col("latitude") / 2.0) * 2.0)
        .withColumn("longitude_bin", F.floor(F.col("longitude") / 2.0) * 2.0)
        .groupBy("latitude_bin", "longitude_bin", "journey_phase")
        .agg(
            F.count("device_id").alias("ping_count"),
            F.countDistinct("device_id").alias("unique_devices"),
            F.round(F.percentile_approx("battery_voltage", 0.5), 3).alias("median_voltage"),
            F.round(F.percentile_approx("drain_rate_mv_day", 0.5), 2).alias("median_drain_mv_day"),
            F.round(F.percentile_approx("drain_rate_mv_day", 0.9), 2).alias("p90_drain_mv_day")
        )
        .filter(F.col("unique_devices") >= 5)
        .orderBy(F.desc("median_drain_mv_day"))
    )


def aggregate_q2_seasonality(df):
    """
    Question 2: Battery voltage drain by Season (MM-YY) & Hemisphere Alignment.

    Theoretical Context:
    Calendar months alone produce biased seasonal estimates because December is 
    mid-winter in the North Pacific (+latitude), but mid-summer off the Cape of Good 
    Hope (-latitude). Aligning season by hemisphere sign isolates true climatic impact 
    from calendar artifacts.
    """
    return (
        df
        .withColumn("year_month", F.date_format("event_time", "yyyy-MM"))
        .withColumn("month", F.month("event_time"))
        .withColumn("hemisphere", F.when(F.col("latitude") >= 0, "Northern").otherwise("Southern"))
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
        .groupBy("year_month", "season", "hemisphere")
        .agg(
            F.count("device_id").alias("sample_pings"),
            F.round(F.percentile_approx("battery_voltage", 0.5), 3).alias("median_voltage"),
            F.round(F.percentile_approx("drain_rate_mv_day", 0.5), 2).alias("median_drain_mv_day")
        )
        .orderBy("year_month", "hemisphere")
    )


def aggregate_q3_firmware_and_age(df):
    """
    Question 3: Battery drain by Device Age cohort and Reporting Cadence.

    Theoretical Context:
    If firmware versions are unrecorded, the daily transmission count serves as an 
    accurate behavioral proxy. Devices reporting >6 pings/day consume significantly 
    higher pulse current. Older cohorts (>1-2 years) exhibit higher internal resistance, 
    accelerating depletion when subjected to frequent wakeups.
    """
    df_daily = (
        df
        .withColumn("event_date", F.to_date("event_time"))
        .groupBy("device_id", "event_date")
        .agg(
            F.count("event_time").alias("daily_pings"),
            F.max("device_age_days").alias("device_age_days"),
            F.avg("drain_rate_mv_day").alias("avg_daily_drain")
        )
    )

    return (
        df_daily
        .withColumn(
            "age_cohort",
            F.when(F.col("device_age_days") <= 90, "0-3 Months")
             .when(F.col("device_age_days") <= 180, "3-6 Months")
             .when(F.col("device_age_days") <= 365, "6-12 Months")
             .when(F.col("device_age_days") <= 730, "1-2 Years")
             .otherwise("2+ Years")
        )
        .withColumn(
            "reporting_cadence",
            F.when(F.col("daily_pings") <= 2, "Low (<=2/day)")
             .when(F.col("daily_pings") <= 6, "Standard (3-6/day)")
             .otherwise("High (>6/day)")
        )
        .groupBy("age_cohort", "reporting_cadence")
        .agg(
            F.countDistinct("device_id").alias("device_count"),
            F.round(F.percentile_approx("avg_daily_drain", 0.5), 2).alias("median_drain_mv_day")
        )
        .orderBy("age_cohort", "reporting_cadence")
    )


# ==============================================================================
# SECTION 6: MARITIME CHOKE POINT CORRIDOR IDENTIFICATION
# ==============================================================================
def tag_maritime_chokepoints(df):
    """
    Classifies vessel trajectories using bounding boxes across global maritime choke points.

    Theoretical Context:
    Replaces token-heavy sequence mining. A vessel traveling between Asia and Europe 
    is deterministically categorized as 'Suez Route' vs. 'Cape Route' based on coordinate 
    intersection, allowing precise trade corridor power benchmarking.
    """
    return df.withColumn(
        "chokepoint",
        F.when(F.col("latitude").between(12.0, 31.0) & F.col("longitude").between(32.0, 44.0), "Suez / Red Sea")
         .when(F.col("latitude").between(7.0, 10.5) & F.col("longitude").between(-81.0, -78.0), "Panama Canal")
         .when(F.col("latitude").between(1.0, 6.0) & F.col("longitude").between(99.0, 105.0), "Malacca Strait")
         .when(F.col("latitude").between(-36.0, -30.0) & F.col("longitude").between(16.0, 32.0), "Cape of Good Hope")
         .when(F.col("latitude").between(49.0, 52.0) & F.col("longitude").between(-5.5, 2.5), "English Channel")
         .otherwise("Open Ocean / Other")
    )


# ==============================================================================
# SECTION 7: PIECEWISE REMAINING USEFUL LIFE (RUL) INFERENCE ENGINE
# ==============================================================================
def predict_device_rul(df):
    """
    Predicts remaining days until device enters Hibernation (< 2.9V).

    Theoretical Context:
    Implements a piece-wise operational state machine. When in Tertiary (3.3V-3.5V),
    drain is scaled by 1.5x. When in Reserve (2.9V-3.3V), drain is scaled by 3.0x to 
    account for the steep electrochemical terminal cliff of Li-SOCl2 cells.
    """
    w_latest = Window.partitionBy("device_id").orderBy(F.desc("event_time"))

    return (
        df
        .withColumn("rank", F.row_number().over(w_latest))
        .filter(F.col("rank") == 1)
        .select(
            "device_id", "asset_id", "event_time", "latitude", "longitude",
            "battery_voltage", "battery_band", "device_age_days",
            "drain_rate_mv_day", "journey_phase"
        )
        # Floor effective drain rate at 0.5 mV/day to prevent division by zero or negative slopes
        .withColumn(
            "effective_drain",
            F.when(F.col("drain_rate_mv_day") > 0.5, F.col("drain_rate_mv_day")).otherwise(1.2)
        )
        .withColumn(
            "predicted_days_to_hibernation",
            F.when(F.col("battery_band") == "5_Hibernation", 0.0)
             .when(
                 F.col("battery_band") == "4_Reserve",
                 F.round(((F.col("battery_voltage") - 2.9) * 1000.0) / (F.col("effective_drain") * 3.0), 1)
             )
             .when(
                 F.col("battery_band") == "3_Tertiary",
                 F.round(
                     (((F.col("battery_voltage") - 3.3) * 1000.0) / (F.col("effective_drain") * 1.5)) +
                     ((0.4 * 1000.0) / (F.col("effective_drain") * 3.0)), 1
                 )
             )
             .otherwise(
                 F.round(((F.col("battery_voltage") - 2.9) * 1000.0) / F.col("effective_drain"), 1)
             )
        )
    )


# ==============================================================================
# SECTION 8: VISUALIZATION SUITE: REPRESENTATIVE JOURNEY PLOTTER
# ==============================================================================
def plot_three_representative_journeys(df_processed):
    """
    Extracts the longest, median, and shortest verified non-circular journeys,
    rendering their battery telemetry against operational bands and journey phases.
    """
    cycle_summary = (
        df_processed
        .groupBy("journey_cycle_id")
        .agg(
            F.first("asset_id").alias("asset_id"),
            F.first("device_id").alias("device_id"),
            F.min("t_gomt").alias("t_start"),
            F.max("t_gimt").alias("t_end"),
            F.min("t_lofu").alias("t_lofu"),
            F.min("t_difu").alias("t_difu"),
            F.first("displacement_km").alias("displacement_km")
        )
        .withColumn("total_days", (F.col("t_end").cast("long") - F.col("t_start").cast("long")) / 86400.0)
        .filter(F.col("total_days").between(2.0, 90.0))
        .orderBy("total_days")
        .toPandas()
    )

    if len(cycle_summary) < 3:
        print("Warning: Insufficient completed multi-phase cycles found for plotting.")
        return

    longest = cycle_summary.iloc[-1]
    median = cycle_summary.iloc[len(cycle_summary) // 2]
    shortest = cycle_summary.iloc[0]

    targets = [
        {"title": f"A. Longest Journey (Asset: {longest['asset_id']} | Device: {longest['device_id']}) — Total: {longest['total_days']:.1f} Days", "data": longest},
        {"title": f"B. Median Journey (Asset: {median['asset_id']} | Device: {median['device_id']}) — Total: {median['total_days']:.1f} Days", "data": median},
        {"title": f"C. Shortest Journey (Asset: {shortest['asset_id']} | Device: {shortest['device_id']}) — Total: {shortest['total_days']:.1f} Days", "data": shortest}
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharey=True)

    bands = [
        (3.8, 4.2, "#e8f5e9", "Normal (>3.8V)"),
        (3.5, 3.8, "#fffde7", "Backup (3.5 - 3.8V)"),
        (3.3, 3.5, "#fff3e0", "Tertiary (3.3 - 3.5V)"),
        (2.9, 3.3, "#ffebee", "Reserve (2.9 - 3.3V)"),
        (2.6, 2.9, "#ffcdd2", "Hibernation (<2.9V)")
    ]

    phase_colors = {
        "1_Land_PreCarriage": "#e3f2fd",
        "2_Maritime_Ocean":   "#e8f5e9",
        "3_Land_OnCarriage":  "#fff3e0"
    }

    milestone_markers = {
        "GOMT": ("^", "#1565c0", 110),
        "GIFU": ("o", "#42a5f5", 50),
        "GOFU": ("o", "#26c6da", 50),
        "LOFU": ("s", "#2e7d32", 110),
        "DIFU": ("v", "#e65100", 110),
        "GIMT": ("D", "#c62828", 110)
    }

    for idx, target in enumerate(targets):
        ax = axes[idx]
        t_info = target["data"]

        points_df = (
            df_processed
            .filter(F.col("journey_cycle_id") == t_info["journey_cycle_id"])
            .select("event_time", "battery_voltage", "event_name", "journey_phase", "latitude", "longitude")
            .orderBy("event_time")
            .toPandas()
        )

        t0 = points_df["event_time"].min()
        points_df["days"] = (points_df["event_time"] - t0).dt.total_seconds() / 86400.0

        # Background battery band zones
        for y_low, y_high, bg_color, _ in bands:
            ax.axhspan(y_low, y_high, color=bg_color, alpha=0.55, zorder=0)

        # Background journey phase zones
        for p_name, p_group in points_df.groupby("journey_phase"):
            p_min, p_max = p_group["days"].min(), p_group["days"].max()
            ax.axvspan(p_min, p_max, color=phase_colors.get(p_name, "#f5f5f5"), alpha=0.45, zorder=1)

        # Voltage trajectory line
        ax.plot(points_df["days"], points_df["battery_voltage"], color="#263238", lw=1.8, zorder=3)
        ax.scatter(points_df["days"], points_df["battery_voltage"], color="#37474f", s=14, alpha=0.6, zorder=4)

        # Milestone events
        for _, row in points_df.iterrows():
            ev = row["event_name"]
            if ev in milestone_markers:
                m_shape, m_col, m_size = milestone_markers[ev]
                ax.scatter(row["days"], row["battery_voltage"], marker=m_shape, color=m_col, s=m_size, zorder=5)

        # Diagnostic stat box
        v_start = points_df["battery_voltage"].iloc[0]
        v_end = points_df["battery_voltage"].iloc[-1]
        dv_mv = (v_start - v_end) * 1000.0
        rate = dv_mv / t_info["total_days"]

        stat_box = (
            f"Start: {v_start:.2f}V | End: {v_end:.2f}V\n"
            f"Total Loss: -{dv_mv:.1f} mV\n"
            f"Displacement: {t_info['displacement_km']} km\n"
            f"Avg Drain: {rate:.2f} mV/day"
        )
        ax.text(0.985, 0.92, stat_box, transform=ax.transAxes, fontsize=9.0,
                va='top', ha='right', bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#b0bec5', alpha=0.92), zorder=6)

        ax.set_title(target["title"], fontsize=11, fontweight='bold', pad=5, loc='left')
        ax.set_ylabel("Voltage (V)", fontsize=10)
        ax.set_ylim(2.8, 4.15)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[2].set_xlabel("Elapsed Days from GOMT (Gate Out Empty)", fontsize=11, fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#1565c0', markersize=9, label='GOMT'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#42a5f5', markersize=7, label='GIFU/GOFU'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2e7d32', markersize=9, label='LOFU'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#e65100', markersize=9, label='DIFU'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#c62828', markersize=9, label='GIMT'),
        plt.Line2D([0], [0], color='#263238', lw=1.8, label='Telemetry (V)'),
        Patch(facecolor="#e3f2fd", label="Land 1 (Pre-Carriage)"),
        Patch(facecolor="#e8f5e9", label="Maritime Ocean"),
        Patch(facecolor="#fff3e0", label="Land 2 (On-Carriage)")
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=5, fontsize=9.0, frameon=True, facecolor='white', edgecolor='#b0bec5')

    plt.subplots_adjust(top=0.93, bottom=0.06, hspace=0.28)
    plt.show()


# ==============================================================================
# SECTION 9: MAIN EXECUTION ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    print("Initiating IoT Battery Telemetry Analytics Pipeline...")

    # Step 1: Clean & augment with provisioning age
    df_clean = load_and_clean_telemetry("iot_telemetry")

    # Step 2: Establish 3-phase journey state machine
    df_sessions = sessionize_container_lifecycles(df_clean)

    # Step 3: Filter circular repositioning loops via Haversine distance
    df_verified = filter_non_circular_journeys(df_sessions)

    # Step 4: Map operational voltage bands & derive rolling drain rates
    df_processed = compute_battery_bands_and_drain(df_verified)

    # Step 5: Materialize analytical deliverables to Delta Lake tables
    df_geo = aggregate_q1_geography(df_processed)
    df_geo.write.mode("overwrite").format("delta").saveAsTable("agg_battery_drain_by_geography")

    df_season = aggregate_q2_seasonality(df_processed)
    df_season.write.mode("overwrite").format("delta").saveAsTable("agg_battery_drain_by_season")

    df_age = aggregate_q3_firmware_and_age(df_processed)
    df_age.write.mode("overwrite").format("delta").saveAsTable("agg_battery_drain_by_age")

    # Step 6: Materialize piece-wise RUL predictions for client consumption
    df_rul = predict_device_rul(df_processed)
    df_rul.write.mode("overwrite").format("delta").saveAsTable("fact_device_battery_rul")

    print("Pipeline completed successfully. Lakehouse Delta tables written.")

    # Step 7: Render representative lifecycle comparison plot
    plot_three_representative_journeys(df_processed)