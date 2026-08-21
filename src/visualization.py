"""
src/visualization.py
--------------------
FlightIQ — AI Travel Price Intelligence
Reusable EDA visualization functions.
Saves all figures to assets/eda/ as high-quality PNGs.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts/notebooks
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_flight_data.csv"
EDA_DIR        = Path(__file__).resolve().parents[1] / "assets" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE_MAIN   = "Blues_r"
ACCENT         = "#2563EB"          # strong blue
ACCENT2        = "#16A34A"          # green accent
NEUTRAL        = "#64748B"
BG             = "#FAFAFA"
TITLE_SIZE     = 15
LABEL_SIZE     = 11
TICK_SIZE      = 10
DPI            = 180

def _style():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.color":       "#E2E8F0",
        "grid.linewidth":   0.7,
        "font.family":      "DejaVu Sans",
        "axes.titlesize":   TITLE_SIZE,
        "axes.labelsize":   LABEL_SIZE,
        "xtick.labelsize":  TICK_SIZE,
        "ytick.labelsize":  TICK_SIZE,
    })

def _save(fig, filename: str):
    path = EDA_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path.name}")
    return path

def _fmt_k(x, _):
    if x >= 1_000_000: return f"₹{x/1_000_000:.1f}M"
    if x >= 1_000:     return f"₹{x/1_000:.0f}K"
    return f"₹{x:.0f}"


# ── 1. Price Distribution ────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Flight Price Distribution", fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.02)

    # Full distribution
    ax = axes[0]
    ax.hist(df["Price"], bins=80, color=ACCENT, edgecolor="white", linewidth=0.4, alpha=0.85)
    for pct, val in [(25, df["Price"].quantile(0.25)),
                     (50, df["Price"].median()),
                     (75, df["Price"].quantile(0.75))]:
        ax.axvline(val, color="#DC2626", linestyle="--", linewidth=1.2,
                   label=f"P{pct}: ₹{val:,.0f}")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Price (INR)")
    ax.set_ylabel("Number of Flights")
    ax.set_title("Full Price Distribution")
    ax.legend(fontsize=9)

    # Log scale
    ax2 = axes[1]
    ax2.hist(np.log10(df["Price"] + 1), bins=80, color=ACCENT2,
             edgecolor="white", linewidth=0.4, alpha=0.85)
    ax2.set_xlabel("log₁₀(Price)")
    ax2.set_ylabel("Number of Flights")
    ax2.set_title("Price Distribution (Log Scale)")

    # Stats box
    stats = (f"Mean:   ₹{df['Price'].mean():>10,.0f}\n"
             f"Median: ₹{df['Price'].median():>10,.0f}\n"
             f"Std:    ₹{df['Price'].std():>10,.0f}\n"
             f"Min:    ₹{df['Price'].min():>10,.0f}\n"
             f"Max:    ₹{df['Price'].max():>10,.0f}")
    ax2.text(0.97, 0.97, stats, transform=ax2.transAxes, fontsize=8.5,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.tight_layout()
    return _save(fig, "01_price_distribution.png")


# ── 2. Airline vs Price ──────────────────────────────────────────────────────
def plot_airline_vs_price(df: pd.DataFrame):
    _style()
    stats = (df.groupby("Airline")["Price"]
               .agg(median="median", q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75), count="count")
               .sort_values("median", ascending=True))

    fig, ax = plt.subplots(figsize=(13, 6))
    y = range(len(stats))
    bars = ax.barh(list(stats.index), stats["median"], color=ACCENT, alpha=0.85,
                   height=0.6, label="Median Price")
    # IQR whiskers
    for i, (_, row) in enumerate(stats.iterrows()):
        ax.plot([row["q25"], row["q75"]], [i, i], color="#1E3A5F",
                linewidth=3, solid_capstyle="round", zorder=3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Median Price (INR)  |  bar = median,  line = IQR")
    ax.set_ylabel("Airline")
    ax.set_title("Flight Price by Airline\n(Median with IQR Range)", fontweight="bold")
    # Count annotations
    for i, (_, row) in enumerate(stats.iterrows()):
        ax.text(row["q75"] + 500, i, f'n={row["count"]:,}',
                va="center", fontsize=8, color=NEUTRAL)
    plt.tight_layout()
    return _save(fig, "02_airline_vs_price.png")


# ── 3. Travel Class vs Price ─────────────────────────────────────────────────
def plot_travel_class_vs_price(df: pd.DataFrame):
    _style()
    order = ["Economy", "Premium Economy", "Business", "First"]
    order = [c for c in order if c in df["Travel_Class"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price by Travel Class", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Box plot
    ax = axes[0]
    data_per_class = [df[df["Travel_Class"] == c]["Price"].values for c in order]
    bp = ax.boxplot(data_per_class, patch_artist=True, notch=False,
                    medianprops=dict(color="#DC2626", linewidth=2))
    colors = [ACCENT, "#3B82F6", "#1D4ED8", "#1E3A5F"]
    for patch, color in zip(bp["boxes"], colors[:len(order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(order, rotation=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_ylabel("Price (INR)")
    ax.set_title("Price Distribution (Box Plot)")

    # Median bar
    ax2 = axes[1]
    medians = [df[df["Travel_Class"] == c]["Price"].median() for c in order]
    bars = ax2.bar(order, medians, color=colors[:len(order)], alpha=0.85, width=0.5)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_ylabel("Median Price (INR)")
    ax2.set_title("Median Price per Class")
    ax2.set_xticklabels(order, rotation=15)
    for bar, med in zip(bars, medians):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                 f"₹{med:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "03_travel_class_vs_price.png")


# ── 4. Total Stops vs Price ──────────────────────────────────────────────────
def plot_stops_vs_price(df: pd.DataFrame):
    _style()
    stop_labels = {0.0: "Non-Stop", 1.0: "1 Stop", 2.0: "2 Stops"}
    df2 = df.copy()
    df2["Stops_Label"] = df2["Total_Stops_Numeric"].map(stop_labels)

    order = ["Non-Stop", "1 Stop", "2 Stops"]
    order = [o for o in order if o in df2["Stops_Label"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Flight Price by Number of Stops", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Violin
    ax = axes[0]
    parts = ax.violinplot([df2[df2["Stops_Label"] == o]["Price"].values for o in order],
                          positions=range(len(order)), showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([ACCENT2, ACCENT, "#1E3A5F"][i])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("#DC2626")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_ylabel("Price (INR)")
    ax.set_title("Price Distribution (Violin)")

    # Median + count bar
    ax2 = axes[1]
    medians = [df2[df2["Stops_Label"] == o]["Price"].median() for o in order]
    counts  = [df2["Stops_Label"].value_counts().get(o, 0) for o in order]
    clrs = [ACCENT2, ACCENT, "#1E3A5F"]
    bars = ax2.bar(order, medians, color=clrs[:len(order)], alpha=0.85, width=0.45)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_ylabel("Median Price (INR)")
    ax2.set_title("Median Price per Stop Category")
    for bar, med, cnt in zip(bars, medians, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f"₹{med:,.0f}\nn={cnt:,}", ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    return _save(fig, "04_stops_vs_price.png")


# ── 5. Source & Destination vs Price ─────────────────────────────────────────
def plot_source_dest_vs_price(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Median Flight Price by Origin & Destination", fontsize=TITLE_SIZE + 1, fontweight="bold")

    for ax, col, title in zip(axes, ["Source", "Destination"],
                               ["Source City (Origin)", "Destination City"]):
        stats = (df.groupby(col)["Price"].median()
                   .sort_values(ascending=True))
        colors_bar = [ACCENT if v >= stats.median() else "#93C5FD" for v in stats.values]
        ax.barh(stats.index, stats.values, color=colors_bar, alpha=0.88, height=0.65)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
        ax.set_xlabel("Median Price (INR)")
        ax.set_title(title)
        ax.axvline(stats.median(), color="#DC2626", linestyle="--",
                   linewidth=1.2, label=f"Overall median: ₹{stats.median():,.0f}")
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    return _save(fig, "05_source_dest_vs_price.png")


# ── 6. Duration vs Price ─────────────────────────────────────────────────────
def plot_duration_vs_price(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Flight Duration vs Price", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Hexbin scatter (handles overplotting with 93k rows)
    ax = axes[0]
    hb = ax.hexbin(df["Duration_Minutes"] / 60, df["Price"],
                   gridsize=45, cmap="Blues", mincnt=1, bins="log")
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("log₁₀(count)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Duration (Hours)")
    ax.set_ylabel("Price (INR)")
    ax.set_title(f"Duration vs Price  (r = {df['Duration_Minutes'].corr(df['Price']):.3f})")

    # Binned median
    ax2 = axes[1]
    df2 = df.copy()
    df2["Duration_Bin"] = pd.cut(df2["Duration_Minutes"] / 60,
                                 bins=[0, 2, 4, 6, 10, 15, 28],
                                 labels=["0-2h", "2-4h", "4-6h", "6-10h", "10-15h", "15h+"])
    med = df2.groupby("Duration_Bin", observed=True)["Price"].median()
    cnt = df2.groupby("Duration_Bin", observed=True)["Price"].count()
    bars = ax2.bar(med.index.astype(str), med.values, color=ACCENT, alpha=0.85, width=0.6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_xlabel("Duration Bucket")
    ax2.set_ylabel("Median Price (INR)")
    ax2.set_title("Median Price by Duration Bucket")
    for bar, c in zip(bars, cnt.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 500, f"n={c:,}",
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return _save(fig, "06_duration_vs_price.png")


# ── 7. Departure & Arrival Time vs Price ─────────────────────────────────────
def plot_time_vs_price(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Departure & Arrival Time vs Price", fontsize=TITLE_SIZE + 1, fontweight="bold")

    df2 = df.copy()
    df2["Dep_Hour"] = (df2["Departure_Time_Minutes"] / 60).astype(int) % 24
    df2["Arr_Hour"] = (df2["Arrival_Time_Minutes"]   / 60).astype(int) % 24

    period_bins   = [-1, 5, 11, 16, 20, 24]
    period_labels = ["Night\n(0–5)", "Morning\n(6–11)", "Afternoon\n(12–16)",
                     "Evening\n(17–20)", "Night\n(21–23)"]

    for row_idx, (col_minutes, col_hour, label) in enumerate([
        ("Departure_Time_Minutes", "Dep_Hour", "Departure"),
        ("Arrival_Time_Minutes",   "Arr_Hour", "Arrival"),
    ]):
        # Hourly median line
        ax = axes[row_idx][0]
        hourly = df2.groupby(col_hour)["Price"].median()
        ax.plot(hourly.index, hourly.values, color=ACCENT, linewidth=2, marker="o",
                markersize=4)
        ax.fill_between(hourly.index, hourly.values, alpha=0.15, color=ACCENT)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
        ax.set_xlabel(f"{label} Hour (0–23)")
        ax.set_ylabel("Median Price (INR)")
        ax.set_title(f"Median Price by {label} Hour")
        ax.set_xticks(range(0, 24, 2))

        # Period bar
        ax2 = axes[row_idx][1]
        df2["Period"] = pd.cut(df2[col_hour], bins=period_bins, labels=period_labels)
        med_period = df2.groupby("Period", observed=True)["Price"].median()
        bars = ax2.bar(med_period.index.astype(str), med_period.values,
                       color=ACCENT, alpha=0.85, width=0.55)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
        ax2.set_xlabel(f"{label} Period")
        ax2.set_ylabel("Median Price (INR)")
        ax2.set_title(f"Median Price by {label} Time Period")
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                     f"₹{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    return _save(fig, "07_time_vs_price.png")


# ── 8. Top Routes vs Price ────────────────────────────────────────────────────
def plot_routes_vs_price(df: pd.DataFrame):
    _style()
    df2 = df.copy()
    df2["Route"] = df2["Source"] + " → " + df2["Destination"]
    route_stats = (df2.groupby("Route")["Price"]
                     .agg(median="median", count="count")
                     .query("count >= 150")
                     .sort_values("median", ascending=True)
                     .tail(20))            # top 20 most expensive routes

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_bar = [ACCENT if v >= route_stats["median"].median() else "#93C5FD"
                  for v in route_stats["median"].values]
    bars = ax.barh(route_stats.index, route_stats["median"],
                   color=colors_bar, alpha=0.87, height=0.65)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Median Price (INR)")
    ax.set_title("Top 20 Costliest Routes\n(Routes with ≥150 flights, sorted by median price)",
                 fontweight="bold")
    for bar, cnt in zip(bars, route_stats["count"].values):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f"n={cnt:,}", va="center", fontsize=7.5, color=NEUTRAL)
    ax.axvline(df["Price"].median(), color="#DC2626", linestyle="--",
               linewidth=1.2, label=f"Overall median: ₹{df['Price'].median():,.0f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save(fig, "08_routes_vs_price.png")


# ── 9. Correlation Heatmap ───────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    _style()
    num_cols = ["Price", "Duration_Minutes", "Distance_km", "Total_Stops_Numeric",
                "Days_Before_Departure", "Departure_Time_Minutes",
                "Arrival_Time_Minutes", "Passenger_Count",
                "Departure_Month", "Weekday_Num"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)     # upper triangle
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#E2E8F0",
                annot_kws={"size": 9}, ax=ax)
    ax.set_title("Correlation Heatmap — Numerical Features vs Price",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return _save(fig, "09_correlation_heatmap.png")


# ── 10. Price by Month & Weekday ─────────────────────────────────────────────
def plot_price_by_calendar(df: pd.DataFrame):
    _style()
    month_map = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
                 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    day_map   = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Price Patterns: Calendar Effects", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Monthly
    ax = axes[0]
    monthly = df.groupby("Departure_Month")["Price"].median().reset_index()
    monthly["Month_Name"] = monthly["Departure_Month"].map(month_map)
    ax.bar(monthly["Month_Name"], monthly["Price"], color=ACCENT, alpha=0.85, width=0.6)
    ax.plot(monthly["Month_Name"], monthly["Price"], color="#DC2626",
            linewidth=2, marker="D", markersize=5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Month")
    ax.set_ylabel("Median Price (INR)")
    ax.set_title("Median Price by Departure Month")
    ax.set_xticklabels(monthly["Month_Name"], rotation=30)

    # Weekday
    ax2 = axes[1]
    weekday = df.groupby("Weekday_Num")["Price"].median().reset_index()
    weekday["Day_Name"] = weekday["Weekday_Num"].map(day_map)
    wknd_colors = ["#DC2626" if d in ["Sat", "Sun"] else ACCENT
                   for d in weekday["Day_Name"]]
    bars = ax2.bar(weekday["Day_Name"], weekday["Price"],
                   color=wknd_colors, alpha=0.85, width=0.55)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_xlabel("Day of Week  (red = weekend)")
    ax2.set_ylabel("Median Price (INR)")
    ax2.set_title("Median Price by Weekday")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                 f"₹{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return _save(fig, "10_price_by_calendar.png")


# ── 11. Season & Booking Channel ─────────────────────────────────────────────
def plot_season_channel_vs_price(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price by Season & Booking Channel", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Season
    ax = axes[0]
    season_order = ["Summer", "Winter", "Autumn", "Monsoon"]
    season_order = [s for s in season_order if s in df["Season"].unique()]
    season_med = df.groupby("Season")["Price"].median().reindex(season_order)
    season_cnt = df.groupby("Season")["Price"].count().reindex(season_order)
    bars = ax.bar(season_order, season_med.values, color=["#F59E0B","#6366F1","#10B981","#3B82F6"],
                  alpha=0.85, width=0.55)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_ylabel("Median Price (INR)")
    ax.set_title("Median Price by Season")
    for bar, cnt in zip(bars, season_cnt.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                f"₹{bar.get_height():,.0f}\nn={cnt:,}", ha="center", va="bottom", fontsize=8.5)

    # Booking Channel
    ax2 = axes[1]
    chan_med = df.groupby("Booking_Channel")["Price"].median().sort_values(ascending=True)
    ax2.barh(chan_med.index, chan_med.values, color=ACCENT, alpha=0.85, height=0.55)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_xlabel("Median Price (INR)")
    ax2.set_title("Median Price by Booking Channel")
    plt.tight_layout()
    return _save(fig, "11_season_channel_vs_price.png")


# ── 12. Days Before Departure vs Price ──────────────────────────────────────
def plot_days_before_departure(df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Booking Lead Time vs Price", fontsize=TITLE_SIZE + 1, fontweight="bold")

    # Scatter hexbin
    ax = axes[0]
    hb = ax.hexbin(df["Days_Before_Departure"], df["Price"],
                   gridsize=40, cmap="Blues", mincnt=1, bins="log")
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("log₁₀(count)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax.set_xlabel("Days Before Departure")
    ax.set_ylabel("Price (INR)")
    ax.set_title(f"Lead Time vs Price  (r = {df['Days_Before_Departure'].corr(df['Price']):.3f})")

    # Binned
    ax2 = axes[1]
    df2 = df.copy()
    df2["Lead_Bin"] = pd.cut(df2["Days_Before_Departure"],
                             bins=[-1, 7, 14, 30, 60, 90, 180],
                             labels=["0-7d", "8-14d", "15-30d", "31-60d", "61-90d", "91-180d"])
    med_lead = df2.groupby("Lead_Bin", observed=True)["Price"].median()
    cnt_lead = df2.groupby("Lead_Bin", observed=True)["Price"].count()
    bars = ax2.bar(med_lead.index.astype(str), med_lead.values,
                   color=ACCENT, alpha=0.85, width=0.6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
    ax2.set_xlabel("Days Before Departure")
    ax2.set_ylabel("Median Price (INR)")
    ax2.set_title("Median Price by Booking Lead Time")
    for bar, cnt in zip(bars, cnt_lead.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                 f"n={cnt:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return _save(fig, "12_days_before_departure.png")


# ── Master runner ─────────────────────────────────────────────────────────────
def run_all_eda(df: pd.DataFrame = None) -> list:
    if df is None:
        df = pd.read_csv(PROCESSED_PATH)
    saved = []
    print("Running EDA visualizations...")
    saved.append(plot_price_distribution(df))
    saved.append(plot_airline_vs_price(df))
    saved.append(plot_travel_class_vs_price(df))
    saved.append(plot_stops_vs_price(df))
    saved.append(plot_source_dest_vs_price(df))
    saved.append(plot_duration_vs_price(df))
    saved.append(plot_time_vs_price(df))
    saved.append(plot_routes_vs_price(df))
    saved.append(plot_correlation_heatmap(df))
    saved.append(plot_price_by_calendar(df))
    saved.append(plot_season_channel_vs_price(df))
    saved.append(plot_days_before_departure(df))
    print(f"\nDone. {len(saved)} visualizations saved to assets/eda/")
    return saved


if __name__ == "__main__":
    run_all_eda()
