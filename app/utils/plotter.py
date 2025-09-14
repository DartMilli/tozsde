import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from io import BytesIO
import warnings
import base64

import app.utils.router as rtr

warnings.filterwarnings("ignore", category=UserWarning)


def get_candle_img_buffer(df, indicators, signals=None):
    """
    https://stackoverflow.com/questions/60599812/how-can-i-customize-mplfinance-plot
    https://github.com/matplotlib/mplfinance/blob/master/src/mplfinance/plotting.py#L87-L276
    """
    buf = BytesIO()

    dark = {
        "base_mpl_style": "dark_background",
        "marketcolors": {
            "candle": {"up": "#3dc985", "down": "#ef4f60"},
            "edge": {"up": "#3dc985", "down": "#ef4f60"},
            "wick": {"up": "#3dc985", "down": "#ef4f60"},
            "ohlc": {"up": "green", "down": "red"},
            "volume": {"up": "#247252", "down": "#82333f"},
            "vcedge": {"up": "green", "down": "red"},
            "vcdopcod": False,
            "alpha": 1,
        },
        "mavcolors": ("#a63ab2", "#62b8ba"),
        "facecolor": "#121212",
        "gridcolor": "#AFAFAF",
        "gridstyle": "dotted",
        "y_on_right": False,
        "rc": {
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.edgecolor": "#AFAFAF",
            "axes.titlecolor": "red",
            "figure.facecolor": "#121212",
            "figure.titlesize": "xx-large",
            "figure.titleweight": "semibold",
            "ytick.left": True,
            "ytick.right": False,
        },
        "base_mpf_style": "binance-dark",
    }

    rsi_overbought = np.full(len(df), 70)
    rsi_oversold = np.full(len(df), 30)
    stoch_overbought = np.full(len(df), 80)
    stoch_oversold = np.full(len(df), 20)
    buy_signals = np.full(len(df), np.nan)
    sell_signals = np.full(len(df), np.nan)

    if signals:
        signal_dates = [s.split(" on ")[1] for s in signals if " on " in s]
        signal_types = ["BUY" if "BUY" in s else "SELL" for s in signals if " on " in s]

        for i, (sig_type, sig_date_str) in enumerate(zip(signal_types, signal_dates)):
            try:
                sig_date = pd.to_datetime(sig_date_str).date()
                if sig_date in df.index.date:
                    price = df.loc[
                        df.index.date == sig_date,
                        "Low" if sig_type == "BUY" else "High",
                    ].iloc[0]
                    idx = df.index.get_loc(df.index[df.index.date == sig_date][0])

                    if sig_type == "BUY":
                        buy_signals[idx] = price * 0.98  # Kicsit a gyertya alá
                    else:
                        sell_signals[idx] = price * 1.02  # Kicsit a gyertya fölé
            except (ValueError, IndexError) as e:
                print(f"Hiba a szignál dátumának feldolgozásakor: {sig_date_str} - {e}")

    apds = [
        mpf.make_addplot(
            df["Close"], type="line", panel=0, color="#ad7739", linestyle="-"
        ),
        mpf.make_addplot(indicators["SMA"], color="blue", label="SMA"),
        mpf.make_addplot(indicators["EMA"], color="orange", label="EMA"),
        mpf.make_addplot(
            indicators["BB_upper"], color="grey", linestyle="dashed", label="BB Upper"
        ),
        mpf.make_addplot(
            indicators["BB_middle"], color="black", linestyle="dotted", label="BB Mid"
        ),
        mpf.make_addplot(
            indicators["BB_lower"], color="grey", linestyle="dashed", label="BB Lower"
        ),
        mpf.make_addplot(
            indicators["RSI"], panel=2, color="purple", ylabel="RSI", label="RSI"
        ),
        mpf.make_addplot(
            rsi_overbought, panel=2, color="red", linestyle="--", label="Overbrought"
        ),
        mpf.make_addplot(
            rsi_oversold, panel=2, color="green", linestyle="--", label="Oversold"
        ),
        mpf.make_addplot(
            indicators["MACD"], panel=3, color="green", ylabel="MACD", label="MACD"
        ),
        mpf.make_addplot(
            indicators["MACD_SIGNAL"], panel=3, color="red", label="Signal"
        ),
        mpf.make_addplot(
            indicators["STOCH_K"], panel=4, color="blue", ylabel="Stoch", label="%K"
        ),
        mpf.make_addplot(indicators["STOCH_D"], panel=4, color="red", label="%D"),
        mpf.make_addplot(
            stoch_overbought, panel=4, color="red", linestyle="--", label="Overbrought"
        ),
        mpf.make_addplot(
            stoch_oversold, panel=4, color="green", linestyle="--", label="Oversold"
        ),
        mpf.make_addplot(
            indicators["ATR"], panel=5, color="yellow", ylabel="ATR", label="ATR"
        ),
        mpf.make_addplot(
            indicators["ADX"], panel=6, color="cyan", ylabel="ADX", label="ADX"
        ),
        mpf.make_addplot(
            indicators["PLUS_DI"],
            panel=6,
            color="lime",
            linestyle="dotted",
            label="+DI",
        ),
        mpf.make_addplot(
            indicators["MINUS_DI"],
            panel=6,
            color="magenta",
            linestyle="dotted",
            label="-DI",
        ),
    ]

    # ---- Markerek hozzáadása az apds listához ----
    if not np.all(np.isnan(buy_signals)):
        apds.append(
            mpf.make_addplot(
                buy_signals, type="scatter", marker="^", color="lime", s=100
            )
        )
    if not np.all(np.isnan(sell_signals)):
        apds.append(
            mpf.make_addplot(
                sell_signals, type="scatter", marker="v", color="red", s=100
            )
        )

    fig, axes = mpf.plot(
        df,
        type="candle",
        style=dark,
        ylabel="Price (ár)",
        ylabel_lower="Volume (forgalom)",
        volume=True,
        addplot=apds,
        datetime_format="%Y-%m-%d",
        tight_layout=True,
        figscale=2.5,
        panel_ratios=(6, 1, 1, 1, 1, 1, 1),
        # savefig=buf,
        returnfig=True,
        main_panel=0,
        xrotation=0,
    )

    # Tengelyek csoportosítása subplotonként
    subplot_axes = {}
    for ax in fig.axes:
        # subplot index koordináták alapján
        pos = tuple(ax.get_position().bounds)
        if pos not in subplot_axes:
            subplot_axes[pos] = []
        subplot_axes[pos].append(ax)

    # Minden subplotban: első legyen a fő (bal) tengely, többit nézzük végig
    for ax_group in subplot_axes.values():
        # jobb tengelyek eltüntetése
        if len(ax_group) < 2:
            continue  # csak bal oldali, nincs twin tengely

        main_ax = ax_group[0]
        main_ylim = main_ax.get_ylim()

        for other_ax in ax_group[1:]:
            # Állítsuk be ugyanazt az y-léptéket
            other_ax.set_ylim(main_ylim)
            # Rejtsük el a jobb oldali tengely skáláját és spinet

            other_ax.yaxis.set_ticks_position("none")
            other_ax.yaxis.set_ticklabels([])
            other_ax.spines["right"].set_visible(False)
            other_ax.yaxis.set_visible(False)

        all_lines = []
        # Töröljük az esetlegesen meglévő legendákat (bal és jobb tengelyeken is)
        for ax in ax_group:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

            all_lines.extend(ax.get_lines())

        # Szűrés: csak értelmes labelűek, ne automatikus _line0, _line1
        handles_labels = [
            (line, line.get_label())
            for line in all_lines
            if line.get_label() and not line.get_label().startswith("_")
        ]

        if handles_labels:
            handles, labels = zip(*handles_labels)
            # Csak az első (bal) tengelyre tesszük a legend-et
            ax_group[0].legend(handles, labels, loc="best")

    fig.savefig(buf)

    buf.seek(0)
    return buf


def get_equity_curve_buffer(ticker, equity_curve):
    # Grafikon generálása
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve["date"], equity_curve["portfolio_value"], label="Stratégia")
    ax.set_title(f"{ticker} Stratégia Teljesítménye")
    ax.set_ylabel("Portfólió Értéke ($)")
    ax.grid(True, linestyle="--")
    ax.legend()

    # Kép mentése memóriába
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    # Base64 kódolás a HTML-be ágyazáshoz
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return image_b64


def plot_bar_chart(subset, ticker, model_type):
    plt.figure(figsize=(10, 5))
    plt.bar(subset["reward_strategy"], subset["final_portfolio_value"])
    plt.title(f"{ticker} - {model_type} - Portfólió érték összehasonlítás")
    plt.ylabel("Végső portfólió érték")
    plt.xlabel("Reward stratégia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{rtr.CHARTS_DIR}/{ticker}_{model_type}_benchmark.png")
    plt.close()


# --- Helper: generate possible positions quickly (small search space) ---
def _generate_spiral_positions(
    ax, x, y, min_r=0.06, max_r=0.25, r_steps=6, angle_steps=12
):
    """
    Gyors spirál pozíciógenerátor. Relatív sugár és irányok alapján ad vissza (adat-koordináták).
    Paramétereket lehet csökkenteni/növelni a sebesség/precizitás tradeoff-hoz.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    radii = np.linspace(min_r, max_r, r_steps)
    angles = np.linspace(0, 2 * np.pi, angle_steps, endpoint=False)

    for r in radii:
        for angle in angles:
            ann_x = x + np.cos(angle) * r * x_range
            ann_y = y + np.sin(angle) * r * y_range
            # maradjon a tengelyen belül
            if xlim[0] <= ann_x <= xlim[1] and ylim[0] <= ann_y <= ylim[1]:
                yield ann_x, ann_y


# --- Helper: check overlap with existing boxes and legend (pixel coords) ---
def _overlaps_any(bbox, boxes):
    return any(bbox.overlaps(b) for b in boxes)


# --- Greedy placement for Top-N (we use N=3) ---
def _place_top_annotations_greedy(
    ax, top_rows, cmap, norm, legend_box=None, fontsize=9
):
    """
    top_rows: list of dict-like rows (sorted by priority — we will place highest priority first)
    cmap, norm: color mapping for text color
    legend_box: pixel bbox of legend (or None)
    Returns: list of created annotation artists
    """
    renderer = ax.figure.canvas.get_renderer()
    placed_annos = []
    placed_boxes = []

    for idx, row in enumerate(top_rows):
        x = row["sharpe_ratio"]
        y = row["final_portfolio_value"]
        text_color = cmap(norm(row["composite_score"]))
        text = (
            f"Top {idx+1}:\n{row['reward_strategy']}\n"
            f"Score: {row['composite_score']:.3f}\n"
            f"Sharpe: {row['sharpe_ratio']:.3f}\n"
            f"Value: ${row['final_portfolio_value']:,.0f}"
        )

        placed = False
        for ann_x, ann_y in _generate_spiral_positions(ax, x, y):
            ann = ax.annotate(
                text,
                xy=(x, y),
                xytext=(ann_x, ann_y),
                fontsize=fontsize,
                fontweight="bold",
                color=text_color,
                arrowprops=dict(arrowstyle="->", color=text_color, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=text_color, lw=1.2),
            )
            bbox = ann.get_window_extent(renderer=renderer)

            # Ne takarja a saját pontját
            point_px = ax.transData.transform((x, y))
            if bbox.contains(point_px[0], point_px[1]):
                ann.remove()
                continue

            # Ütközés ellenőrzés más annotációkkal
            if _overlaps_any(bbox, placed_boxes):
                ann.remove()
                continue

            # Ütközés ellenőrzés legendával
            if legend_box is not None:
                expanded_legend = legend_box.expanded(1.05, 1.1)
                if bbox.overlaps(expanded_legend):
                    ann.remove()
                    continue

            placed_annos.append(ann)
            placed_boxes.append(bbox)
            placed = True
            break

        # Ha semmi nem jó, fallback: fix jobbra-fel
        if not placed:
            ann_x = x + 0.06 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            ann_y = y + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ann = ax.annotate(
                text,
                xy=(x, y),
                xytext=(ann_x, ann_y),
                fontsize=fontsize,
                fontweight="bold",
                color=text_color,
                arrowprops=dict(arrowstyle="->", color=text_color, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=text_color, lw=1.2),
            )
            placed_annos.append(ann)
            placed_boxes.append(ann.get_window_extent(renderer=renderer))

    return placed_annos, placed_boxes


# --- Gradient scatter ---
def plot_gradient_scatter(subset, ticker, model_type):
    if subset.empty:
        return

    x = subset["sharpe_ratio"]
    y = subset["final_portfolio_value"]
    scores = subset["composite_score"]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = cm.get_cmap("viridis")

    scatter = ax.scatter(
        x, y, c=scores, cmap=cmap, s=100, alpha=0.92, edgecolor="k", linewidth=0.6
    )
    fig.colorbar(scatter, ax=ax, label="Kompozit Score")

    ax.set_title(f"{ticker} – {model_type} | Gradient színezés (kompozit score)")
    ax.set_xlabel("Sharpe-ráta")
    ax.set_ylabel("Végső portfólió érték")
    ax.grid(True, linestyle="--", alpha=0.25)

    legend = ax.get_legend()
    legend_box = None
    if legend is not None:
        legend_box = legend.get_window_extent(renderer=fig.canvas.get_renderer())

    top3 = subset.nlargest(3, "composite_score").reset_index(drop=True)
    _place_top_annotations_greedy(ax, top3.to_dict("records"), cmap, norm, legend_box)

    fig.savefig(
        f"{rtr.CHARTS_DIR}/{ticker}_{model_type}_sharpe_vs_value_gradient.png",
        bbox_inches="tight",
    )
    plt.close(fig)


# --- Strategy-colored scatter ---
def plot_strategy_colored_scatter(subset, ticker, model_type):
    if subset.empty:
        return

    strategies = subset["reward_strategy"].unique()
    color_map = cm.get_cmap("tab10", len(strategies))
    color_dict = {strategy: color_map(i) for i, strategy in enumerate(strategies)}

    scores = subset["composite_score"]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    for strategy in strategies:
        data = subset[subset["reward_strategy"] == strategy]
        ax.scatter(
            data["sharpe_ratio"],
            data["final_portfolio_value"],
            label=strategy,
            color=color_dict[strategy],
            s=100,
            alpha=0.88,
            edgecolor="k",
            linewidth=0.5,
        )

    legend = ax.legend(
        title="Reward stratégia", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    ax.set_title(f"{ticker} – {model_type} | Színezés reward stratégia szerint")
    ax.set_xlabel("Sharpe-ráta")
    ax.set_ylabel("Végső portfólió érték")
    ax.grid(True, linestyle="--", alpha=0.25)

    legend_box = legend.get_window_extent(renderer=fig.canvas.get_renderer())

    norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = cm.get_cmap("viridis")
    top3 = subset.nlargest(3, "composite_score").reset_index(drop=True)
    _place_top_annotations_greedy(ax, top3.to_dict("records"), cmap, norm, legend_box)

    fig.savefig(
        f"{rtr.CHARTS_DIR}/{ticker}_{model_type}_sharpe_vs_value_by_strategy.png",
        bbox_inches="tight",
    )
    plt.close(fig)
