import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import random


def get_data_from_sheet_or_simulated(sheet_id, default_seed=7, n=300):
    """
    Intenta leer datos directamente de un Google Sheet público.
    Si falla, genera datos simulados con una semilla por defecto.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df = pd.read_csv(url, header=None)  # sin cabeceras
        values = df[0].dropna().astype(float).to_numpy()  # usar primera columna
        if len(values) == 0:
            raise ValueError("La hoja está vacía")
        return values, True
    except Exception as e:
        rng = np.random.default_rng(default_seed)
        part1 = rng.normal(62, 12, int(n * 0.7))
        part2 = rng.normal(78, 8, n - len(part1))
        fallback = np.clip(np.concatenate([part1, part2]), 0, 100)
        print(f"[WARN] No se pudo leer Google Sheets: {e}. Usando datos simulados.")
        return fallback, False



def ogive(values, bins=10):
    counts, edges = np.histogram(values, bins=bins, range=(values.min(), values.max()))
    cum_counts = counts.cumsum()
    cum_pct = 100 * cum_counts / cum_counts[-1]
    x_plot = np.r_[edges[0], edges[1:]]
    y_plot = np.r_[0, cum_pct]
    return x_plot, y_plot, edges, counts


def frequency_table(values, edges, counts):
    """
    Calcula la tabla de frecuencias a partir de los bordes e histogramas.
    Devuelve un DataFrame con intervalos, frecuencia, frecuencia relativa y acumulada.
    """
    rel_freq = counts / counts.sum() * 100
    cum_freq = counts.cumsum()
    pct = rel_freq.round(2)
    pct_acum = cum_freq / counts.sum() * 100
    # Intervalos en notación estándar: [a, b) excepto el último [a, b]
    intervalos = [f"[{li:.2f}, {ls:.2f})" for li, ls in zip(edges[:-1], edges[1:])]
    if intervalos:
        intervalos[-1] = intervalos[-1][:-1] + "]"
    freq_table = pd.DataFrame({
        "Intervalo": intervalos,
        "Frecuencia": counts,
        "Frecuencia relativa (%)": pct,
        "Frecuencia acumulada": cum_freq,
        "% del total": pct,
        "% acumulado": pct_acum.round(2)
    })
    return freq_table


def interp_percentile_at(value, x_plot, y_plot):
    return float(np.interp(value, x_plot, y_plot))


def interp_value_at(percentile, x_plot, y_plot):
    return float(np.interp(percentile, y_plot, x_plot))



def ogive_inverse(counts, edges):
    """
    Calcula la ogiva mayor-que (inversa) a partir de los histogramas y bordes.
    """
    rev_cum = counts[::-1].cumsum()[::-1]
    y_gt_core = 100 * rev_cum / rev_cum[0]
    x_gt = np.r_[edges[:-1], edges[-1]]
    y_gt = np.r_[y_gt_core, 0]
    return x_gt, y_gt


def render_streamlit_app():
    st.set_page_config(page_title="Simulación Diagrama de Ojiva", layout="wide")

    st.title("Ogiva (menor-que) e inversa (mayor-que)")
    st.caption(
        "Lee datos de Google Sheets o, si falla, genera datos simulados para graficar percentiles, distribuciones y tabla de frecuencias."
    )

    with st.sidebar:
        st.header("Parámetros")
        bins = st.slider(
            "Número de clases (bins)", min_value=5, max_value=30, value=12, step=1
        )

        SHEET_ID = "10qzbSjIYQXPxjjNn9ELzXyDow_bgwSwdWcYlxAFrSuI"
        data, from_sheet = get_data_from_sheet_or_simulated(SHEET_ID)

        pass_mark = st.slider(
            "Umbral (puntos)", min_value=0.0, max_value=100.0, value=60.0, step=0.5
        )

    # Ogiva menor-que
    x_plot, y_plot, edges, counts = ogive(data, bins=int(bins))

    # Tabla de frecuencias
    freq_table = frequency_table(data, edges, counts)

    # Ogiva mayor-que
    x_gt, y_gt = ogive_inverse(counts, edges)

    # Percentiles
    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    pass_pct = interp_percentile_at(pass_mark, x_plot, y_plot)
    top_quartile_cut = interp_value_at(75, x_plot, y_plot)
    above_pct = float(np.interp(pass_mark, x_gt, y_gt))

    # Gráfica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

    ax1.plot(x_plot, y_plot, "-o", lw=2, ms=4, color="#1f77b4", label="Ogiva (menor-que)")
    ax1.set_title("Ogiva (menor-que)")
    ax1.set_xlabel("Valor")
    ax1.set_ylabel("Frecuencia acumulada (%)")
    ax1.grid(alpha=0.2)

    # Marcar cuartiles en la ogiva menor-que
    for p, y_val, color, label in zip([p25, p50, p75], [25, 50, 75], ["red", "green", "purple"], ["Q1", "Q2", "Q3"]):
        ax1.axvline(x=p, ymax=y_val / 100, color=color, linestyle="--", alpha=0.7)
        ax1.axhline(y=y_val, xmax=p / ax1.get_xlim()[1], color=color, linestyle="--", alpha=0.7)
        ax1.plot(p, y_val, 'o', color=color, label=f'{label} ({p:.1f})')

    ax1.legend(loc="lower right")

    ax2.plot(x_gt, y_gt, "-o", lw=2, ms=4, color="#ff7f0e", label="Ogiva inversa (mayor-que)")
    ax2.set_title("Ogiva inversa (mayor-que)")
    ax2.set_xlabel("Valor")
    ax2.grid(alpha=0.2)

    # Marcar cuartiles en la ogiva mayor-que
    for p, y_val, color, label in zip([p25, p50, p75], [75, 50, 25], ["red", "green", "purple"], ["Q1", "Q2", "Q3"]):
        ax2.axvline(x=p, ymax=y_val / 100, color=color, linestyle="--", alpha=0.7)
        ax2.axhline(y=y_val, xmax=p / ax2.get_xlim()[1], color=color, linestyle="--", alpha=0.7)
        ax2.plot(p, y_val, 'o', color=color, label=f'{label} ({p:.1f})')

    ax2.legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    # Resumen
    st.subheader("Resumen")
    c1, c2, c3 = st.columns(3)
    c1.metric("Q1 (25%)", f"{p25:.2f}")
    c2.metric("Mediana (50%)", f"{p50:.2f}")
    c3.metric("Q3 (75%)", f"{p75:.2f}")

    if from_sheet:
        st.success("Datos cargados desde Google Sheets ✅")
    else:
        st.warning("Usando datos simulados ⚠️")

    st.write(f"Percentil a {pass_mark:.1f} pts: {pass_pct:.1f}% (por debajo)")
    st.write(f"Proporción por encima de {pass_mark:.1f} pts: {above_pct:.1f}%")
    st.write(f"Valor en el percentil 75: {top_quartile_cut:.2f}")

    # Mostrar tabla de frecuencias
    st.subheader("Tabla de frecuencias")
    st.dataframe(freq_table, use_container_width=True)


# Ejecutar
render_streamlit_app()

