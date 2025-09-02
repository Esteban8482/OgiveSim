import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def make_data(n=300, seed=7):
	rng = np.random.default_rng(seed)
	# Toy exam-like scores: a slightly right-skewed mixture clipped to [0, 100]
	part1 = rng.normal(62, 12, int(n * 0.7))
	part2 = rng.normal(78, 8, n - len(part1))
	x = np.clip(np.concatenate([part1, part2]), 0, 100)
	return x


def ogive(values, bins=10):
	# Less-than ogive: cumulative frequency vs. upper class boundary
	counts, edges = np.histogram(values, bins=bins, range=(values.min(), values.max()))
	cum_counts = counts.cumsum()
	cum_pct = 100 * cum_counts / cum_counts[-1]
	x_plot = np.r_[edges[0], edges[1:]]  # start at the first lower edge
	y_plot = np.r_[0, cum_pct]            # start at 0%
	return x_plot, y_plot, edges, counts


def interp_percentile_at(value, x_plot, y_plot):
	# Estimate percentile of a score by linear interpolation on the ogive
	return float(np.interp(value, x_plot, y_plot))


def interp_value_at(percentile, x_plot, y_plot):
	# Estimate score for a given percentile (inverse lookup)
	return float(np.interp(percentile, y_plot, x_plot))

def render_streamlit_app():
	st.set_page_config(page_title="Simulación Diagrama de Ojiva", layout="wide")

	st.title("Simulación: Ogiva (menor-que) e inversa (mayor-que)")
	st.caption("Crea datos simulados tipo calificaciones, calcula ogivas y percentiles, y visualiza ambas curvas.")

	with st.sidebar:
		st.header("Parámetros")
		n = st.number_input("Tamaño de muestra", min_value=50, max_value=10000, value=300, step=50)
		bins = st.slider("Número de clases (bins)", min_value=5, max_value=30, value=12, step=1)
		seed = st.number_input("Semilla aleatoria", min_value=0, max_value=1000000, value=7, step=1)
		pass_mark = st.slider("Umbral (puntos)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)

		# Botón para regenerar datos (mantiene la semilla base pero añade jitter)
		if "_jitter" not in st.session_state:
			st.session_state._jitter = 0
		if st.button("Regenerar datos"):
			st.session_state._jitter += 1

	effective_seed = int(seed + st.session_state._jitter)

	# Datos y cálculos base (misma lógica que el script original)
	data = make_data(n=int(n), seed=effective_seed)
	x_plot, y_plot, edges, counts = ogive(data, bins=int(bins))

	# Ogiva inversa (mayor-que)
	rev_cum = counts[::-1].cumsum()[::-1]
	y_gt_core = 100 * rev_cum / rev_cum[0]
	x_gt = np.r_[edges[:-1], edges[-1]]
	y_gt = np.r_[y_gt_core, 0]

	# Percentiles clave desde los datos crudos
	p25, p50, p75 = np.percentile(data, [25, 50, 75])

	# Interpretaciones
	pass_pct = interp_percentile_at(pass_mark, x_plot, y_plot)
	top_quartile_cut = interp_value_at(75, x_plot, y_plot)
	above_pct = float(np.interp(pass_mark, x_gt, y_gt))

	# Gráfica con dos subgráficos
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

	# Ogiva menor-que (ascendente)
	ax1.plot(x_plot, y_plot, '-o', lw=2, ms=4, color='#1f77b4', label='Ogiva (menor-que)')
	for q, label in [(25, 'Q1'), (50, 'Mediana'), (75, 'Q3')]:
		v = np.percentile(data, q)
		ax1.hlines(q, x_plot[0], v, colors='gray', linestyles='dotted', lw=1)
		ax1.vlines(v, 0, q, colors='gray', linestyles='dotted', lw=1)
		ax1.plot([v], [q], 'o', color='gray')
		ax1.text(v, q + 1.5, f"{label}\n≈ {v:.1f}", ha='center', va='bottom', fontsize=8)

	# Umbral (p. ej., nota de aprobación) en menor-que
	ax1.vlines(pass_mark, 0, pass_pct, colors='#d62728', linestyles='--', lw=1)
	ax1.hlines(pass_pct, x_plot[0], pass_mark, colors='#d62728', linestyles='--', lw=1)
	ax1.text(pass_mark, pass_pct + 2, f"{pass_mark} puntos → {pass_pct:.1f}% por debajo",
			 color='#d62728', ha='center', fontsize=8)

	ax1.set_xlabel('Valor de la variable (p. ej., puntaje)')
	ax1.set_ylabel('Frecuencia acumulada (%)')
	ax1.set_title('Ogiva (menor-que)')
	ax1.set_ylim(0, 100)
	ax1.grid(alpha=0.2)
	ax1.legend(loc='lower right')

	# Ogiva mayor-que (descendente)
	ax2.plot(x_gt, y_gt, '-o', lw=2, ms=4, color='#ff7f0e', label='Ogiva inversa (mayor-que)')
	for q, label in [(25, 'Q1'), (50, 'Mediana'), (75, 'Q3')]:
		v = np.percentile(data, q)
		ax2.hlines(q, x_gt[0], v, colors='gray', linestyles='dotted', lw=1)
		ax2.vlines(v, 0, q, colors='gray', linestyles='dotted', lw=1)
		ax2.plot([v], [q], 'o', color='gray')
		ax2.text(v, q + 1.5, f"{label}\n≈ {v:.1f}", ha='center', va='bottom', fontsize=8)

	# Umbral en mayor-que (proporción por encima)
	ax2.vlines(pass_mark, 0, above_pct, colors='#ff7f0e', linestyles='--', lw=1)
	ax2.hlines(above_pct, pass_mark, x_gt[-1], colors='#ff7f0e', linestyles='--', lw=1)
	ax2.text(pass_mark, above_pct + 2, f"{above_pct:.1f}% por encima",
			 color='#ff7f0e', ha='center', fontsize=8)

	ax2.set_xlabel('Valor de la variable (p. ej., puntaje)')
	ax2.set_title('Ogiva inversa (mayor-que)')
	ax2.grid(alpha=0.2)
	ax2.legend(loc='upper right')

	plt.tight_layout()
	st.pyplot(fig, use_container_width=True)

	# Resumen en la interfaz
	st.subheader("Resumen")
	c1, c2, c3 = st.columns(3)
	c1.metric("Q1 (25%)", f"{p25:.2f}")
	c2.metric("Mediana (50%)", f"{p50:.2f}")
	c3.metric("Q3 (75%)", f"{p75:.2f}")

	st.write(f"Percentil a {pass_mark:.1f} pts: {pass_pct:.1f}% (por debajo)")
	st.write(f"Proporción por encima de {pass_mark:.1f} pts: {above_pct:.1f}%")
	st.write(f"Valor en el percentil 75: {top_quartile_cut:.2f}")


# Ejecutar como app de Streamlit
render_streamlit_app()

