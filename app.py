import streamlit as st
import gspread
import pandas as pd
import altair as alt

# -------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Operativo | Refinería",
    page_icon="🇦🇷",
    layout="wide",
    initial_sidebar_state="collapsed" # Menos distracciones
)

# CSS para "limpiar" la UI y hacerla ver profesional
st.markdown("""
    <style>
    /* Títulos grandes y limpios */
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #2C3E50; }
    
    /* Métricas destacadas */
    [data-testid="stMetricValue"] { font-size: 2.5rem; color: #2E86C1; }
    
    /* Fondo de gráficos limpio */
    .vega-embed { background-color: transparent !important; }
    
    /* Eliminar padding excesivo arriba */
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 1. CARGA Y CURACIÓN DE DATOS (Auto-Healing & Localización)
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        creds_dict = st.secrets.get("google_credentials")
        if not creds_dict:
            st.error("⚠️ Error: Falta configurar 'google_credentials' en secrets.")
            return pd.DataFrame(), False
            
        gc = gspread.service_account_from_dict(creds_dict)
        spreadsheet = gc.open("Resultados_Planta")
        worksheet = spreadsheet.worksheet("Resultados_Hibridos_RF")
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        if df.empty: return df, False

        # --- AUTO-HEALING (Si faltan columnas, las creamos) ---
        for col in ['caudal_naoh_in', 'caudal_agua_in', 'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'Costo_Real_Hora' not in df.columns:
            df['Costo_Real_Hora'] = (df['caudal_naoh_in'] * 0.5) + (df['caudal_agua_in'] * 0.1)
        
        if 'Costo_Optimo_Hora' not in df.columns:
            df['Costo_Optimo_Hora'] = (df['opt_hibrida_naoh_Lh'] * 0.5) + (df['opt_hibrida_agua_Lh'] * 0.1)

        # --- LOCALIZACIÓN Y FECHAS ---
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Crear columna formateada para Argentina
            df['Horario'] = df['timestamp'] # Mantenemos el objeto datetime para ordenar
            
        return df, True

    except Exception as e:
        st.error(f"Error técnico: {e}")
        return pd.DataFrame(), False

df_raw, exito = cargar_datos()

# -------------------------------------------------------------------
# 2. FILTROS (Sidebar minimalista)
# -------------------------------------------------------------------
st.sidebar.header("📅 Configuración")
if exito and not df_raw.empty:
    min_date = df_raw['Horario'].min().to_pydatetime()
    max_date = df_raw['Horario'].max().to_pydatetime()
    
    fechas = st.sidebar.slider(
        "Rango de Fecha/Hora",
        min_value=min_date, max_value=max_date,
        value=(min_date, max_date),
        format="DD/MM HH:mm"
    )
    df = df_raw[(df_raw['Horario'] >= fechas[0]) & (df_raw['Horario'] <= fechas[1])]
else:
    df = pd.DataFrame()

# -------------------------------------------------------------------
# 3. DASHBOARD (Long Scroll)
# -------------------------------------------------------------------

if not df.empty:
    
    # === HEADER ===
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("Panel de Control de Refinería")
        st.markdown(f"**Estado:** Análisis comparativo Operador vs. IA ({len(df)} registros analizados)")
    with c2:
        # KPI PRINCIPAL (AHORRO)
        ahorro_total = df['Costo_Real_Hora'].sum() - df['Costo_Optimo_Hora'].sum()
        st.metric("Ahorro Detectado", f"${ahorro_total:,.0f}", delta="Potencial", delta_color="normal")

    st.markdown("---")

    # === SECCIÓN 1: ANÁLISIS DE DOSIFICACIÓN (LO QUE FALTABA) ===
    st.header("1. Análisis Detallado de Dosificación")
    st.markdown("Comparativa directa de caudales. Líneas suaves indican tendencias.")

    # Preparar datos para gráficos combinados
    base = alt.Chart(df).encode(
        x=alt.X('Horario:T', axis=alt.Axis(title='Horario (Hora Argentina)', format='%H:%M', grid=False)),
        tooltip=[alt.Tooltip('Horario', format='%H:%M'), 'caudal_naoh_in', 'opt_hibrida_naoh_Lh']
    )

    col_sosa, col_agua = st.columns(2)

    # --- GRÁFICO DE SOSA (NaOH) ---
    with col_sosa:
        st.subheader("🧪 Dosificación de Sosa")
        
        # Línea Real (Área sombreada suave)
        area_real = base.mark_area(opacity=0.3, interpolate='monotone', color='#1f77b4').encode(
            y=alt.Y('caudal_naoh_in', title='Caudal NaOH (L/h)')
        )
        line_real = base.mark_line(interpolate='monotone', color='#1f77b4').encode(
            y='caudal_naoh_in'
        )
        
        # Línea Óptima (Roja fuerte)
        line_opt = base.mark_line(interpolate='monotone', color='#d62728', strokeDash=[5,5], strokeWidth=3).encode(
            y='opt_hibrida_naoh_Lh'
        )

        chart_sosa = (area_real + line_real + line_opt).properties(height=400).interactive()
        st.altair_chart(chart_sosa, use_container_width=True)
        
        # KPI Sosa
        diff_sosa = (df['opt_hibrida_naoh_Lh'] - df['caudal_naoh_in']).mean()
        st.caption(f"🔴 La línea punteada roja es la recomendación. Diferencia promedio: **{diff_sosa:.2f} L/h**")

    # --- GRÁFICO DE AGUA ---
    with col_agua:
        st.subheader("💧 Dosificación de Agua")
        
        # Usamos mark_line smooth
        df_agua = df.melt('Horario', value_vars=['caudal_agua_in', 'opt_hibrida_agua_Lh'], var_name='Tipo', value_name='Litros')
        df_agua['Tipo'] = df_agua['Tipo'].replace({'caudal_agua_in': 'Operador (Actual)', 'opt_hibrida_agua_Lh': 'IA (Sugerido)'})

        chart_agua = alt.Chart(df_agua).mark_line(interpolate='monotone', strokeWidth=3).encode(
            x=alt.X('Horario:T', axis=alt.Axis(title='Horario', format='%H:%M')),
            y=alt.Y('Litros', title='Caudal Agua (L/h)', scale=alt.Scale(zero=False)),
            color=alt.Color('Tipo', scale=alt.Scale(range=['#1f77b4', '#2ca02c'])),
            tooltip=['Horario', 'Tipo', 'Litros']
        ).properties(height=400).interactive()
        
        st.altair_chart(chart_agua, use_container_width=True)
        st.caption("La IA (Verde) busca consistentemente reducir la dilución innecesaria.")

    st.markdown("---")

    # === SECCIÓN 2: INTELIGENCIA DEL MODELO (SCATTER) ===
    st.header("2. Diagnóstico de Comportamiento (Bias vs. Realidad)")
    
    col_bias_txt, col_bias_chart = st.columns([1, 3])
    
    with col_bias_txt:
        st.markdown("""
        **¿Qué estamos viendo?**
        
        Este gráfico revela si el modelo "copia" al operador.
        
        - **Línea Diagonal:** Copia perfecta.
        - **Dispersión:** El modelo está "pensando" y ajustando activamente.
        
        Observa cómo el modelo se atreve a desviarse para encontrar eficiencias.
        """)
        st.metric("Correlación (Sosa)", f"{df['caudal_naoh_in'].corr(df['opt_hibrida_naoh_Lh']):.2f}")

    with col_bias_chart:
        # Scatter Plot Grande
        min_axis = min(df['caudal_naoh_in'].min(), df['opt_hibrida_naoh_Lh'].min()) - 5
        max_axis = max(df['caudal_naoh_in'].max(), df['opt_hibrida_naoh_Lh'].max()) + 5

        scatter = alt.Chart(df).mark_circle(size=100, opacity=0.6, color='#8e44ad').encode(
            x=alt.X('caudal_naoh_in', title='Operador (L/h)', scale=alt.Scale(domain=[min_axis, max_axis])),
            y=alt.Y('opt_hibrida_naoh_Lh', title='Modelo IA (L/h)', scale=alt.Scale(domain=[min_axis, max_axis])),
            tooltip=['Horario', 'caudal_naoh_in', 'opt_hibrida_naoh_Lh']
        )
        
        linea_ref = alt.Chart(pd.DataFrame({'x': [min_axis, max_axis]})).mark_rule(color='gray', strokeDash=[3,3]).encode(x='x', y='x')
        
        st.altair_chart((scatter + linea_ref).properties(height=500), use_container_width=True)

    st.markdown("---")

    # === SECCIÓN 3: IMPACTO ECONÓMICO ===
    st.header("3. Evolución Económica")
    
    # Gráfico de Costos (Área superpuesta para ver el gap)
    costo_chart = alt.Chart(df).mark_area(opacity=0.4, interpolate='monotone').encode(
        x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M')),
        y=alt.Y('Costo_Real_Hora', title='Costo ($/h)'),
        color=alt.value('#E74C3C') # Rojo para costo real
    ).properties(title="Costo Real (Zona Roja)", height=350)
    
    costo_opt_chart = alt.Chart(df).mark_line(color='#27AE60', interpolate='monotone', strokeWidth=4).encode(
        x='Horario:T',
        y='Costo_Optimo_Hora'
    )
    
    st.altair_chart((costo_chart + costo_opt_chart).interactive(), use_container_width=True)
    st.success("La línea verde (Modelo) representa el costo objetivo. Toda el área roja por encima de la línea verde es **dinero recuperable**.")

else:
    st.info("⏳ Esperando conexión a la base de datos...")
