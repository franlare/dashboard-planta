import streamlit as st
import gspread
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL (THEME COLIBRÍ)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Operativo | Colibrí",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# COLORES CORPORATIVOS
COLOR_PRIMARIO = "#2E7D32"      # Verde Bosque (Real / Histórico)
COLOR_ACCENT = "#76FF03"        # Verde Lima (IA / Óptimo)
COLOR_ERROR = "#D32F2F"         # Rojo para alertas/errores
COLOR_NEUTRO = "#455A64"        # Gris azulado para textos

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    
    /* Métricas tipo Card */
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    [data-testid="stMetricValue"] {{ color: {COLOR_PRIMARIO}; font-size: 1.6rem; }}
    
    /* Tabs personalizadas */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        height: 45px;
        background-color: white;
        border-radius: 5px;
        color: {COLOR_NEUTRO};
        font-weight: 600;
        border: 1px solid #E0E0E0;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {COLOR_PRIMARIO};
        color: white;
        border: none;
    }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA DE DATOS ROBUSTA
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        creds = st.secrets.get("google_credentials")
        if not creds: return pd.DataFrame(), False
        
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open("Resultados_Planta")
        ws = sh.worksheet("Resultados_Hibridos_RF")
        df = pd.DataFrame(ws.get_all_records())

        if df.empty: return df, False

        # Conversión numérica segura
        cols = ['caudal_naoh_in', 'caudal_agua_in', 'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh', 
                'sim_acidez_HIBRIDA', 'sim_jabones_HIBRIDO', 'sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # Generar Costos si faltan
        if 'Costo_Real_Hora' not in df.columns:
            df['Costo_Real_Hora'] = (df['caudal_naoh_in'] * 0.5) + (df['caudal_agua_in'] * 0.1)
        if 'Costo_Optimo_Hora' not in df.columns:
            df['Costo_Optimo_Hora'] = (df['opt_hibrida_naoh_Lh'] * 0.5) + (df['opt_hibrida_agua_Lh'] * 0.1)

        # Fechas
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['Horario'] = df['timestamp']

        return df, True
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), False

df_raw, exito = cargar_datos()

# -------------------------------------------------------------------
# 3. FILTROS Y CÁLCULOS GLOBALES
# -------------------------------------------------------------------
if exito and not df_raw.empty:
    st.sidebar.header("🔧 Configuración")
    
    # Filtro Fecha
    min_d, max_d = df_raw['Horario'].min().to_pydatetime(), df_raw['Horario'].max().to_pydatetime()
    fechas = st.sidebar.slider("Periodo", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="DD/MM HH:mm")
    df = df_raw[(df_raw['Horario'] >= fechas[0]) & (df_raw['Horario'] <= fechas[1])].copy()
    
    # Parámetros de Calidad (Inputs)
    st.sidebar.subheader("Límites de Calidad")
    usl = st.sidebar.number_input("Acidez Max (USL)", 0.045, format="%.3f")
    lsl = st.sidebar.number_input("Acidez Min (LSL)", 0.025, format="%.3f")
    
    # --- CÁLCULOS TÉCNICOS ---
    # 1. CPK
    acidez = df['sim_acidez_HIBRIDA']
    media, std = acidez.mean(), acidez.std()
    cpk = min((usl - media)/(3*std), (media - lsl)/(3*std)) if std > 0 else 0
    
    # 2. MAE (Error de Seguimiento)
    mae_sosa = (df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']).abs().mean()
    
    # 3. Ahorro
    ahorro = df['Costo_Real_Hora'].sum() - df['Costo_Optimo_Hora'].sum()
    pct_ahorro = (ahorro / df['Costo_Real_Hora'].sum() * 100) if df['Costo_Real_Hora'].sum() > 0 else 0

else:
    df = pd.DataFrame()

# -------------------------------------------------------------------
# 4. DASHBOARD
# -------------------------------------------------------------------
if not df.empty:
    
    st.markdown(f"<h2 style='color:{COLOR_PRIMARIO}; border-bottom: 2px solid {COLOR_ACCENT};'>🐦 Aceitera Colibrí | Dashboard de Control</h2>", unsafe_allow_html=True)
    
    # --- KPIS SUPERIORES ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ahorro Potencial", f"${ahorro:,.0f}", f"{pct_ahorro:.1f}%")
    
    # Lógica de color para CPK
    cpk_delta = "Excelente" if cpk > 1.33 else "Aceptable" if cpk > 1.0 else "Crítico"
    cpk_color = "normal" if cpk > 1.0 else "inverse"
    k2.metric("Calidad (Cpk)", f"{cpk:.2f}", cpk_delta, delta_color=cpk_color)
    
    k3.metric("Error Seguimiento (Sosa)", f"{mae_sosa:.2f} L/h", delta="Desviación Promedio", delta_color="off")
    k4.metric("Registros Analizados", f"{len(df)}")
    
    st.markdown("###")

    # --- PESTAÑAS ---
    tab_fin, tab_brain, tab_qual, tab_ops = st.tabs([
        "💰 Finanzas & Estrategia", 
        "🧠 Inteligencia (Bias)", 
        "🧪 Calidad & Distribución", 
        "⚙️ Operación Detallada"
    ])

    # =================================================================
    # TAB 1: FINANZAS (GRAFICO DE COSTOS)
    # =================================================================
    with tab_fin:
        df_cost = df.melt('Horario', value_vars=['Costo_Real_Hora', 'Costo_Optimo_Hora'], var_name='Tipo', value_name='Valor')
        df_cost['Tipo'] = df_cost['Tipo'].replace({'Costo_Real_Hora': 'Actual', 'Costo_Optimo_Hora': 'Optimizado (IA)'})
        
        chart_fin = alt.Chart(df_cost).mark_area(interpolate='monotone', opacity=0.4).encode(
            x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M', title='')),
            y=alt.Y('Valor', title='Costo ($/h)'),
            color=alt.Color('Tipo', scale=alt.Scale(range=[COLOR_PRIMARIO, COLOR_ACCENT])),
            tooltip=['Horario', 'Tipo', 'Valor']
        ).properties(height=350)
        
        line_fin = alt.Chart(df_cost).mark_line(interpolate='monotone', strokeWidth=2).encode(
            x='Horario:T', y='Valor', color='Tipo'
        )
        
        st.altair_chart((chart_fin + line_fin).interactive(), use_container_width=True)

    # =================================================================
    # TAB 2: BIAS (SCATTER PLOT)
    # =================================================================
    with tab_brain:
        c_b1, c_b2 = st.columns([2, 1])
        with c_b1:
            st.subheader("Mapa de Decisión IA")
            # Scatter Plot
            min_v = min(df['caudal_naoh_in'].min(), df['opt_hibrida_naoh_Lh'].min())
            max_v = max(df['caudal_naoh_in'].max(), df['opt_hibrida_naoh_Lh'].max())
            
            scatter = alt.Chart(df).mark_circle(size=60, color=COLOR_PRIMARIO, opacity=0.5).encode(
                x=alt.X('caudal_naoh_in', title='Operador (L/h)', scale=alt.Scale(domain=[min_v, max_v])),
                y=alt.Y('opt_hibrida_naoh_Lh', title='IA (L/h)', scale=alt.Scale(domain=[min_v, max_v])),
                tooltip=['Horario', 'caudal_naoh_in', 'opt_hibrida_naoh_Lh']
            )
            line = alt.Chart(pd.DataFrame({'x': [min_v, max_v]})).mark_rule(color='gray', strokeDash=[4,4]).encode(x='x', y='x')
            st.altair_chart((scatter + line).properties(height=400), use_container_width=True)
        
        with c_b2:
            st.info("""
            **¿Cómo leer esto?**
            * **Diagonal:** La IA imita al humano.
            * **Dispersión:** La IA está optimizando activamente.
            """)
            var_ratio = df['opt_hibrida_naoh_Lh'].var() / df['caudal_naoh_in'].var()
            st.metric("Índice de Dinamismo", f"{var_ratio:.1f}x", help="La IA ajusta X veces más que el operador.")

    # =================================================================
    # TAB 3: CALIDAD (HISTOGRAMAS RECUPERADOS)
    # =================================================================
    with tab_qual:
        c_q1, c_q2 = st.columns(2)
        
        # 1. HISTOGRAMA ACIDEZ
        with c_q1:
            st.subheader("Distribución de Acidez")
            base_acidez = alt.Chart(df).mark_bar(color=COLOR_PRIMARIO, opacity=0.7).encode(
                x=alt.X('sim_acidez_HIBRIDA', bin=alt.Bin(maxbins=30), title='Acidez (%FFA)'),
                y=alt.Y('count()', title='Frecuencia'),
                tooltip=['count()']
            )
            # Líneas de límites
            rule_usl = alt.Chart(pd.DataFrame({'x': [usl]})).mark_rule(color=COLOR_ERROR, strokeDash=[5,5]).encode(x='x')
            rule_lsl = alt.Chart(pd.DataFrame({'x': [lsl]})).mark_rule(color=COLOR_ERROR, strokeDash=[5,5]).encode(x='x')
            
            st.altair_chart((base_acidez + rule_usl + rule_lsl).properties(height=300), use_container_width=True)
            st.caption(f"Límites: {lsl} - {usl} %FFA")

        # 2. JABONES (Serie de Tiempo)
        with c_q2:
            st.subheader("Jabones Finales (ppm)")
            chart_soap = alt.Chart(df).mark_line(interpolate='monotone', color='#E6B0AA').encode(
                x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M')),
                y=alt.Y('sim_jabones_HIBRIDO', title='Jabones (ppm)'),
                tooltip=['Horario', 'sim_jabones_HIBRIDO']
            ).properties(height=300)
            st.altair_chart(chart_soap, use_container_width=True)

    # =================================================================
    # TAB 4: OPERACIÓN DETALLADA (ERROR CHARTS RECUPERADOS)
    # =================================================================
    with tab_ops:
        st.subheader("Diagnóstico de Dosificación")
        
        # Gráfico Dual: Caudal + Error
        col_op1, col_op2 = st.columns(2)
        
        with col_op1:
            st.markdown("**Sosa (NaOH): Real vs IA**")
            df_naoh = df.melt('Horario', value_vars=['caudal_naoh_in', 'opt_hibrida_naoh_Lh'], var_name='Origen', value_name='Lh')
            df_naoh['Origen'] = df_naoh['Origen'].replace({'caudal_naoh_in': 'Real', 'opt_hibrida_naoh_Lh': 'IA'})
            
            chart_naoh = alt.Chart(df_naoh).mark_line(interpolate='monotone').encode(
                x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M')),
                y='Lh',
                color=alt.Color('Origen', scale=alt.Scale(range=[COLOR_PRIMARIO, COLOR_ACCENT]))
            ).properties(height=250)
            st.altair_chart(chart_naoh, use_container_width=True)
            
            # Gráfico de Error (Residuales)
            df['Error_Sosa'] = df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']
            chart_err = alt.Chart(df).mark_bar(color=COLOR_ERROR).encode(
                x='Horario:T',
                y=alt.Y('Error_Sosa', title='Desviación (L/h)')
            ).properties(height=100)
            st.altair_chart(chart_err, use_container_width=True)

        with col_op2:
            st.markdown("**Agua: Real vs IA**")
            df_agua = df.melt('Horario', value_vars=['caudal_agua_in', 'opt_hibrida_agua_Lh'], var_name='Origen', value_name='Lh')
            df_agua['Origen'] = df_agua['Origen'].replace({'caudal_agua_in': 'Real', 'opt_hibrida_agua_Lh': 'IA'})
            
            chart_agua = alt.Chart(df_agua).mark_line(interpolate='monotone').encode(
                x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M')),
                y='Lh',
                color=alt.Color('Origen', scale=alt.Scale(range=[COLOR_PRIMARIO, COLOR_ACCENT]))
            ).properties(height=250)
            st.altair_chart(chart_agua, use_container_width=True)
            
            # Error Agua
            df['Error_Agua'] = df['caudal_agua_in'] - df['opt_hibrida_agua_Lh']
            chart_err_w = alt.Chart(df).mark_bar(color=COLOR_ERROR).encode(
                x='Horario:T',
                y=alt.Y('Error_Agua', title='Desviación (L/h)')
            ).properties(height=100)
            st.altair_chart(chart_err_w, use_container_width=True)

else:
    st.info("Esperando datos de planta...")
