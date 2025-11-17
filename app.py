import streamlit as st
import gspread
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL (THEME COLIBRÍ - INTEGRADO)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Operativo | Colibrí",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PALETA COLIBRÍ REFINADA
COLOR_PRIMARIO = "#2E7D32"      # Verde Bosque (Identidad)
COLOR_ACCENT = "#76FF03"        # Verde Lima (IA/Highlights)
COLOR_FONDO_CARD = "#F1F8E9"    # <--- NUEVO: Verde muy pálido (No más blanco duro)
COLOR_BORDE_CARD = "#C5E1A5"    # Borde verde suave
COLOR_TEXTO = "#1B5E20"         # Verde muy oscuro
COLOR_ERROR = "#D32F2F"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {COLOR_TEXTO};
    }}
    
    /* --- ARREGLO DE LOS CUADROS BLANCOS --- */
    div[data-testid="stMetric"] {{
        background-color: {COLOR_FONDO_CARD}; /* Fondo verde suave */
        border: 1px solid {COLOR_BORDE_CARD};
        padding: 15px;
        border-radius: 12px; /* Bordes más redondeados */
        box-shadow: 0 2px 4px rgba(46, 125, 50, 0.1); /* Sombra sutil verdosa */
    }}
    
    [data-testid="stMetricValue"] {{
        color: {COLOR_PRIMARIO};
        font-size: 1.8rem;
        font-weight: 700;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLOR_TEXTO};
        opacity: 0.8;
    }}
    
    /* --- PESTAÑAS MEJORADAS --- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 45px;
        background-color: {COLOR_FONDO_CARD}; /* Coherencia con las cards */
        border-radius: 8px;
        color: {COLOR_TEXTO};
        font-weight: 600;
        border: 1px solid {COLOR_BORDE_CARD};
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {COLOR_PRIMARIO};
        color: white;
        border: none;
    }}
    
    /* Título principal con estilo */
    h2 {{
        color: {COLOR_PRIMARIO};
        border-bottom: 3px solid {COLOR_ACCENT};
        padding-bottom: 10px;
        margin-bottom: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA DE DATOS
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

        # Conversiones y limpieza
        cols = ['caudal_naoh_in', 'caudal_agua_in', 'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh', 
                'sim_acidez_HIBRIDA', 'sim_jabones_HIBRIDO', 'sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        if 'Costo_Real_Hora' not in df.columns:
            df['Costo_Real_Hora'] = (df['caudal_naoh_in'] * 0.5) + (df['caudal_agua_in'] * 0.1)
        if 'Costo_Optimo_Hora' not in df.columns:
            df['Costo_Optimo_Hora'] = (df['opt_hibrida_naoh_Lh'] * 0.5) + (df['opt_hibrida_agua_Lh'] * 0.1)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['Horario'] = df['timestamp']

        return df, True
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return pd.DataFrame(), False

df_raw, exito = cargar_datos()

# -------------------------------------------------------------------
# 3. LÓGICA DE NEGOCIO
# -------------------------------------------------------------------
if exito and not df_raw.empty:
    st.sidebar.header("⚙️ Filtros")
    min_d, max_d = df_raw['Horario'].min().to_pydatetime(), df_raw['Horario'].max().to_pydatetime()
    fechas = st.sidebar.slider("Rango", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="DD/MM HH:mm")
    
    df = df_raw[(df_raw['Horario'] >= fechas[0]) & (df_raw['Horario'] <= fechas[1])].copy()
    
    # Cálculos
    ahorro = df['Costo_Real_Hora'].sum() - df['Costo_Optimo_Hora'].sum()
    pct_ahorro = (ahorro / df['Costo_Real_Hora'].sum() * 100) if df['Costo_Real_Hora'].sum() > 0 else 0
    
    acidez = df['sim_acidez_HIBRIDA']
    media, std = acidez.mean(), acidez.std()
    usl, lsl = 0.045, 0.025
    cpk = min((usl - media)/(3*std), (media - lsl)/(3*std)) if std > 0 else 0
    
    mae_sosa = (df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']).abs().mean()

else:
    df = pd.DataFrame()

# -------------------------------------------------------------------
# 4. DASHBOARD VISUAL
# -------------------------------------------------------------------
if not df.empty:
    
    st.markdown("<h2>🐦 Aceitera Colibrí | Dashboard Híbrido</h2>", unsafe_allow_html=True)
    
    # KPI CARDS (Ahora con fondo verde suave)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ahorro Detectado", f"${ahorro:,.0f}", f"{pct_ahorro:.1f}%")
    
    cpk_delta = "Óptimo" if cpk > 1.33 else "Revisar"
    cpk_col = "normal" if cpk > 1.33 else "inverse"
    k2.metric("Calidad (Cpk)", f"{cpk:.2f}", cpk_delta, delta_color=cpk_col)
    
    k3.metric("Error Seguimiento", f"{mae_sosa:.2f} L/h", "Sosa Caústica", delta_color="off")
    k4.metric("Datos Procesados", f"{len(df)}", "Registros")
    
    st.markdown("###")

    # PESTAÑAS
    tab_fin, tab_brain, tab_qual, tab_ops = st.tabs([
        "💰 Finanzas", "🧠 Inteligencia IA", "🧪 Calidad", "⚙️ Operación"
    ])

    # --- TAB 1: FINANZAS ---
    with tab_fin:
        df_cost = df.melt('Horario', value_vars=['Costo_Real_Hora', 'Costo_Optimo_Hora'], var_name='Tipo', value_name='Valor')
        df_cost['Tipo'] = df_cost['Tipo'].replace({'Costo_Real_Hora': 'Actual (Operador)', 'Costo_Optimo_Hora': 'Objetivo (IA)'})
        
        # Gráfico de Área Suave
        chart = alt.Chart(df_cost).mark_area(interpolate='monotone', opacity=0.5).encode(
            x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M', title='')),
            y=alt.Y('Valor', title='Costo ($/h)'),
            color=alt.Color('Tipo', scale=alt.Scale(domain=['Actual (Operador)', 'Objetivo (IA)'], range=[COLOR_PRIMARIO, COLOR_ACCENT])),
            tooltip=['Horario', 'Tipo', 'Valor']
        ).properties(height=350)
        
        st.altair_chart(chart.interactive(), use_container_width=True)

    # --- TAB 2: INTELIGENCIA (BIAS) ---
    with tab_brain:
        c1, c2 = st.columns([2,1])
        with c1:
            st.subheader("Mapa de Decisión")
            min_v = min(df['caudal_naoh_in'].min(), df['opt_hibrida_naoh_Lh'].min())
            max_v = max(df['caudal_naoh_in'].max(), df['opt_hibrida_naoh_Lh'].max())
            
            scatter = alt.Chart(df).mark_circle(size=70, color=COLOR_PRIMARIO, opacity=0.6).encode(
                x=alt.X('caudal_naoh_in', title='Operador (L/h)', scale=alt.Scale(domain=[min_v, max_v])),
                y=alt.Y('opt_hibrida_naoh_Lh', title='IA (L/h)', scale=alt.Scale(domain=[min_v, max_v])),
                tooltip=['Horario', 'caudal_naoh_in', 'opt_hibrida_naoh_Lh']
            )
            line = alt.Chart(pd.DataFrame({'x': [min_v, max_v]})).mark_rule(color='gray', strokeDash=[4,4]).encode(x='x', y='x')
            st.altair_chart((scatter + line).properties(height=400), use_container_width=True)
        
        with c2:
            st.info("Los puntos alejados de la línea punteada indican momentos donde la IA propuso una mejora activa frente a la operación manual.")

    # --- TAB 3: CALIDAD ---
    with tab_qual:
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.subheader("Histograma de Acidez")
            base = alt.Chart(df).mark_bar(color=COLOR_PRIMARIO, opacity=0.8).encode(
                x=alt.X('sim_acidez_HIBRIDA', bin=alt.Bin(maxbins=25), title='Acidez (%FFA)'),
                y='count()'
            )
            st.altair_chart(base.properties(height=300), use_container_width=True)
        
        with col_q2:
            st.subheader("Tendencia de Jabones")
            line_j = alt.Chart(df).mark_line(interpolate='monotone', color='#8D6E63').encode( # Marrón suave
                x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M')),
                y=alt.Y('sim_jabones_HIBRIDO', title='ppm')
            ).properties(height=300)
            st.altair_chart(line_j, use_container_width=True)

    # --- TAB 4: OPERACIÓN ---
    with tab_ops:
        st.subheader("Detalle de Dosificación (Sosa Cáustica)")
        df_sosa = df.melt('Horario', value_vars=['caudal_naoh_in', 'opt_hibrida_naoh_Lh'], var_name='Origen', value_name='Lh')
        df_sosa['Origen'] = df_sosa['Origen'].replace({'caudal_naoh_in': 'Real', 'opt_hibrida_naoh_Lh': 'IA'})
        
        chart_s = alt.Chart(df_sosa).mark_line(interpolate='monotone', strokeWidth=2).encode(
            x=alt.X('Horario:T', axis=alt.Axis(format='%H:%M')),
            y='Lh',
            color=alt.Color('Origen', scale=alt.Scale(range=[COLOR_PRIMARIO, COLOR_ACCENT]))
        ).properties(height=300)
        st.altair_chart(chart_s.interactive(), use_container_width=True)

else:
    st.warning("⏳ Conectando con base de datos...")
