import streamlit as st
import gspread
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL Y CSS "DARK MODE INDUSTRIAL"
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Panel de Control de Proceso",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PALETA DE COLORES REFINADA ---
# Soda
C_SODA_REAL = "#FF6B35"   # Naranja Intenso
C_SODA_OPT = "#CC5500"    # Naranja Oscuro (Referencia)

# Agua (Diferenciada)
C_AGUA_REAL = "#00B4D8"   # Cyan Brillante
C_AGUA_OPT = "#0077B6"    # Azul Profundo (Referencia)

# Generales
C_MODEL_GENERIC = "#2D7DD2" 
C_ERROR = "#8B0000"       # Rojo Oscuro (Sangre/Alerta)
C_LIMITS = "#F1C40F"      # Amarillo (Límites SPC)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .main { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.5px; color: #E0E0E0; }
    
    /* Estilo de Tarjetas de Métricas (Botones de Lectura) */
    div[data-testid="stMetric"] {
        background: #161b22;
        border-radius: 8px;
        padding: 10px 15px;
        border: 1px solid #30363d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #8b949e;
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #FFFFFF; font-size: 1.4rem; }
    div[data-testid="stMetricLabel"] { color: #8D99AE; font-size: 0.8rem; font-weight: 600; }
    
    /* Ajuste de Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA DE DATOS
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def get_data():
    try:
        creds = st.secrets.get("google_credentials")
        if not creds: return pd.DataFrame(), False
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open("Resultados_Planta").worksheet("Resultados_Hibridos_RF")
        df = pd.DataFrame(sh.get_all_records())

        cols_num = [
            'caudal_naoh_in', 'caudal_agua_in', 'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh',
            'sim_acidez_HIBRIDA', 'sim_jabones_HIBRIDO', 'sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L'
        ]
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
        # CÁLCULO DE ERRORES (RESIDUOS)
        df['err_soda'] = df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']
        df['err_agua'] = df['caudal_agua_in'] - df['opt_hibrida_agua_Lh']
            
        return df.dropna(), True
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), False

df, loaded = get_data()

# -------------------------------------------------------------------
# 3. INTERFAZ DE USUARIO
# -------------------------------------------------------------------
if loaded and not df.empty:
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        dates = st.date_input("Rango de Fecha", [df.index.min(), df.index.max()])
        if len(dates) == 2:
            df = df[(df.index >= pd.to_datetime(dates[0])) & (df.index <= pd.to_datetime(dates[1]) + pd.Timedelta(days=1))]
        st.divider()
        st.caption("Panel de Control v2.1")

    # --- TÍTULO Y ÚLTIMAS LECTURAS (BOTONES) ---
    st.title("Panel de Control de Proceso")
    
    # Obtener últimos valores
    last = df.iloc[-1]
    
    # Panel de Estado Instantáneo (Botones de visualización)
    st.markdown("### ⏱️ Última Lectura de Proceso")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.metric("Soda: REAL", f"{last['caudal_naoh_in']:.1f} L/h", delta="Lectura Sensor")
    with col_kpi2:
        diff_soda = last['caudal_naoh_in'] - last['opt_hibrida_naoh_Lh']
        st.metric("Soda: MODELO", f"{last['opt_hibrida_naoh_Lh']:.1f} L/h", 
                  delta=f"{diff_soda:+.1f} Desvío", delta_color="inverse")
        
    with col_kpi3:
        st.metric("Agua: REAL", f"{last['caudal_agua_in']:.1f} L/h", delta="Lectura Sensor")
    with col_kpi4:
        diff_agua = last['caudal_agua_in'] - last['opt_hibrida_agua_Lh']
        st.metric("Agua: MODELO", f"{last['opt_hibrida_agua_Lh']:.1f} L/h", 
                  delta=f"{diff_agua:+.1f} Desvío", delta_color="inverse")
    
    st.markdown("---")

    # --- PESTAÑAS PRINCIPALES ---
    tab_control, tab_error, tab_brain, tab_eco = st.tabs([
        "🎛️ Sala de Control", 
        "⚠️ Análisis de Error (Modelo)", 
        "🧠 Inteligencia Artificial", 
        "📉 Calidad & Costos"
    ])

    # ==============================================================================
    # TAB 1: SALA DE CONTROL (COLORES DIFERENCIADOS)
    # ==============================================================================
    with tab_control:
        def plot_process(data, col_real, col_opt, title, color_real, color_opt):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_real], mode='lines', name='Real (Planta)',
                line=dict(color=color_real, width=3),
                fill='tozeroy', fillcolor=f"rgba{tuple(int(color_real.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_opt], mode='lines', name='Modelo (Target)',
                line=dict(color=color_opt, width=2, dash='dash')
            ))
            fig.update_layout(
                title=title, height=320, hovermode="x unified", template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_process(df, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', "🟠 Control de Soda (NaOH)", C_SODA_REAL, C_SODA_OPT), use_container_width=True)
        with c2:
            st.plotly_chart(plot_process(df, 'caudal_agua_in', 'opt_hibrida_agua_Lh', "💧 Control de Agua", C_AGUA_REAL, C_AGUA_OPT), use_container_width=True)

    # ==============================================================================
    # TAB 2: ANÁLISIS DE ERROR DEL MODELO (NUEVO)
    # ==============================================================================
    with tab_error:
        st.markdown("### Diagnóstico de Desviaciones (Residuos)")
        
        # Seleccionar variable para analizar
        var_analisis = st.radio("Variable a analizar:", ["Soda (NaOH)", "Agua"], horizontal=True)
        col_err_data = 'err_soda' if var_analisis == "Soda (NaOH)" else 'err_agua'
        
        # --- GRÁFICO 1: SERIE DE TIEMPO DE RESIDUOS (ROJO OSCURO) ---
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=df.index, y=df[col_err_data],
            mode='lines', name='Error (Real - Modelo)',
            line=dict(color=C_ERROR, width=2)
        ))
        fig_res.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)
        fig_res.update_layout(
            title=f"Evolución Temporal del Error ({var_analisis})",
            height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Error (L/h)"
        )
        st.plotly_chart(fig_res, use_container_width=True)
        
        col_spc1, col_spc2 = st.columns(2)
        
        # --- GRÁFICO 2: SPC (CONTROL ESTADÍSTICO) ---
        with col_spc1:
            # Cálculos SPC
            mu = df[col_err_data].mean()
            sigma = df[col_err_data].std()
            ucl = mu + 3*sigma
            lcl = mu - 3*sigma
            
            fig_spc = go.Figure()
            fig_spc.add_trace(go.Scatter(y=df[col_err_data], mode='markers+lines', name='Error', line=dict(color=C_ERROR, width=1)))
            fig_spc.add_hline(y=mu, line_dash="dash", line_color="white", annotation_text="Media")
            fig_spc.add_hline(y=ucl, line_dash="dash", line_color=C_LIMITS, annotation_text="+3σ (UCL)")
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color=C_LIMITS, annotation_text="-3σ (LCL)")
            
            fig_spc.update_layout(
                title="Gráfico de Control SPC (Estabilidad del Error)",
                height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_spc, use_container_width=True)
            
            if abs(mu) > 2:
                st.warning(f"⚠️ Alerta de Sesgo: El error promedio no es cero ({mu:.2f}). El modelo tiende a {'subestimar' if mu > 0 else 'sobreestimar'}.")

        # --- GRÁFICO 3: HISTOGRAMA DE RESIDUOS (NORMALIDAD) ---
        with col_spc2:
            fig_hist = px.histogram(
                df, x=col_err_data, nbins=30, 
                title="Distribución de Errores (¿Es ruido normal?)",
                color_discrete_sequence=[C_ERROR]
            )
            fig_hist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ==============================================================================
    # TAB 3: BRAIN HEALTH
    # ==============================================================================
    with tab_brain:
        c_b1, c_b2 = st.columns([1, 2])
        with c_b1:
            st.markdown("### 🕵️ Auditoría de IA")
            st.info("Comparativa de comportamiento entre Operador y Modelo.")
            st.metric("Correlación (Independencia)", f"{(1 - df['caudal_naoh_in'].corr(df['opt_hibrida_naoh_Lh']))*100:.1f}%")
        with c_b2:
            fig_scat = px.scatter(
                df, x='caudal_naoh_in', y='opt_hibrida_naoh_Lh',
                color='sim_acidez_HIBRIDA', color_continuous_scale='Viridis',
                title="Dispersión: Operador vs Modelo", opacity=0.7
            )
            fig_scat.add_shape(type="line", x0=df['caudal_naoh_in'].min(), y0=df['caudal_naoh_in'].min(),
                               x1=df['caudal_naoh_in'].max(), y1=df['caudal_naoh_in'].max(),
                               line=dict(color="white", dash="dash"))
            fig_scat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig_scat, use_container_width=True)

    # ==============================================================================
    # TAB 4: ECONOMÍA & CALIDAD
    # ==============================================================================
    with tab_eco:
        c_eco1, c_eco2 = st.columns(2)
        with c_eco1:
            st.markdown("### 📉 Merma Acumulada")
            fig_merma = go.Figure()
            fig_merma.add_trace(go.Box(y=df['sim_merma_TEORICA_L'], name="Teórica", marker_color=C_MODEL_GENERIC))
            fig_merma.add_trace(go.Box(y=df['sim_merma_ML_TOTAL'], name="Real (ML)", marker_color=C_SODA_REAL))
            fig_merma.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_merma, use_container_width=True)
        with c_eco2:
            st.markdown("### 🧪 Control de Acidez")
            fig_acid = px.histogram(df, x='sim_acidez_HIBRIDA', nbins=30, color_discrete_sequence=["#00CC99"])
            fig_acid.add_vline(x=0.045, line_dash="dash", line_color="red", annotation_text="Límite")
            fig_acid.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_acid, use_container_width=True)

else:
    st.info("Conectando con el proceso...")
