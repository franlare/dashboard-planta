import streamlit as st
import gspread
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import pytz

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Panel de Control de Proceso",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- AUTO-REFRESH (Cada 60 seg) ---
count = st_autorefresh(interval=60000, limit=None, key="fizzbuzzcounter")

# --- PALETA DE COLORES ---
C_SODA_REAL = "#FF6B35"   # Naranja
C_SODA_OPT = "#CC5500"    # Naranja Oscuro
C_AGUA_REAL = "#00B4D8"   # Cyan
C_AGUA_OPT = "#0077B6"    # Azul Oscuro
C_TEMP = "#9D4EDD"        # Violeta (Temperatura)
C_ACID_IN = "#F4D35E"     # Amarillo Industrial (Acidez Crudo)
C_ERROR = "#E63946"       # Rojo Alerta
C_COSTO_REAL = "#FF6B35"
C_COSTO_OPT = "#2D7DD2"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .main { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #E0E0E0; }
    
    /* KPI Cards */
    div[data-testid="stMetric"] {
        background: #161b22;
        border-radius: 8px;
        padding: 10px 15px;
        border: 1px solid #30363d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #FFFFFF; font-size: 1.4rem; }
    div[data-testid="stMetricLabel"] { color: #8D99AE; font-size: 0.8rem; font-weight: 600; }
    
    /* Info Bar Style */
    .info-bar {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #8D99AE;
        background-color: #161b22;
        padding: 8px;
        border-radius: 5px;
        border: 1px solid #30363d;
        text-align: right;
        margin-bottom: 10px;
    }
    
    /* REMOVER PADDING SUPERIOR (Block Container) */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA DE DATOS
# -------------------------------------------------------------------
@st.cache_data(ttl=60) 
def get_data():
    try:
        creds = st.secrets.get("google_credentials")
        if not creds: return pd.DataFrame(), False
        gc = gspread.service_account_from_dict(creds)
        sh = gc.open("Resultados_Planta").worksheet("Resultados_Hibridos_RF")
        df = pd.DataFrame(sh.get_all_records())

        # AGREGADO: 'ffa_pct_in' a la lista
        cols_num = [
            'caudal_naoh_in', 'caudal_agua_in', 'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh',
            'sim_acidez_HIBRIDA', 'sim_jabones_HIBRIDO', 'sim_merma_ML_TOTAL', 
            'sim_merma_TEORICA_L', 'temperatura_in', 'ffa_pct_in'
        ]
        
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
        # Calcular errores
        df['err_soda'] = df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']
        df['err_agua'] = df['caudal_agua_in'] - df['opt_hibrida_agua_Lh']
            
        return df.dropna(), True
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), False

df, loaded = get_data()

# -------------------------------------------------------------------
# 3. UI PRINCIPAL
# -------------------------------------------------------------------
if loaded and not df.empty:
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Filtros")
        dates = st.date_input("Rango", [df.index.min(), df.index.max()])
        if len(dates) == 2:
            df = df[(df.index >= pd.to_datetime(dates[0])) & (df.index <= pd.to_datetime(dates[1]) + pd.Timedelta(days=1))]
        st.divider()
        cost_soda = st.number_input("Costo Soda ($/L)", 0.0, 100.0, 0.5, 0.1)

    # --- HEADER CON LOGO ---
    col_logo, col_title = st.columns([1, 7])
    with col_logo:
        st.image("logo2.png", use_container_width=True)
    with col_title:
        st.title("Panel de Control de Proceso")
        st.caption("Monitorización en Tiempo Real - Planta Neural")

    last = df.iloc[-1]
    
    # KPI HUD
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Soda: REAL", f"{last['caudal_naoh_in']:.1f} L/h", delta="Sensor")
    with k2: 
        diff_soda = last['caudal_naoh_in'] - last['opt_hibrida_naoh_Lh']
        st.metric("Soda: MODELO", f"{last['opt_hibrida_naoh_Lh']:.1f} L/h", delta=f"{diff_soda:+.1f}", delta_color="inverse")
    with k3: st.metric("Agua: REAL", f"{last['caudal_agua_in']:.1f} L/h", delta="Sensor")
    with k4: 
        diff_agua = last['caudal_agua_in'] - last['opt_hibrida_agua_Lh']
        st.metric("Agua: MODELO", f"{last['opt_hibrida_agua_Lh']:.1f} L/h", delta=f"{diff_agua:+.1f}", delta_color="inverse")

    st.markdown("---")

    # --- TABS ---
    tab_control, tab_error, tab_brain, tab_eco = st.tabs([
        "🎛️ Sala de Control", 
        "⚠️ Análisis de Error", 
        "🧠 Inteligencia Artificial", 
        "📉 Calidad & Costos"
    ])

    # ==============================================================================
    # TAB 1: SALA DE CONTROL (CON ACIDEZ CRUDO + TEMP)
    # ==============================================================================
    with tab_control:
        tz_ar = pytz.timezone('America/Argentina/Buenos_Aires')
        hora_ar = datetime.now(tz_ar).strftime('%H:%M:%S')

        st.markdown(f"""
            <div class="info-bar">
                ⏱️ Última Act: {hora_ar} | 
                📊 Muestras Analizadas: {len(df)} | 
                📡 Estado: ONLINE
            </div>
        """, unsafe_allow_html=True)
        
        # --- CÁLCULO DE LÍMITES Y ZOOM ---
        end_8h = df.index.max()
        start_8h = end_8h - pd.Timedelta(hours=8)
        df_8h = df[df.index >= start_8h]
        
        # Inicializar límites en None
        ylim_soda, ylim_agua, ylim_acid, ylim_temp = None, None, None, None

        if not df_8h.empty:
            # Soda (+- 1 L/h)
            max_soda = max(df_8h['caudal_naoh_in'].max(), df_8h['opt_hibrida_naoh_Lh'].max())
            min_soda = min(df_8h['caudal_naoh_in'].min(), df_8h['opt_hibrida_naoh_Lh'].min())
            ylim_soda = [min_soda - 1, max_soda + 1]
            
            # Agua (+- 5 L/h)
            max_agua = max(df_8h['caudal_agua_in'].max(), df_8h['opt_hibrida_agua_Lh'].max())
            min_agua = min(df_8h['caudal_agua_in'].min(), df_8h['opt_hibrida_agua_Lh'].min())
            ylim_agua = [min_agua - 5, max_agua + 5]

            # Acidez Crudo (+- 0.05 %)
            if 'ffa_pct_in' in df_8h.columns:
                max_acid = df_8h['ffa_pct_in'].max()
                min_acid = df_8h['ffa_pct_in'].min()
                ylim_acid = [min_acid - 0.05, max_acid + 0.05]

            # Temperatura (+- 2 °C)
            if 'temperatura_in' in df_8h.columns:
                max_temp = df_8h['temperatura_in'].max()
                min_temp = df_8h['temperatura_in'].min()
                ylim_temp = [min_temp - 2, max_temp + 2]

        # --- FUNCIÓN DE PLOTEO (DOS VARIABLES) ---
        def plot_control(data, col_real, col_opt, title, color_real, color_opt, xlim=None, ylim=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_real], mode='lines', name='Real',
                line=dict(color=color_real, width=3),
                fill='tozeroy', fillcolor=f"rgba{tuple(int(color_real.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
            ))
            if col_opt: # Solo si hay modelo
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col_opt], mode='lines', name='Modelo',
                    line=dict(color=color_opt, width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=title, height=280, hovermode="x unified", template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(range=xlim), 
                yaxis=dict(range=ylim)
            )
            return fig

        # --- FUNCIÓN DE PLOTEO (SINGLE VARIABLE - INPUTS) ---
        # Usamos una función similar para mantener la consistencia visual (filled area)
        def plot_input(data, col_val, title, color_val, xlim=None, ylim=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_val], mode='lines', name='Valor Real',
                line=dict(color=color_val, width=2),
                fill='tozeroy', fillcolor=f"rgba{tuple(int(color_val.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
            ))
            fig.update_layout(
                title=title, height=250, hovermode="x unified", template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(range=xlim), 
                yaxis=dict(range=ylim)
            )
            return fig

        # FILA 1: VARIABLES DE CONTROL
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_control(df, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', "🟠 Control de Soda", C_SODA_REAL, C_SODA_OPT, xlim=[start_8h, end_8h], ylim=ylim_soda), use_container_width=True)
        with c2:
            st.plotly_chart(plot_control(df, 'caudal_agua_in', 'opt_hibrida_agua_Lh', "💧 Control de Agua", C_AGUA_REAL, C_AGUA_OPT, xlim=[start_8h, end_8h], ylim=ylim_agua), use_container_width=True)

        # FILA 2: VARIABLES DE ENTRADA (PERTURBACIONES)
        st.markdown("##### 🔎 Variables de Entrada (Perturbaciones)")
        c3, c4 = st.columns(2)
        
        with c3:
            if 'ffa_pct_in' in df.columns:
                st.plotly_chart(plot_input(df, 'ffa_pct_in', "🛢️ Acidez de Crudo (%FFA)", C_ACID_IN, xlim=[start_8h, end_8h], ylim=ylim_acid), use_container_width=True)
            else:
                st.warning("Columna 'ffa_pct_in' no encontrada.")
        
        with c4:
            if 'temperatura_in' in df.columns:
                st.plotly_chart(plot_input(df, 'temperatura_in', "🌡️ Temperatura de Entrada (°C)", C_TEMP, xlim=[start_8h, end_8h], ylim=ylim_temp), use_container_width=True)
            else:
                st.warning("Columna 'temperatura_in' no encontrada.")

    # ==============================================================================
    # TAB 2: DIAGNÓSTICO DE ERROR
    # ==============================================================================
    with tab_error:
        col_sel1, col_sel2 = st.columns([1,3])
        with col_sel1:
            st.markdown("#### Configuración")
            var_analisis = st.radio("Variable a auditar:", ["Soda (NaOH)", "Agua"])
            col_err = 'err_soda' if var_analisis == "Soda (NaOH)" else 'err_agua'
            
            mae = df[col_err].abs().mean()
            bias = df[col_err].mean()
            
            st.divider()
            st.metric("MAE (Error Abs)", f"{mae:.2f} L/h", help="Promedio del error absoluto (magnitud)")
            st.metric("BIAS (Sesgo)", f"{bias:.2f} L/h", 
                      delta="Sesgo Positivo" if bias > 0 else "Sesgo Negativo",
                      help="Promedio del error. Positivo = Operador pone más que el modelo.")

        with col_sel2:
            st.markdown("#### 🕵️ Detección de Deriva (CUSUM)")
            df['cusum_temp'] = df[col_err].cumsum()
            fig_cusum = go.Figure()
            fig_cusum.add_trace(go.Scatter(
                x=df.index, y=df['cusum_temp'], mode='lines', fill='tozeroy',
                name='Error Acumulado', line=dict(color='#E056FD')
            ))
            fig_cusum.update_layout(
                height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10), yaxis_title="Lts Acumulados"
            )
            st.plotly_chart(fig_cusum, use_container_width=True)

        c_err1, c_err2 = st.columns(2)
        with c_err1:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=df.index, y=df[col_err], mode='lines', line=dict(color=C_ERROR)))
            fig_res.add_hline(y=0, line_color="white")
            fig_res.update_layout(title="Residuos Instantáneos", height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_res, use_container_width=True)
            
        with c_err2:
            fig_hist = px.histogram(df, x=col_err, nbins=40, title="Distribución de Error", color_discrete_sequence=[C_ERROR])
            fig_hist.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

    # ==============================================================================
    # TAB 3: BRAIN HEALTH
    # ==============================================================================
    with tab_brain:
        c_b1, c_b2 = st.columns([1, 2])
        with c_b1:
            st.markdown("### Auditoría IA")
            corr = df['caudal_naoh_in'].corr(df['opt_hibrida_naoh_Lh'])
            st.metric("Independencia", f"{(1-corr)*100:.1f}%")
            std_op = df['caudal_naoh_in'].std()
            std_mod = df['opt_hibrida_naoh_Lh'].std()
            st.metric("Volatilidad Modelo", f"{std_mod:.2f}", delta=f"{std_mod-std_op:.2f} vs Op")

        with c_b2:
            fig_scat = px.scatter(
                df, x='caudal_naoh_in', y='opt_hibrida_naoh_Lh',
                color='sim_acidez_HIBRIDA', color_continuous_scale='Viridis',
                title="Dispersión: Operador (X) vs Modelo (Y)"
            )
            fig_scat.add_shape(type="line", x0=df['caudal_naoh_in'].min(), y0=df['caudal_naoh_in'].min(),
                               x1=df['caudal_naoh_in'].max(), y1=df['caudal_naoh_in'].max(),
                               line=dict(color="white", dash="dash"))
            fig_scat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig_scat, use_container_width=True)

    # ==============================================================================
    # TAB 4: ECONOMÍA
    # ==============================================================================
    with tab_eco:
        st.markdown("### 💰 Impacto Financiero Acumulado")
        df['costo_real_acum'] = (df['caudal_naoh_in'] * cost_soda).cumsum()
        df['costo_opt_acum'] = (df['opt_hibrida_naoh_Lh'] * cost_soda).cumsum()
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=df.index, y=df['costo_real_acum'], mode='lines', name='Real', line=dict(color=C_COSTO_REAL)))
        fig_cost.add_trace(go.Scatter(x=df.index, y=df['costo_opt_acum'], mode='lines', name='Modelo', line=dict(color=C_COSTO_OPT, dash='dash'), fill='tonexty', fillcolor='rgba(45, 125, 210, 0.1)'))
        fig_cost.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cost, use_container_width=True)

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown("##### Distribución de Merma")
            fig_merma = go.Figure()
            fig_merma.add_trace(go.Box(y=df['sim_merma_TEORICA_L'], name="Teórica"))
            fig_merma.add_trace(go.Box(y=df['sim_merma_ML_TOTAL'], name="Real (ML)"))
            fig_merma.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig_merma, use_container_width=True)
        with col_q2:
            st.markdown("##### Control de Acidez")
            fig_acid = px.histogram(df, x='sim_acidez_HIBRIDA', nbins=30, color_discrete_sequence=["#00CC99"])
            fig_acid.add_vline(x=0.045, line_color="red", line_dash="dash")
            fig_acid.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig_acid, use_container_width=True)

else:
    st.info("Conectando con base de datos...")

