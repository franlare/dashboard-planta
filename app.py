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
# 1. CONFIGURACIÓN VISUAL (ESTILO HONEYWELL FORGE / CYBERPUNK)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Panel de Control de Proceso",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- AUTO-REFRESH (Cada 20 min) ---
count = st_autorefresh(interval=1200000, limit=None, key="fizzbuzzcounter")

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
    
    /* REMOVER PADDING SUPERIOR */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA DE DATOS (CON REDONDEO DE TIEMPO PARA UNIÓN PERFECTA)
# -------------------------------------------------------------------
@st.cache_data(ttl=30)  # Bajamos el caché a 30 segs para pruebas
def get_data():
    try:
        creds = st.secrets.get("google_credentials")
        if not creds: return pd.DataFrame(), False
        gc = gspread.service_account_from_dict(creds)
        wb = gc.open("Resultados_Planta")
        
        # --- 1. LEER AMBAS HOJAS ---
        # Hoja Nueva (RTO)
        sh_rto = wb.worksheet("Resultados_Hibridos_RTO")
        df_rto = pd.DataFrame(sh_rto.get_all_records())
        
        # Hoja Inputs (Agua Real)
        sh_inputs = wb.worksheet("Inputs_Historicos_Analytics")
        df_inputs = pd.DataFrame(sh_inputs.get_all_records())

        # --- 2. LIMPIEZA DE COLUMNAS ---
        df_rto.columns = df_rto.columns.str.strip()
        df_inputs.columns = df_inputs.columns.str.strip()

        # --- 3. SINCRONIZACIÓN DE TIEMPO (CRÍTICO) ---
        # Convertimos a datetime
        df_rto['Timestamp'] = pd.to_datetime(df_rto['Timestamp'])
        df_inputs['Timestamp'] = pd.to_datetime(df_inputs['Timestamp'])

        # TRUCO: Redondeamos al minuto más cercano (o 'floor' para ir hacia abajo)
        # Esto hace que 19:02:05 coincida con 19:02:01
        df_rto['time_key'] = df_rto['Timestamp'].dt.floor('min')
        df_inputs['time_key'] = df_inputs['Timestamp'].dt.floor('min')

        # Eliminar duplicados en inputs por si acaso hay varios registros en el mismo minuto
        df_inputs = df_inputs.drop_duplicates(subset=['time_key'])

        # --- 4. MERGE INTELIGENTE ---
        # Usamos 'time_key' para unir, pero mantenemos el Timestamp original del RTO
        if 'Caudal_Agua_L_h' in df_inputs.columns:
            # Traemos solo la llave de tiempo y el dato que nos falta
            df = pd.merge(df_rto, df_inputs[['time_key', 'Caudal_agua_L_h']], on='time_key', how='left')
        else:
            st.warning("⚠️ No se encontró la columna 'Caudal_agua_L_h' en Inputs.")
            df = df_rto

        # --- 5. RENOMBRAR Y MAPEAR ---
        column_map = {
            "Timestamp": "timestamp", # Usamos el original del RTO
            "NaOH_Actual": "caudal_naoh_in",
            "RTO_NaOH": "opt_hibrida_naoh_Lh",
            "RTO_Agua": "opt_hibrida_agua_Lh",
            "FFA_In": "ffa_pct_in",
            "Temperatura": "temperatura_in",
            "Acidez_Real_Est": "sim_acidez_HIBRIDA",
            "Jabones_Real_Est": "sim_jabones_HIBRIDO",
            "Merma_Real_Est": "sim_merma_ML_TOTAL",
            "Merma_FQ": "sim_merma_TEORICA_L",
            "Caudal_agua_L_h": "caudal_agua_in" # El dato traído de la otra hoja
        }
        
        df = df.rename(columns=column_map)

        # --- 6. CONVERTIR NÚMEROS ---
        cols_num = [
            'caudal_naoh_in', 'caudal_agua_in', 'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh',
            'sim_acidez_HIBRIDA', 'sim_jabones_HIBRIDO', 'sim_merma_ML_TOTAL', 
            'sim_merma_TEORICA_L', 'temperatura_in', 'ffa_pct_in'
        ]

        for c in cols_num:
            if c in df.columns: 
                df[c] = pd.to_numeric(df[c], errors='coerce')
            else:
                df[c] = np.nan 

        df = df.set_index('timestamp').sort_index()

        # Calcular errores
        if {'caudal_naoh_in', 'opt_hibrida_naoh_Lh'}.issubset(df.columns):
            df['err_soda'] = df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']
        
        if {'caudal_agua_in', 'opt_hibrida_agua_Lh'}.issubset(df.columns):
            df['err_agua'] = df['caudal_agua_in'] - df['opt_hibrida_agua_Lh']

        # DEBUG: Si estás probando, descomenta la siguiente línea para ver en pantalla qué columnas llegaron
        # st.write("Columnas cargadas:", df.columns.tolist())

        return df.dropna(subset=['opt_hibrida_naoh_Lh']), True 

    except Exception as e:
        st.error(f"Error crítico en carga de datos: {e}")
        return pd.DataFrame(), False

df, loaded = get_data()

# -------------------------------------------------------------------
# 3. UI PRINCIPAL (SIN CAMBIOS MAYORES PARA MANTENER ESTÉTICA)
# -------------------------------------------------------------------
if loaded and not df.empty:

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Filtros")
        # Aseguramos que haya datos para el slider
        min_date = df.index.min()
        max_date = df.index.max()
        
        dates = st.date_input("Rango", [min_date, max_date])
        
        if len(dates) == 2:
            start_d, end_d = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
            df = df[(df.index >= start_d) & (df.index <= end_d + pd.Timedelta(days=1))]
            
        st.divider()
        cost_soda = st.number_input("Costo Soda ($/L)", 0.0, 100.0, 0.5, 0.1)

    # --- HEADER CON LOGO ---
    col_logo, col_title = st.columns([1, 7])
    with col_logo:
        # Placeholder si no hay logo, o usa st.image("tu_logo.png")
        st.markdown("## 🏭") 
    with col_title:
        st.title("Refinería - RTO Dashboard")
        st.caption("Optimización en Tiempo Real - Honeywell Forge Style")

    last = df.iloc[-1]

    # KPI HUD
    k1, k2, k3, k4 = st.columns(4)
    
    # Manejo seguro de NaN para visualización
    val_soda_real = last.get('caudal_naoh_in', 0)
    val_soda_opt = last.get('opt_hibrida_naoh_Lh', 0)
    val_agua_real = last.get('caudal_agua_in', 0)
    val_agua_opt = last.get('opt_hibrida_agua_Lh', 0)

    with k1: st.metric("Soda: REAL", f"{val_soda_real:.1f} L/h", delta="Sensor")
    with k2: 
        diff_soda = val_soda_real - val_soda_opt
        st.metric("Soda: MODELO", f"{val_soda_opt:.1f} L/h", delta=f"{diff_soda:+.1f}", delta_color="inverse")
    with k3: st.metric("Agua: REAL", f"{val_agua_real:.1f} L/h", delta="Sensor")
    with k4: 
        diff_agua = val_agua_real - val_agua_opt
        st.metric("Agua: MODELO", f"{val_agua_opt:.1f} L/h", delta=f"{diff_agua:+.1f}", delta_color="inverse")

    st.markdown("---")

    # --- TABS ---
    tab_control, tab_error, tab_brain, tab_eco = st.tabs([
        "🎛️ Sala de Control", 
        "⚠️ Análisis de Error", 
        "🧠 Inteligencia Artificial", 
        "📉 Calidad & Costos"
    ])

    # ==============================================================================
    # TAB 1: SALA DE CONTROL
    # ==============================================================================
    with tab_control:
        tz_ar = pytz.timezone('America/Argentina/Buenos_Aires')
        hora_ar = datetime.now(tz_ar).strftime('%H:%M:%S')

        st.markdown(f"""
            <div class="info-bar">
                ⏱️ Última Act: {hora_ar} | 
                📊 Datapoints: {len(df)} | 
                📡 Conexión: RTO_HIBRIDO_V2
            </div>
        """, unsafe_allow_html=True)

        # --- CÁLCULO DE LÍMITES Y ZOOM (Últimas 8h) ---
        end_8h = df.index.max()
        start_8h = end_8h - pd.Timedelta(hours=8)
        df_8h = df[df.index >= start_8h]

        # Función de ploteo robusta
        def plot_control(data, col_real, col_opt, title, color_real, color_opt, xlim=None):
            fig = go.Figure()
            # Chequeamos si la columna existe antes de plotear
            if col_real in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col_real], mode='lines', name='Real',
                    line=dict(color=color_real, width=3),
                    fill='tozeroy', fillcolor=f"rgba{tuple(int(color_real.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
                ))
            if col_opt in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col_opt], mode='lines', name='Modelo (RTO)',
                    line=dict(color=color_opt, width=2, dash='dash')
                ))

            fig.update_layout(
                title=title, height=280, hovermode="x unified", template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(range=xlim)
            )
            return fig

        # FILA 1: VARIABLES DE CONTROL
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_control(df, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', "🟠 Control de Soda (NaOH)", C_SODA_REAL, C_SODA_OPT, xlim=[start_8h, end_8h]), use_container_width=True)
        with c2:
            # Nota: Si caudal_agua_in es NaN, solo graficará el Modelo
            st.plotly_chart(plot_control(df, 'caudal_agua_in', 'opt_hibrida_agua_Lh', "💧 Control de Agua", C_AGUA_REAL, C_AGUA_OPT, xlim=[start_8h, end_8h]), use_container_width=True)

        # FILA 2: PERTURBACIONES
        st.markdown("##### 🔎 Variables de Entrada")
        c3, c4 = st.columns(2)
        with c3:
            if 'ffa_pct_in' in df.columns:
                fig_acid = px.line(df_8h, y='ffa_pct_in', title="🛢️ Acidez Entrada (%FFA)", color_discrete_sequence=[C_ACID_IN])
                fig_acid.update_traces(fill='tozeroy')
                fig_acid.update_layout(height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_acid, use_container_width=True)
            else: 
                st.warning("Variable FFA_In no encontrada en hoja.")

        with c4:
            if 'temperatura_in' in df.columns and not df_8h['temperatura_in'].isna().all():
                fig_temp = px.line(df_8h, y='temperatura_in', title="🌡️ Temperatura (°C)", color_discrete_sequence=[C_TEMP])
                fig_temp.update_traces(fill='tozeroy')
                fig_temp.update_layout(height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("Temperatura no disponible en la hoja de resultados.")

    # ==============================================================================
    # TAB 2: DIAGNÓSTICO DE ERROR
    # ==============================================================================
    with tab_error:
        # Solo mostrar si tenemos datos de soda
        if 'err_soda' in df.columns:
            st.markdown("#### 🕵️ Desviación Operativa vs Modelo")
            
            col_err1, col_err2 = st.columns([3, 1])
            with col_err1:
                df['cusum_soda'] = df['err_soda'].fillna(0).cumsum()
                fig_cusum = go.Figure()
                fig_cusum.add_trace(go.Scatter(
                    x=df.index, y=df['cusum_soda'], mode='lines', fill='tozeroy',
                    name='Error Acumulado Soda', line=dict(color='#E056FD')
                ))
                fig_cusum.update_layout(
                    title="CUSUM Soda (Litros de exceso/falta acumulados)",
                    height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_cusum, use_container_width=True)
            
            with col_err2:
                mae_soda = df['err_soda'].abs().mean()
                st.metric("MAE Soda", f"{mae_soda:.2f} L/h")
                st.markdown("Si el MAE es alto, el operador no está siguiendo al RTO.")

    # ==============================================================================
    # TAB 3: BRAIN HEALTH (SIMULACIÓN)
    # ==============================================================================
    with tab_brain:
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if 'sim_acidez_HIBRIDA' in df.columns:
                fig_sim_acid = px.histogram(df, x='sim_acidez_HIBRIDA', nbins=30, title="Distribución Acidez Simulada", color_discrete_sequence=["#00CC99"])
                fig_sim_acid.add_vline(x=0.05, line_color="red", line_dash="dash", annotation_text="Max Spec")
                fig_sim_acid.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=350)
                st.plotly_chart(fig_sim_acid, use_container_width=True)
        
        with c_b2:
            if 'sim_merma_ML_TOTAL' in df.columns and 'sim_merma_TEORICA_L' in df.columns:
                fig_merma = go.Figure()
                fig_merma.add_trace(go.Box(y=df['sim_merma_TEORICA_L'], name="Merma FQ (Teórica)"))
                fig_merma.add_trace(go.Box(y=df['sim_merma_ML_TOTAL'], name="Merma ML (Real Est.)"))
                fig_merma.update_layout(title="Comparativa de Mermas", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=350)
                st.plotly_chart(fig_merma, use_container_width=True)

    # ==============================================================================
    # TAB 4: ECONOMÍA
    # ==============================================================================
    with tab_eco:
        if 'caudal_naoh_in' in df.columns and 'opt_hibrida_naoh_Lh' in df.columns:
            st.markdown("### 💰 Ahorro Potencial")
            df['ahorro_acum'] = ((df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']) * cost_soda).cumsum()
            
            # Último valor
            total_ahorro = df['ahorro_acum'].iloc[-1]
            color_ahorro = "#2ecc71" if total_ahorro > 0 else "#e74c3c"
            
            st.metric("Ahorro Acumulado (vs Operación Manual)", f"${total_ahorro:,.2f}", delta="Dinero retenido")
            
            fig_money = go.Figure()
            fig_money.add_trace(go.Scatter(x=df.index, y=df['ahorro_acum'], mode='lines', line=dict(color=color_ahorro, width=3)))
            fig_money.add_hline(y=0, line_dash="dash", line_color="white")
            fig_money.update_layout(title="Evolución del Ahorro ($)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_money, use_container_width=True)

else:
    st.info("Esperando conexión con base de datos 'Resultados_Hibridos_RTO'...")


