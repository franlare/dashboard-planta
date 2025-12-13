import streamlit as st
import gspread
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import pytz
from datetime import datetime

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Refinería RTO Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Refresco cada 60 segundos (ajustable)
count = st_autorefresh(interval=60000, limit=None, key="fizzbuzzcounter")

# --- PALETA DE COLORES ---
C_SODA_REAL = "#FF6B35"
C_SODA_OPT = "#CC5500"
C_AGUA_REAL = "#00B4D8"
C_AGUA_OPT = "#0077B6"
C_TEMP = "#9D4EDD"
C_ACID_IN = "#F4D35E"
C_ERROR = "#E63946"
C_COSTO_REAL = "#FF6B35"
C_COSTO_OPT = "#2D7DD2"

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    .main { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #E0E0E0; }
    div[data-testid="stMetric"] {
        background: #161b22; border-radius: 8px; padding: 10px; border: 1px solid #30363d;
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #FFFFFF; font-size: 1.4rem; }
    .info-bar {
        font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #8D99AE;
        background-color: #161b22; padding: 8px; border-radius: 5px; text-align: right; margin-bottom: 10px;
    }
    .block-container { padding-top: 0rem; padding-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA DE DATOS (FIX: AGREGADA TEMPERATURA Y MERGE)
# -------------------------------------------------------------------
@st.cache_data(ttl=30) 
def get_data():
    try:
        creds = st.secrets.get("google_credentials")
        if not creds: return pd.DataFrame(), False
        gc = gspread.service_account_from_dict(creds)
        wb = gc.open("Resultados_Planta")
        
        # 1. LEER HOJAS
        sh_rto = wb.worksheet("Resultados_Hibridos_RTO")
        df_rto = pd.DataFrame(sh_rto.get_all_records())
        
        sh_inputs = wb.worksheet("Inputs_Historicos_Analytics")
        df_inputs = pd.DataFrame(sh_inputs.get_all_records())

        # 2. LIMPIEZA
        df_rto.columns = df_rto.columns.str.strip()
        df_inputs.columns = df_inputs.columns.str.strip()

        # 3. SINCRONIZACIÓN TIEMPO (Redondeo a minutos)
        df_rto['Timestamp'] = pd.to_datetime(df_rto['Timestamp'])
        df_inputs['Timestamp'] = pd.to_datetime(df_inputs['Timestamp'])

        df_rto['time_key'] = df_rto['Timestamp'].dt.floor('min')
        df_inputs['time_key'] = df_inputs['Timestamp'].dt.floor('min')
        
        # Eliminar duplicados de inputs por si acaso
        df_inputs = df_inputs.drop_duplicates(subset=['time_key'])

        # 4. MERGE (FIX: Agregamos 'Temperatura_C' aquí)
        # Seleccionamos las columnas clave de inputs que queremos traer
        cols_input = ['time_key', 'Caudal_agua_L_h', 'Temperatura_C']
        
        # Verificamos que existan antes de llamar al merge para evitar errores
        existing_cols = [c for c in cols_input if c in df_inputs.columns]
        
        df = pd.merge(df_rto, df_inputs[existing_cols], on='time_key', how='left')

        # 5. MAPEO (FIX: Agregado mapeo de Temperatura)
        column_map = {
            "Timestamp": "timestamp",
            "NaOH_Actual": "caudal_naoh_in",
            "RTO_NaOH": "opt_hibrida_naoh_Lh",
            "RTO_Agua": "opt_hibrida_agua_Lh",
            "FFA_In": "ffa_pct_in",
            
            # Inputs Traídos del Merge
            "Caudal_agua_L_h": "caudal_agua_in",
            "Temperatura_C": "temperatura_in",   # <--- FIX
            
            # Simulaciones
            "Acidez_Real_Est": "sim_acidez_HIBRIDA",
            "Jabones_Real_Est": "sim_jabones_HIBRIDO",
            "Merma_Real_Est": "sim_merma_ML_TOTAL",
            "Merma_FQ": "sim_merma_TEORICA_L"
        }
        
        df = df.rename(columns=column_map)

        # 6. CONVERTIR NÚMEROS
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

        # 7. CÁLCULO DE ERRORES (FIX: Asegurar que se crean siempre)
        if 'caudal_naoh_in' in df.columns and 'opt_hibrida_naoh_Lh' in df.columns:
            df['err_soda'] = df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']
        else:
            df['err_soda'] = np.nan
        
        if 'caudal_agua_in' in df.columns and 'opt_hibrida_agua_Lh' in df.columns:
            df['err_agua'] = df['caudal_agua_in'] - df['opt_hibrida_agua_Lh']
        else:
            df['err_agua'] = np.nan

        return df.dropna(subset=['opt_hibrida_naoh_Lh']), True 

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

    # --- HEADER ---
    col_logo, col_title = st.columns([1, 7])
    with col_logo: st.markdown("## 🏭") 
    with col_title:
        st.title("Refinería - RTO Dashboard")
        st.caption("Honeywell Forge Style | Neural Control")

    last = df.iloc[-1]

    # KPI HUD
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Soda REAL", f"{last.get('caudal_naoh_in',0):.1f} L/h")
    with k2: 
        diff_soda = last.get('caudal_naoh_in',0) - last.get('opt_hibrida_naoh_Lh',0)
        st.metric("Soda RTO", f"{last.get('opt_hibrida_naoh_Lh',0):.1f} L/h", delta=f"{diff_soda:+.1f}", delta_color="inverse")
    with k3: st.metric("Agua REAL", f"{last.get('caudal_agua_in',0):.1f} L/h")
    with k4: 
        diff_agua = last.get('caudal_agua_in',0) - last.get('opt_hibrida_agua_Lh',0)
        st.metric("Agua RTO", f"{last.get('opt_hibrida_agua_Lh',0):.1f} L/h", delta=f"{diff_agua:+.1f}", delta_color="inverse")

    st.markdown("---")

    tab_control, tab_error, tab_brain, tab_eco = st.tabs([
        "🎛️ Sala de Control", "⚠️ Análisis de Error", "🧠 Inteligencia Artificial", "📉 Economía"
    ])

    # ==============================================================================
    # TAB 1: SALA DE CONTROL (FIX: GRÁFICO TEMP)
    # ==============================================================================
    with tab_control:
        tz_ar = pytz.timezone('America/Argentina/Buenos_Aires')
        st.markdown(f"""<div class="info-bar">Act: {datetime.now(tz_ar).strftime('%H:%M:%S')}</div>""", unsafe_allow_html=True)
        
        end_8h = df.index.max()
        start_8h = end_8h - pd.Timedelta(hours=8)
        
        def plot_control(data, col_real, col_opt, title, c_real, c_opt):
            fig = go.Figure()
            if col_real in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col_real], name='Real', line=dict(color=c_real, width=3), fill='tozeroy', fillcolor=f"rgba{tuple(int(c_real.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"))
            if col_opt in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col_opt], name='Modelo', line=dict(color=c_opt, width=2, dash='dash')))
            fig.update_layout(title=title, height=280, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(range=[start_8h, end_8h]))
            return fig

        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_control(df, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', "Control Soda", C_SODA_REAL, C_SODA_OPT), use_container_width=True)
        with c2: st.plotly_chart(plot_control(df, 'caudal_agua_in', 'opt_hibrida_agua_Lh', "Control Agua", C_AGUA_REAL, C_AGUA_OPT), use_container_width=True)

        st.markdown("##### Variables de Entrada")
        c3, c4 = st.columns(2)
        with c3:
            if 'ffa_pct_in' in df.columns:
                fig = px.line(df, y='ffa_pct_in', title="Acidez Crudo", color_discrete_sequence=[C_ACID_IN])
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=250, xaxis=dict(range=[start_8h, end_8h]))
                st.plotly_chart(fig, use_container_width=True)
        
        with c4:
            # FIX: Ahora 'temperatura_in' debería existir gracias al merge
            if 'temperatura_in' in df.columns and not df['temperatura_in'].isnull().all():
                fig = px.line(df, y='temperatura_in', title="Temperatura Proceso", color_discrete_sequence=[C_TEMP])
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=250, xaxis=dict(range=[start_8h, end_8h]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos de Temperatura (revisar nombre 'Temperatura_C' en Inputs)")

    # ==============================================================================
    # TAB 2: ANÁLISIS DE ERROR (FIX: BOTONES Y LÓGICA REPARADA)
    # ==============================================================================
    with tab_error:
        # 1. SELECTOR DE VARIABLE (Botones)
        col_sel, col_empty = st.columns([1, 3])
        with col_sel:
            modo_analisis = st.radio("Analizar Desviación en:", ["Soda (NaOH)", "Agua de Proceso"], horizontal=True)
        
        # 2. SELECCIÓN DE COLUMNA SEGÚN BOTÓN
        col_error_actual = 'err_soda' if modo_analisis == "Soda (NaOH)" else 'err_agua'
        
        if col_error_actual in df.columns and not df[col_error_actual].isnull().all():
            
            # --- SECCIÓN SUPERIOR: CUSUM y MÉTRICAS ---
            c_e1, c_e2 = st.columns([3, 1])
            
            with c_e1:
                # CUSUM (Suma acumulada del error para ver tendencias)
                df['cusum_temp'] = df[col_error_actual].fillna(0).cumsum()
                
                fig_cusum = go.Figure()
                fig_cusum.add_trace(go.Scatter(
                    x=df.index, y=df['cusum_temp'], 
                    mode='lines', fill='tozeroy', 
                    name='Desvío Acumulado', 
                    line=dict(color='#E056FD')
                ))
                fig_cusum.update_layout(
                    title=f"Gráfico CUSUM - Tendencia de Error ({modo_analisis})",
                    yaxis_title="Litros Acumulados (Diferencia)",
                    height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_cusum, use_container_width=True)

            with c_e2:
                # MÉTRICAS ESTADÍSTICAS
                mae = df[col_error_actual].abs().mean()
                bias = df[col_error_actual].mean()
                std_dev = df[col_error_actual].std()
                
                st.markdown("#### Estadísticas")
                st.metric("MAE (Error Abs)", f"{mae:.2f} L/h")
                st.metric("BIAS (Sesgo)", f"{bias:.2f} L/h", 
                         delta="Sobredosificación" if bias > 0 else "Subdosificación", 
                         delta_color="inverse")
                st.metric("Desviación Std", f"{std_dev:.2f}")

            # --- SECCIÓN INFERIOR: RESIDUOS E HISTOGRAMA ---
            c_res1, c_res2 = st.columns(2)
            
            with c_res1:
                # Gráfico de Residuos en el tiempo
                fig_res = px.line(df, y=col_error_actual, title="Residuos Instantáneos (Real - Modelo)")
                fig_res.add_hline(y=0, line_dash="dash", line_color="white")
                fig_res.update_traces(line_color=C_ERROR)
                fig_res.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_res, use_container_width=True)
            
            with c_res2:
                # Histograma de distribución del error
                fig_hist = px.histogram(df, x=col_error_actual, nbins=40, title="Distribución de Errores")
                fig_hist.update_traces(marker_color=C_ERROR)
                fig_hist.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)

        else:
            st.warning(f"No hay datos suficientes para calcular el error de {modo_analisis}.")


    # ==============================================================================
    # TAB 3: BRAIN (IA)
    # ==============================================================================
    with tab_brain:
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if 'sim_acidez_HIBRIDA' in df.columns:
                fig = px.histogram(df, x='sim_acidez_HIBRIDA', title="Simulación Acidez", color_discrete_sequence=["#00CC99"])
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig, use_container_width=True)
        with c_b2:
            st.info("Módulo de autodiagnóstico de IA en desarrollo...")

    # ==============================================================================
    # TAB 4: ECONOMÍA
    # ==============================================================================
    with tab_eco:
        if 'caudal_naoh_in' in df.columns:
            df['ahorro'] = (df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']) * cost_soda
            total = df['ahorro'].sum()
            st.metric("Ahorro Estimado Total", f"${total:,.2f}")
            fig = px.line(df, y=df['ahorro'].cumsum(), title="Ahorro Acumulado")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Conectando con la planta...")
