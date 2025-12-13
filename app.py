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
    page_title="Panel de Control de Proceso",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Refresco automático cada 60 segundos
count = st_autorefresh(interval=60000, limit=None, key="fizzbuzzcounter")

# --- PALETA DE COLORES ---
C_SODA_REAL = "#FF6B35"   # Naranja
C_SODA_OPT = "#CC5500"    # Naranja Oscuro
C_AGUA_REAL = "#00B4D8"   # Cyan
C_AGUA_OPT = "#0077B6"    # Azul Oscuro
C_TEMP = "#9D4EDD"        # Violeta
C_ACID_IN = "#F4D35E"     # Amarillo
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
# 2. CARGA DE DATOS
# -------------------------------------------------------------------
@st.cache_data(ttl=30) 
def get_data():
    try:
        creds = st.secrets.get("google_credentials")
        if not creds: return pd.DataFrame(), False
        gc = gspread.service_account_from_dict(creds)
        wb = gc.open("Resultados_Planta")
        
        # --- 1. LEER HOJAS ---
        sh_rto = wb.worksheet("Resultados_Hibridos_RTO")
        df_rto = pd.DataFrame(sh_rto.get_all_records())
        
        sh_inputs = wb.worksheet("Inputs_Historicos_Analytics")
        df_inputs = pd.DataFrame(sh_inputs.get_all_records())

        # --- 2. LIMPIEZA DE COLUMNAS ---
        df_rto.columns = df_rto.columns.str.strip()
        df_inputs.columns = df_inputs.columns.str.strip()

        # --- 3. PREPARACIÓN DE TIEMPO ---
        df_rto['Timestamp'] = pd.to_datetime(df_rto['Timestamp'])
        df_inputs['Timestamp'] = pd.to_datetime(df_inputs['Timestamp'])

        # Ordenar para merge_asof
        df_rto = df_rto.sort_values('Timestamp')
        df_inputs = df_inputs.sort_values('Timestamp')

        # --- 4. UNIÓN INTELIGENTE (MERGE_ASOF) ---
        cols_input = ['Timestamp', 'Caudal_Agua_L_h', 'Temperatura_C']
        cols_available = [c for c in cols_input if c in df_inputs.columns]
        
        if 'Caudal_Agua_L_h' in df_inputs.columns:
            df = pd.merge_asof(
                df_rto, 
                df_inputs[cols_available], 
                on='Timestamp', 
                direction='nearest', 
                tolerance=pd.Timedelta('5min')
            )
        else:
            st.error("⚠️ Columna 'Caudal_Agua_L_h' no encontrada en Inputs.")
            df = df_rto

        # --- 5. MAPEO DE VARIABLES ---
        column_map = {
            "Timestamp": "timestamp",
            "NaOH_Actual": "caudal_naoh_in",
            "RTO_NaOH": "opt_hibrida_naoh_Lh",
            "RTO_Agua": "opt_hibrida_agua_Lh",
            "FFA_In": "ffa_pct_in",
            "Caudal_Agua_L_h": "caudal_agua_in",
            "Temperatura_C": "temperatura_in",
            "Acidez_Real_Est": "sim_acidez_HIBRIDA",
            "Jabones_Real_Est": "sim_jabones_HIBRIDO",
            "Merma_Real_Est": "sim_merma_ML_TOTAL",
            "Merma_FQ": "sim_merma_TEORICA_L"
        }
        
        df = df.rename(columns=column_map)

        # --- 6. CONVERSIÓN NUMÉRICA ---
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

        # Relleno hacia adelante (ffill) para visualización continua
        df['caudal_agua_in'] = df['caudal_agua_in'].ffill()
        df['temperatura_in'] = df['temperatura_in'].ffill()

        # --- 7. CÁLCULO DE ERRORES ---
        if 'caudal_naoh_in' in df.columns and 'opt_hibrida_naoh_Lh' in df.columns:
            df['err_soda'] = df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']
        else: df['err_soda'] = np.nan
        
        if 'caudal_agua_in' in df.columns and 'opt_hibrida_agua_Lh' in df.columns:
            df['err_agua'] = df['caudal_agua_in'] - df['opt_hibrida_agua_Lh']
        else: df['err_agua'] = np.nan

        return df.dropna(subset=['opt_hibrida_naoh_Lh']), True 

    except Exception as e:
        st.error(f"Error crítico: {e}")
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
    with col_logo:
        try: st.image("logo2.png", use_container_width=True)
        except: st.markdown("# 🏭")
            
    with col_title:
        st.title("Panel de Control de Proceso")
        st.caption("Monitorización en Tiempo Real - Planta Neural")

    last = df.iloc[-1]

    # KPI HUD
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Soda: REAL", f"{last.get('caudal_naoh_in',0):.1f} L/h", delta="Sensor")
    with k2: 
        diff_soda = last.get('caudal_naoh_in',0) - last.get('opt_hibrida_naoh_Lh',0)
        st.metric("Soda: MODELO", f"{last.get('opt_hibrida_naoh_Lh',0):.1f} L/h", delta=f"{diff_soda:+.1f}", delta_color="inverse")
    with k3: st.metric("Agua: REAL", f"{last.get('caudal_agua_in',0):.1f} L/h", delta="Sensor")
    with k4: 
        diff_agua = last.get('caudal_agua_in',0) - last.get('opt_hibrida_agua_Lh',0)
        st.metric("Agua: MODELO", f"{last.get('opt_hibrida_agua_Lh',0):.1f} L/h", delta=f"{diff_agua:+.1f}", delta_color="inverse")

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
                📊 Muestras: {len(df)} | 
                📡 Estado: ONLINE
            </div>
        """, unsafe_allow_html=True)

        end_8h = df.index.max()
        start_8h = end_8h - pd.Timedelta(hours=8)
        
        # --- FUNCIÓN DE PLOTEO CON ZOOM INTELIGENTE ---
        def plot_control(data, col_real, col_opt, title, color_real, color_opt):
            fig = go.Figure()
            
            # 1. Agregar Trazos
            if col_real in data.columns and not data[col_real].isnull().all():
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col_real], mode='lines', name='Real',
                    line=dict(color=color_real, width=3),
                    fill='tozeroy', fillcolor=f"rgba{tuple(int(color_real.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
                ))
            if col_opt in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data[col_opt], mode='lines', name='Modelo',
                    line=dict(color=color_opt, width=2, dash='dash')
                ))

            # 2. Calcular Rango Y Dinámico (Zoom a las últimas 8 horas)
            # Filtramos los datos que se verán en pantalla
            mask = (data.index >= start_8h) & (data.index <= end_8h)
            df_view = data.loc[mask]
            
            y_min, y_max = None, None
            
            # Recopilar todos los valores visibles de Real y Modelo
            vals = []
            if col_real in df_view.columns: vals.extend(df_view[col_real].dropna().values)
            if col_opt in df_view.columns: vals.extend(df_view[col_opt].dropna().values)
            
            if vals:
                v_min, v_max = min(vals), max(vals)
                diff = v_max - v_min
                if diff == 0: diff = 1.0 # Evitar división por cero si es una línea plana
                
                # Dejamos un 10% de margen arriba y abajo para que no toque los bordes
                padding = diff * 0.1 
                y_min = v_min - padding
                y_max = v_max + padding

            # 3. Actualizar Layout con el nuevo rango Y
            fig.update_layout(
                title=title, height=280, hovermode="x unified", template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis=dict(range=[start_8h, end_8h]), # Rango X fijo (8h)
                yaxis=dict(range=[y_min, y_max])      # Rango Y calculado (Zoom)
            )
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_control(df, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', "🟠 Control de Soda", C_SODA_REAL, C_SODA_OPT), use_container_width=True)
        with c2:
            st.plotly_chart(plot_control(df, 'caudal_agua_in', 'opt_hibrida_agua_Lh', "💧 Control de Agua", C_AGUA_REAL, C_AGUA_OPT), use_container_width=True)

        st.markdown("##### 🔎 Variables de Entrada (Perturbaciones)")
        c3, c4 = st.columns(2)
        with c3:
            if 'ffa_pct_in' in df.columns:
                # También aplicamos zoom a los inputs para consistencia
                mask_in = (df.index >= start_8h) & (df.index <= end_8h)
                view_acid = df.loc[mask_in, 'ffa_pct_in']
                
                fig = px.line(df, y='ffa_pct_in', title="🛢️ Acidez de Crudo (%FFA)", color_discrete_sequence=[C_ACID_IN])
                fig.update_traces(fill='tozeroy')
                
                # Ajuste manual del rango Y para Acidez
                if not view_acid.empty:
                    ymin, ymax = view_acid.min(), view_acid.max()
                    pad = (ymax - ymin) * 0.1 if ymax != ymin else 0.05
                    fig.update_layout(yaxis=dict(range=[ymin - pad, ymax + pad]))
                
                fig.update_layout(height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(range=[start_8h, end_8h]))
                st.plotly_chart(fig, use_container_width=True)
        
        with c4:
            if 'temperatura_in' in df.columns and not df['temperatura_in'].isnull().all():
                mask_temp = (df.index >= start_8h) & (df.index <= end_8h)
                view_temp = df.loc[mask_temp, 'temperatura_in']

                fig = px.line(df, y='temperatura_in', title="🌡️ Temperatura MD2 (°C)", color_discrete_sequence=[C_TEMP])
                fig.update_traces(fill='tozeroy')
                
                # Ajuste manual del rango Y para Temp
                if not view_temp.empty:
                    tmin, tmax = view_temp.min(), view_temp.max()
                    pad = (tmax - tmin) * 0.1 if tmax != tmin else 1.0
                    fig.update_layout(yaxis=dict(range=[tmin - pad, tmax + pad]))

                fig.update_layout(height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(range=[start_8h, end_8h]))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Datos de temperatura esperando sincronización...")

    # ==============================================================================
    # TAB 2: DIAGNÓSTICO DE ERROR
    # ==============================================================================
    with tab_error:
        col_sel1, col_sel2 = st.columns([1,3])
        with col_sel1:
            st.markdown("#### Configuración")
            var_analisis = st.radio("Variable a auditar:", ["Soda (NaOH)", "Agua"])
            col_err = 'err_soda' if var_analisis == "Soda (NaOH)" else 'err_agua'
            
            if col_err in df.columns and not df[col_err].isnull().all():
                mae = df[col_err].abs().mean()
                bias = df[col_err].mean()
                st.divider()
                st.metric("MAE (Error Abs)", f"{mae:.2f} L/h")
                st.metric("BIAS (Sesgo)", f"{bias:.2f} L/h", 
                        delta="Sesgo Positivo" if bias > 0 else "Sesgo Negativo",
                        help="Positivo = Operador pone más que el modelo.")
            else:
                st.warning("Datos insuficientes.")

        with col_sel2:
            st.markdown("#### 🕵️ Detección de Deriva (CUSUM)")
            if col_err in df.columns:
                df['cusum_temp'] = df[col_err].fillna(0).cumsum()
                fig_cusum = go.Figure()
                fig_cusum.add_trace(go.Scatter(
                    x=df.index, y=df['cusum_temp'], mode='lines', fill='tozeroy',
                    name='Error Acumulado', line=dict(color='#E056FD')
                ))
                fig_cusum.update_layout(
                    height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Lts Acumulados"
                )
                st.plotly_chart(fig_cusum, use_container_width=True)

        c_err1, c_err2 = st.columns(2)
        with c_err1:
            if col_err in df.columns:
                fig_res = px.line(df, y=col_err, title="Residuos Instantáneos")
                fig_res.add_hline(y=0, line_color="white", line_dash="dash")
                fig_res.update_traces(line_color=C_ERROR)
                fig_res.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_res, use_container_width=True)

        with c_err2:
            if col_err in df.columns:
                fig_hist = px.histogram(df, x=col_err, nbins=40, title="Distribución de Error", color_discrete_sequence=[C_ERROR])
                fig_hist.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_hist, use_container_width=True)

    # ==============================================================================
    # TAB 3: BRAIN (IA)
    # ==============================================================================
    with tab_brain:
        c_b1, c_b2 = st.columns([1, 2])
        with c_b1:
            st.markdown("### Auditoría IA")
            if 'caudal_naoh_in' in df.columns and 'opt_hibrida_naoh_Lh' in df.columns:
                corr = df['caudal_naoh_in'].corr(df['opt_hibrida_naoh_Lh'])
                st.metric("Correlación Op-Modelo", f"{corr*100:.1f}%")
        
        with c_b2:
            if {'caudal_naoh_in', 'opt_hibrida_naoh_Lh', 'sim_acidez_HIBRIDA'}.issubset(df.columns):
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
        if 'caudal_naoh_in' in df.columns and 'opt_hibrida_naoh_Lh' in df.columns:
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
            if 'sim_merma_TEORICA_L' in df.columns:
                fig_merma = go.Figure()
                fig_merma.add_trace(go.Box(y=df['sim_merma_TEORICA_L'], name="Teórica"))
                if 'sim_merma_ML_TOTAL' in df.columns:
                    fig_merma.add_trace(go.Box(y=df['sim_merma_ML_TOTAL'], name="Real (ML)"))
                fig_merma.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_merma, use_container_width=True)
        with col_q2:
            st.markdown("##### Control de Acidez")
            if 'sim_acidez_HIBRIDA' in df.columns:
                fig_acid = px.histogram(df, x='sim_acidez_HIBRIDA', nbins=30, color_discrete_sequence=["#00CC99"])
                fig_acid.add_vline(x=0.045, line_color="red", line_dash="dash")
                fig_acid.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_acid, use_container_width=True)

else:
    st.info("Conectando con base de datos...")

