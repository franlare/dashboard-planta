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
C_ACEITE = "#E9C46A"      
C_JABON = "#2A9D8F"       
C_MERMA_REAL = "#E63946"  # Rojo para Merma Real
C_MERMA_OPT = "#F1FAEE"   # Blanco/Claro para Modelo Merma

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .main { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #E0E0E0; }
    
    div[data-testid="stMetric"] {
        background: #161b22;
        border-radius: 8px;
        padding: 10px 15px;
        border: 1px solid #30363d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #FFFFFF; font-size: 1.4rem; }
    div[data-testid="stMetricLabel"] { color: #8D99AE; font-size: 0.8rem; font-weight: 600; }
    
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
    
    .block-container { padding-top: 0rem; padding-bottom: 0rem; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# FUNCIÓN AUXILIAR PARA GAUGES (CORREGIDA VISUALMENTE)
# -------------------------------------------------------------------
def plot_gauge(current_val, target_val, title, color_bar, min_v, max_v, suffix=""):
    if pd.isna(target_val): target_val = 0
    if pd.isna(current_val): current_val = 0
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': "white"}}, # Aumenté un poco la fuente
        delta = {'reference': target_val, 'increasing': {'color': "#E63946"}, 'decreasing': {'color': "#2ecc71"}}, 
        number = {'suffix': suffix, 'font': {'color': "white", 'size': 24}},
        gauge = {
            'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color_bar},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [min_v, target_val], 'color': 'rgba(255, 255, 255, 0.1)'} 
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': target_val 
            }
        }
    ))
    # FIX: Aumenté el margen superior (t=80) para que entre el título
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=80, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    return fig

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

        # --- 2. LIMPIEZA ---
        df_rto.columns = df_rto.columns.str.strip()
        df_inputs.columns = df_inputs.columns.str.strip()
        df_rto['Timestamp'] = pd.to_datetime(df_rto['Timestamp'])
        df_inputs['Timestamp'] = pd.to_datetime(df_inputs['Timestamp'])
        df_rto = df_rto.sort_values('Timestamp')
        df_inputs = df_inputs.sort_values('Timestamp')

        # --- 3. MERGE INTELIGENTE (ASOF) ---
        cols_input_wanted = ['Timestamp', 'Caudal_agua_L_h', 'Temperatura_C', 'MERMA_REAL_SCADA_waste']
        cols_available = [c for c in cols_input_wanted if c in df_inputs.columns]
        
        if len(cols_available) > 1:
            df = pd.merge_asof(df_rto, df_inputs[cols_available], on='Timestamp', direction='nearest', tolerance=pd.Timedelta('5min'))
        else:
            df = df_rto

        # --- 4. MAPEO ---
        column_map = {
            "Timestamp": "timestamp",
            "Aceite_In": "caudal_aceite_in",
            "NaOH_Actual": "caudal_naoh_in",
            "RTO_NaOH": "opt_hibrida_naoh_Lh",
            "RTO_Agua": "opt_hibrida_agua_Lh",
            "FFA_In": "ffa_pct_in",
            "Caudal_agua_L_h": "caudal_agua_in",
            "Temperatura_C": "temperatura_in",
            "MERMA_REAL_SCADA_waste": "merma_scada_real",
            "Acidez_Real_Est": "sim_acidez_HIBRIDA",
            "Jabones_Real_Est": "sim_jabones_HIBRIDO",
            "Merma_Real_Est": "sim_merma_ML_TOTAL",
            "Merma_FQ": "sim_merma_TEORICA_L"
        }
        
        df = df.rename(columns=column_map)

        # --- 5. CONVERSIONES ---
        cols_num = [
            'caudal_aceite_in', 'caudal_naoh_in', 'caudal_agua_in', 
            'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh',
            'sim_acidez_HIBRIDA', 'sim_jabones_HIBRIDO', 'sim_merma_ML_TOTAL', 
            'sim_merma_TEORICA_L', 'temperatura_in', 'ffa_pct_in',
            'merma_scada_real'
        ]

        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            else: df[c] = np.nan 

        df = df.set_index('timestamp').sort_index()
        
        cols_ffill = ['caudal_agua_in', 'temperatura_in', 'merma_scada_real']
        for c in cols_ffill:
            if c in df.columns: df[c] = df[c].ffill()

        # Errores
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
    tab_control, tab_balance, tab_inputs, tab_error, tab_brain, tab_eco = st.tabs([
        "🎛️ Sala de Control", 
        "⚖️ Balance de Masa",
        "🏭 Inputs Planta",
        "⚠️ Análisis de Error", 
        "🧠 Inteligencia Artificial", 
        "📉 Costos"
    ])

    # ==============================================================================
    # TAB 1: SALA DE CONTROL
    # ==============================================================================
    with tab_control:
        tz_ar = pytz.timezone('America/Argentina/Buenos_Aires')
        st.markdown(f"""<div class="info-bar">⏱️ Act: {datetime.now(tz_ar).strftime('%H:%M:%S')} | 📡 Estado: ONLINE</div>""", unsafe_allow_html=True)

        end_8h = df.index.max()
        start_8h = end_8h - pd.Timedelta(hours=8)
        
        # --- FILA 1: GAUGE INDICATORS ---
        st.markdown("##### 🎛️ Indicadores de Proceso Instantáneos")
        g1, g2, g3 = st.columns(3)
        last_row = df.iloc[-1]
        
        # GAUGE 1: SODA
        val_soda = last_row.get('caudal_naoh_in', 0)
        tgt_soda = last_row.get('opt_hibrida_naoh_Lh', 0)
        with g1:
            st.plotly_chart(plot_gauge(val_soda, tgt_soda, "Caudal Soda", C_SODA_REAL, val_soda*0.7, val_soda*1.3, " L/h"), use_container_width=True)

        # GAUGE 2: AGUA
        val_agua = last_row.get('caudal_agua_in', 0)
        tgt_agua = last_row.get('opt_hibrida_agua_Lh', 0)
        with g2:
            st.plotly_chart(plot_gauge(val_agua, tgt_agua, "Caudal Agua", C_AGUA_REAL, val_agua*0.7, val_agua*1.3, " L/h"), use_container_width=True)

        # GAUGE 3: MERMA SCADA (REAL vs MODELO)
        val_merma = last_row.get('merma_scada_real', 0)
        tgt_merma = last_row.get('sim_merma_ML_TOTAL', 0)
        max_merma = max(10, val_merma*1.5) 
        
        with g3:
            st.plotly_chart(plot_gauge(val_merma, tgt_merma, "Merma SCADA", C_MERMA_REAL, 0, max_merma, " %"), use_container_width=True)

        st.divider()

        # --- FILA 2: GRÁFICOS DE TENDENCIA ---
        def plot_control(data, col_real, col_opt, title, color_real, color_opt, unit=""):
            fig = go.Figure()
            if col_real in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col_real], mode='lines', name='Real', line=dict(color=color_real, width=3), fill='tozeroy', fillcolor=f"rgba{tuple(int(color_real.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"))
            if col_opt in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[col_opt], mode='lines', name='Modelo', line=dict(color=color_opt, width=2, dash='dash')))
            
            mask = (data.index >= start_8h) & (data.index <= end_8h)
            vals = []
            if col_real in data.columns: vals.extend(data.loc[mask, col_real].dropna().values)
            if col_opt in data.columns: vals.extend(data.loc[mask, col_opt].dropna().values)
            y_range = None
            if vals:
                v_min, v_max = min(vals), max(vals)
                pad = (v_max - v_min) * 0.1 if v_max != v_min else 1.0
                y_range = [v_min - pad, v_max + pad]

            fig.update_layout(title=title, height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(range=[start_8h, end_8h]), yaxis=dict(range=y_range))
            return fig

        c1, c2, c3 = st.columns(3)
        with c1: 
            st.plotly_chart(plot_control(df, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', "🟠 Soda (L/h)", C_SODA_REAL, C_SODA_OPT), use_container_width=True)
        with c2: 
            st.plotly_chart(plot_control(df, 'caudal_agua_in', 'opt_hibrida_agua_Lh', "💧 Agua (L/h)", C_AGUA_REAL, C_AGUA_OPT), use_container_width=True)
        with c3:
            st.plotly_chart(plot_control(df, 'merma_scada_real', 'sim_merma_ML_TOTAL', "🔴 Merma: SCADA vs RTO", C_MERMA_REAL, C_MERMA_OPT), use_container_width=True)

    # ==============================================================================
    # TAB 2: BALANCE DE MASA (SANKEY)
    # ==============================================================================
    with tab_balance:
        st.markdown("### ⚖️ Diagrama de Flujo de Masa (Sankey)")
        last_h = df.iloc[-15:] 
        
        if 'caudal_aceite_in' in last_h.columns and 'merma_scada_real' in last_h.columns:
            q_aceite = last_h['caudal_aceite_in'].mean()
            q_soda = last_h['caudal_naoh_in'].mean()
            q_agua = last_h['caudal_agua_in'].mean()
            q_merma = last_h['merma_scada_real'].mean() 
            q_jabones = last_h['sim_jabones_HIBRIDO'].mean()
            
            q_total_in = q_aceite + q_soda + q_agua
            q_neutro = q_total_in - q_merma - q_jabones
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node = dict(
                  pad = 15, thickness = 20,
                  line = dict(color = "black", width = 0.5),
                  label = [f"Aceite Crudo ({q_aceite:.0f})", "Soda", "Agua", 
                           "MIXER", 
                           f"Aceite Neutro ({q_neutro:.0f})", f"Jabones ({q_jabones:.0f})", f"Merma SCADA ({q_merma:.0f})"],
                  color = [C_ACEITE, C_SODA_REAL, C_AGUA_REAL, "#264653", C_ACEITE, C_JABON, C_ERROR]
                ),
                link = dict(
                  source = [0, 1, 2, 3, 3, 3],
                  target = [3, 3, 3, 4, 5, 6],
                  value =  [q_aceite, q_soda, q_agua, q_neutro, q_jabones, q_merma]
              ))])
            fig_sankey.update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', font_size=14)
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.warning("Esperando datos de flujo de aceite y merma...")

    # ==============================================================================
    # TAB 3: INPUTS PLANTA
    # ==============================================================================
    with tab_inputs:
        st.markdown("### 🏭 Estado de Entradas")
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            if 'ffa_pct_in' in df.columns:
                vals = df.loc[(df.index >= start_8h) & (df.index <= end_8h), 'ffa_pct_in']
                yr = [vals.min()*0.95, vals.max()*1.05] if not vals.empty else None
                fig = px.line(df, y='ffa_pct_in', title="🛢️ Acidez (%FFA)", color_discrete_sequence=[C_ACID_IN])
                fig.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(range=[start_8h, end_8h]), yaxis=dict(range=yr))
                st.plotly_chart(fig, use_container_width=True)
        with col_in2:
            if 'temperatura_in' in df.columns:
                vals = df.loc[(df.index >= start_8h) & (df.index <= end_8h), 'temperatura_in']
                yr = [vals.min()-1, vals.max()+1] if not vals.empty else None
                fig = px.line(df, y='temperatura_in', title="🌡️ Temperatura (°C)", color_discrete_sequence=[C_TEMP])
                fig.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(range=[start_8h, end_8h]), yaxis=dict(range=yr))
                st.plotly_chart(fig, use_container_width=True)

    # ==============================================================================
    # TAB 4, 5, 6: RESTO
    # ==============================================================================
    with tab_error:
        col_sel1, col_sel2 = st.columns([1,3])
        with col_sel1:
            var_analisis = st.radio("Variable a auditar:", ["Soda (NaOH)", "Agua"])
            col_err = 'err_soda' if var_analisis == "Soda (NaOH)" else 'err_agua'
            if col_err in df.columns:
                mae = df[col_err].abs().mean()
                bias = df[col_err].mean()
                st.metric("MAE", f"{mae:.2f} L/h")
                st.metric("BIAS", f"{bias:.2f} L/h")
        with col_sel2:
            if col_err in df.columns:
                df['cusum'] = df[col_err].fillna(0).cumsum()
                fig = px.line(df, y='cusum', title="CUSUM - Deriva Acumulada")
                fig.update_traces(fill='tozeroy', line_color='#E056FD')
                fig.update_layout(height=250, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

    with tab_brain:
        st.markdown("### Auditoría IA")
        if {'caudal_naoh_in', 'opt_hibrida_naoh_Lh'}.issubset(df.columns):
            fig = px.scatter(df, x='caudal_naoh_in', y='opt_hibrida_naoh_Lh', color='sim_acidez_HIBRIDA', title="Op vs Modelo")
            fig.add_shape(type="line", x0=df['caudal_naoh_in'].min(), y0=df['caudal_naoh_in'].min(), x1=df['caudal_naoh_in'].max(), y1=df['caudal_naoh_in'].max(), line=dict(color="white", dash="dash"))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab_eco:
        st.markdown("### 💰 Costos")
        if {'caudal_naoh_in', 'opt_hibrida_naoh_Lh'}.issubset(df.columns):
            df['ahorro'] = (df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']) * cost_soda
            fig = px.line(df, y=df['ahorro'].cumsum(), title="Ahorro Acumulado ($)")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Conectando con base de datos...")
