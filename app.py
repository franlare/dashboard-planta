import streamlit as st
import gspread
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL "2026" & CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Neural Ops | Soybean",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" # Sidebar oculta para look app nativa
)

# Colores Semánticos (Palette: "Cyber-Industrial")
C_REAL = "#FF6B35"       # Naranja (Operador/Realidad)
C_MODEL = "#2D7DD2"      # Azul (IA/Objetivo)
C_GOOD = "#00CC99"       # Verde (En rango)
C_BAD = "#FF3366"        # Rojo (Fuera de rango)
C_NEUTRAL = "#8D99AE"    # Gris

st.markdown("""
    <style>
    /* Importar fuentes técnicas */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .main { background-color: #0E1117; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.5px; }
    
    /* Estilo de Métricas Flotantes */
    div[data-testid="stMetric"] {
        background: #1E212B;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #2E3440;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #E5E9F0; }
    div[data-testid="stMetricLabel"] { color: #8D99AE; font-size: 0.8rem; }
    
    /* Tabs Modernos */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA DE DATOS & CACHÉ
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def get_data():
    try:
        creds = st.secrets.get("google_credentials")
        if not creds: return pd.DataFrame(), False
        gc = gspread.service_account_from_dict(creds)
        # Ajustar nombres según tu sheet
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
        return df.dropna(), True
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), False

df, loaded = get_data()

# -------------------------------------------------------------------
# 3. UI PRINCIPAL
# -------------------------------------------------------------------
if loaded and not df.empty:
    
    # --- SIDEBAR (Solo Filtros Esenciales) ---
    with st.sidebar:
        st.header("🎛️ Filtros")
        dates = st.date_input("Rango", [df.index.min(), df.index.max()])
        if len(dates) == 2:
            df = df[(df.index >= pd.to_datetime(dates[0])) & (df.index <= pd.to_datetime(dates[1]) + pd.Timedelta(days=1))]
        
        st.divider()
        st.subheader("Límites QA")
        usl_a = st.number_input("Max Acidez", 0.045, 0.1, 0.045, 0.001)
        cost_soda = st.number_input("Costo Soda", 0.0, 10.0, 0.5)

    # --- HEADER: RESUMEN TÁCTICO ---
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.title("Panel de Control Neural")
        st.markdown(f"**Estado:** 🟢 Sistema en Línea | **Muestras:** {len(df)}")
    with col_head2:
        # Botón de Pánico Simulado o Refresh
        if st.button("🔄 Actualizar Análisis"):
            st.cache_data.clear()
            st.rerun()

    # --- KPI ROW (HEADS UP DISPLAY) ---
    # Calculos rapidos
    gap_soda = (df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']).mean()
    gap_costo = ((df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']) * cost_soda).sum()
    last_acid = df['sim_acidez_HIBRIDA'].iloc[-1]
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Gap Soda (Avg)", f"{gap_soda:+.2f} L/h", delta_color="inverse")
    kpi2.metric("Impacto Económico", f"${gap_costo:,.0f}", delta="Perdido" if gap_costo > 0 else "Ahorrado", delta_color="inverse")
    kpi3.metric("Acidez Actual", f"{last_acid:.3f}%", delta=f"{last_acid-usl_a:.3f} vs Limit", delta_color="inverse")
    
    # KPI Inteligente: ¿Qué tan diferente es el modelo del operador?
    correlation = df['caudal_naoh_in'].corr(df['opt_hibrida_naoh_Lh'])
    independence_score = (1 - correlation) * 100 
    kpi4.metric("Independencia IA", f"{independence_score:.1f}%", help="100% = IA piensa distinto al operador. 0% = IA copia al operador.")

    st.markdown("---")

    # --- PESTAÑAS DE PROFUNDIDAD ---
    tab1, tab2, tab3 = st.tabs(["🎛️ Sala de Control", "🧠 Inteligencia del Modelo", "📉 Economía & Calidad"])

    # ==============================================================================
    # TAB 1: SALA DE CONTROL (Interactivo & Veloz)
    # ==============================================================================
    with tab1:
        def plot_control(data, col_real, col_opt, title, unit):
            fig = go.Figure()
            # Area de Rango Operativo (Real)
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_real],
                mode='lines', name='Operador (Real)',
                line=dict(color=C_REAL, width=3),
                fill='tozeroy', fillcolor='rgba(255, 107, 53, 0.1)'
            ))
            # Linea de Objetivo (IA)
            fig.add_trace(go.Scatter(
                x=data.index, y=data[col_opt],
                mode='lines', name='Modelo (Optimo)',
                line=dict(color=C_MODEL, width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=title, height=350, hovermode="x unified",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title=unit, gridcolor='#333'), xaxis=dict(gridcolor='#333')
            )
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_control(df, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', "Control de Soda", "L/h"), use_container_width=True)
        with c2:
            st.plotly_chart(plot_control(df, 'caudal_agua_in', 'opt_hibrida_agua_Lh', "Control de Agua", "L/h"), use_container_width=True)

        # Heatmap de Errores (Para ver patrones horarios)
        st.subheader("Mapa de Calor: ¿Cuándo nos equivocamos más?")
        df['hour'] = df.index.hour
        df['error_abs'] = (df['caudal_naoh_in'] - df['opt_hibrida_naoh_Lh']).abs()
        heatmap_data = df.groupby('hour')['error_abs'].mean().reset_index()
        
        fig_heat = px.bar(heatmap_data, x='hour', y='error_abs', 
                          color='error_abs', color_continuous_scale='reds',
                          labels={'error_abs': 'Error Promedio (L/h)', 'hour': 'Hora del Día'})
        fig_heat.update_layout(height=250, margin=dict(t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ==============================================================================
    # TAB 2: BRAIN HEALTH (¿El modelo es vago?)
    # ==============================================================================
    with tab2:
        col_b1, col_b2 = st.columns([1, 2])
        
        with col_b1:
            st.markdown("### 🕵️ Auditoría de IA")
            st.info("""
            **¿Qué buscamos aquí?**
            Verificar si el modelo está proponiendo cambios reales o si el operador está ignorando al modelo.
            
            * **Puntos en la diagonal:** El modelo y el operador están de acuerdo (o el modelo copia).
            * **Nube dispersa:** Hay desacuerdo (Oportunidad de optimización).
            """)
            
            # Métricas de Volatilidad
            std_op = df['caudal_naoh_in'].std()
            std_mod = df['opt_hibrida_naoh_Lh'].std()
            
            st.metric("Volatilidad Operador", f"{std_op:.2f}", help="Cuánto varía la mano del operador")
            st.metric("Volatilidad Modelo", f"{std_mod:.2f}", help="Cuánto varía la recomendación del modelo")
            
            if std_mod < (std_op * 0.5):
                st.warning("⚠️ El modelo es muy conservador (Poca varianza).")
            elif std_mod > (std_op * 1.5):
                st.success("⚡ El modelo es agresivo/dinámico.")

        with col_b2:
            # Scatter Plot: Real vs Optimo
            fig_scat = px.scatter(
                df, x='caudal_naoh_in', y='opt_hibrida_naoh_Lh',
                color='sim_acidez_HIBRIDA', color_continuous_scale='Viridis',
                title="Correlación: Operador (X) vs Modelo (Y)",
                labels={'caudal_naoh_in': 'Lo que puso el Operador', 'opt_hibrida_naoh_Lh': 'Lo que pidió el Modelo'}
            )
            # Agregar linea de identidad (y=x)
            fig_scat.add_shape(type="line", x0=df['caudal_naoh_in'].min(), y0=df['caudal_naoh_in'].min(),
                               x1=df['caudal_naoh_in'].max(), y1=df['caudal_naoh_in'].max(),
                               line=dict(color="white", dash="dash", width=1))
            
            fig_scat.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scat, use_container_width=True)

    # ==============================================================================
    # TAB 3: ECONOMÍA & CALIDAD
    # ==============================================================================
    with tab3:
        c_eco1, c_eco2 = st.columns(2)
        
        with c_eco1:
            st.markdown("### 📉 Merma Acumulada")
            # Grafico de Merma comparativa
            fig_merma = go.Figure()
            fig_merma.add_trace(go.Box(y=df['sim_merma_TEORICA_L'], name="Teórica (Ideal)", marker_color=C_MODEL))
            fig_merma.add_trace(go.Box(y=df['sim_merma_ML_TOTAL'], name="Predicción Real (ML)", marker_color=C_REAL))
            fig_merma.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', title="Distribución de Merma", showlegend=False)
            st.plotly_chart(fig_merma, use_container_width=True)
            
        with c_eco2:
            st.markdown("### 🧪 Control de Acidez")
            fig_acid = go.Figure()
            
            # Histograma lateral para ver capacidad
            fig_acid.add_trace(go.Histogram(
                x=df['sim_acidez_HIBRIDA'], 
                nbinsx=30, 
                marker_color=C_GOOD, 
                opacity=0.7,
                name="Frecuencia"
            ))
            # Lineas de Limite
            fig_acid.add_vline(x=usl_a, line_dash="dash", line_color="red", annotation_text="Max Spec")
            
            fig_acid.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', title="Capacidad de Proceso (Acidez)")
            st.plotly_chart(fig_acid, use_container_width=True)

else:
    st.warning("Esperando datos... Verifica tu conexión a Google Sheets.")
