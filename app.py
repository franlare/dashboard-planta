import streamlit as st
import gspread
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh  # <--- Importación Correcta
import pytz
import json

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Panel de Control V27",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CORRECCIÓN AQUÍ: Llamada directa a la función importada ---
st_autorefresh(interval=60000, limit=None, key="refresh")

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1E1E1E; border: 1px solid #333; border-radius: 5px; padding: 10px; }
    /* Status Badges */
    .status-badge { padding: 5px 10px; border-radius: 4px; font-weight: bold; color: white; }
    .status-verde { background-color: #28a745; } 
    .status-amarilla { background-color: #ffc107; color: black; } 
    .status-roja { background-color: #dc3545; } 
    .status-recup { background-color: #17a2b8; } 
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. CARGA Y MAPEO DE DATOS
# -------------------------------------------------------------------
@st.cache_data(ttl=60) 
def get_data():
    try:
        # CONEXIÓN
        # Intenta leer secrets de Streamlit Cloud, sino busca local
        if "google_credentials" in st.secrets:
            creds_dict = dict(st.secrets["google_credentials"])
            # Fix común para saltos de linea en private_key
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            gc = gspread.service_account_from_dict(creds_dict)
        else:
            # Fallback local
            gc = gspread.service_account(filename='client_secret.json')
            
        sh = gc.open("Resultados_Planta").worksheet("Resultados_Hibridos_RF")
        
        # Usar get_all_values para evitar error de headers duplicados
        raw_data = sh.get_all_values()
        
        if not raw_data or len(raw_data) < 2: 
            return pd.DataFrame(), False
        
        # Asumimos que la primera fila NO son headers confiables si vienen de appends
        # O si lo son, los usamos con cuidado.
        # Para máxima seguridad en V27, definimos las columnas manualmente por posición
        # ya que sabemos el orden exacto que definimos en el script Python.
        
        columnas_v27 = [
            'timestamp', 'ffa_pct', 'caudal_aceite',
            'caudal_naoh', 'caudal_agua', 'temperatura', # Reales
            'opt_naoh', 'opt_agua', # Optimos
            'pred_acidez', 'real_waste', 'pred_merma', 'merma_teorica',
            'costo', 'estado_merma'
        ]
        
        # Si el Excel tiene más columnas o menos, ajustamos
        df = pd.DataFrame(raw_data[1:]) # Datos sin header
        
        # Aseguramos que no explote si las columnas no coinciden en numero
        num_cols_excel = df.shape[1]
        if num_cols_excel >= len(columnas_v27):
            df = df.iloc[:, :len(columnas_v27)]
            df.columns = columnas_v27
        else:
            # Si faltan columnas, mapeamos las que podamos
            df.columns = columnas_v27[:num_cols_excel]

        # Conversión Numérica
        cols_to_numeric = [
            'ffa_pct', 'caudal_aceite', 'caudal_naoh', 'caudal_agua', 'temperatura',
            'opt_naoh', 'opt_agua', 'pred_acidez', 'real_waste', 'pred_merma', 
            'merma_teorica', 'costo'
        ]
        
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

        return df, True
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return pd.DataFrame(), False

df, loaded = get_data()

# -------------------------------------------------------------------
# 3. DASHBOARD
# -------------------------------------------------------------------
if loaded and not df.empty:
    
    # FILTROS
    with st.sidebar:
        st.header("Filtros")
        horas = st.slider("Historial (Horas)", 1, 24, 8)
        rows_to_show = horas * 6
        df_view = df.tail(rows_to_show)
    
    last = df_view.iloc[-1]

    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🛡️ Panel de Estabilidad Gianazza")
        ts = last.name.strftime('%H:%M:%S') if isinstance(last.name, pd.Timestamp) else "N/A"
        st.markdown(f"**Última Actualización:** {ts} | **Aceite:** {last.get('caudal_aceite', 0):.1f} kg/h")
    
    with c2:
        estado = str(last.get('estado_merma', 'N/A'))
        color_cls = "status-amarilla"
        if "VERDE" in estado: color_cls = "status-verde"
        elif "ROJA" in estado: color_cls = "status-roja"
        elif "RECUPERANDO" in estado: color_cls = "status-recup"
        
        st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: #262730; border-radius: 10px;">
                <div style="font-size: 12px; color: #aaa;">ESTADO ACTUAL</div>
                <div class="{color_cls} status-badge" style="font-size: 14px; margin-top: 5px;">{estado}</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- KPIS ---
    k1, k2, k3, k4 = st.columns(4)
    
    merma_real = last.get('real_waste', 0)
    merma_teo = last.get('merma_teorica', 0)
    delta_merma = merma_real - merma_teo
    
    k1.metric("Merma SCADA", f"{merma_real:.2f}%", 
              delta=f"{delta_merma:+.2f}% vs Gianazza", delta_color="inverse")
    
    k2.metric("Acidez Proyectada", f"{last.get('pred_acidez', 0):.3f}%", 
              delta="Target 0.035%", delta_color="off")

    real_naoh = last.get('caudal_naoh', 0)
    opt_naoh = last.get('opt_naoh', 0)
    k3.metric("NaOH (Real / Sug)", f"{real_naoh:.1f} / {opt_naoh:.1f}", 
             delta=f"{opt_naoh - real_naoh:+.1f} L/h")
    
    real_agua = last.get('caudal_agua', 0)
    opt_agua = last.get('opt_agua', 0)
    k4.metric("Agua (Real / Sug)", f"{real_agua:.1f} / {opt_agua:.1f}",
             delta=f"{opt_agua - real_agua:+.1f} L/h")

    # --- GRAFICO MERMA ---
    st.subheader("📉 Dinámica de Merma")
    fig_merma = go.Figure()
    
    if 'merma_teorica' in df_view.columns:
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view['merma_teorica']*1.15, 
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view['merma_teorica'],
            mode='lines', line=dict(width=0), fill='tonexty', 
            fillcolor='rgba(40, 167, 69, 0.1)', name='Zona Óptima'
        ))
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view['merma_teorica'], 
            mode='lines', name='Gianazza (L)', line=dict(color='#28a745', dash='dash')
        ))
    
    if 'real_waste' in df_view.columns:
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view['real_waste'], 
            mode='lines+markers', name='Merma Real', line=dict(color='#ffc107', width=3)
        ))

    fig_merma.update_layout(height=350, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_merma, use_container_width=True)

    # --- CONTROL ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🟠 Control de Soda")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['caudal_naoh'], name='Real', line=dict(color='#fd7e14')))
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['opt_naoh'], name='Sugerido', line=dict(color='#fd7e14', dash='dot')))
        fig.update_layout(height=250, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### 💧 Control de Agua")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['caudal_agua'], name='Real', line=dict(color='#0dcaf0')))
        fig.add_trace(go.Scatter(x=df_view.index, y=df_view['opt_agua'], name='Sugerido', line=dict(color='#0dcaf0', dash='dot')))
        fig.update_layout(height=250, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Conectando a base de datos... Si esto persiste, verifica el archivo JSON de credenciales.")
