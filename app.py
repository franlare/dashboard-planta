import streamlit as st
import gspread
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import pytz

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN VISUAL
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Panel de Control V27",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh cada 60 segundos
st.autorefresh(interval=60000, limit=None, key="refresh")

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
# 2. CARGA Y MAPEO DE DATOS (CORREGIDO: get_all_values)
# -------------------------------------------------------------------
@st.cache_data(ttl=60) 
def get_data():
    try:
        # CONEXIÓN
        creds = st.secrets.get("google_credentials")
        if creds:
            gc = gspread.service_account_from_dict(creds)
        else:
            # Fallback local
            gc = gspread.service_account(filename='client_secret.json')
            
        sh = gc.open("Resultados_Planta").worksheet("Resultados_Hibridos_RF")
        
        # --- CORRECCIÓN CRÍTICA: Usar get_all_values() ---
        # Esto evita el error de "duplicate headers"
        raw_data = sh.get_all_values()
        
        if not raw_data or len(raw_data) < 2: 
            return pd.DataFrame(), False
        
        # La primera fila son los headers, el resto son datos
        headers = raw_data[0]
        data_rows = raw_data[1:]
        
        # Crear DataFrame manualmente
        df = pd.DataFrame(data_rows, columns=headers)

        # --- MAPEO DE NOMBRES ---
        # Estandarizamos nombres para que el dashboard siempre encuentre las columnas
        # El modelo V27/V28 guarda en este orden (ver modelo_unificado.py):
        # 0:timestamp, 1:ffa, 2:aceite, 3:naoh_real, 4:agua_real, 5:temp, 
        # 6:opt_naoh, 7:opt_agua, 8:pred_acidez, 9:waste_real, ...
        
        # Como los headers en el Excel pueden variar o estar vacíos, 
        # renombramos basado en lo que espera el dashboard si detectamos columnas clave.
        
        # Convertir a numérico forzoso
        cols_to_numeric = [
            'caudal_naoh', 'caudal_agua', 'temperatura', # Inputs Reales
            'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh', # Outputs Viejos
            'opt_naoh', 'opt_agua', # Outputs Nuevos (si usaste nombres cortos)
            'waste_pct', 'merma_teorica', 'pred_merma', 'pred_acidez'
        ]
        
        # Normalización de nombres (si el excel tiene nombres viejos o nuevos)
        if 'opt_hibrida_naoh_Lh' in df.columns:
            df.rename(columns={'opt_hibrida_naoh_Lh': 'opt_naoh'}, inplace=True)
        if 'opt_hibrida_agua_Lh' in df.columns:
            df.rename(columns={'opt_hibrida_agua_Lh': 'opt_agua'}, inplace=True)
        if 'sim_merma_TEORICA_L' in df.columns:
            df.rename(columns={'sim_merma_TEORICA_L': 'merma_teorica'}, inplace=True)
            
        # Conversión Numérica
        for c in df.columns:
            # Intentamos convertir todo lo que parezca número
            if c in cols_to_numeric or 'caudal' in c or 'pct' in c or 'ppm' in c:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

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
        # Calcular filas aproximadas (1 dato cada 10 min = 6 por hora)
        rows_to_show = horas * 6
        df_view = df.tail(rows_to_show)
    
    last = df_view.iloc[-1]

    # --- HEADER & ESTADO ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🛡️ Panel de Estabilidad Gianazza")
        time_str = last.name.strftime('%H:%M:%S') if isinstance(last.name, pd.Timestamp) else "N/A"
        st.markdown(f"**Última Actualización:** {time_str} | **Aceite:** {last.get('caudal_aceite', 0):.1f} kg/h")
    
    with c2:
        # Badge de Estado (Busca la columna 'estado_merma' o similar)
        estado_col = next((c for c in df.columns if 'estado' in c), None)
        estado = str(last[estado_col]) if estado_col else "N/A"
        
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

    # --- KPI ROW ---
    k1, k2, k3, k4 = st.columns(4)
    
    # 1. Merma Real vs Teórica
    col_waste = 'waste_pct' if 'waste_pct' in df.columns else 'real_waste'
    col_teo = 'merma_teorica'
    
    val_waste = last.get(col_waste, 0)
    val_teo = last.get(col_teo, 0)
    delta_merma = val_waste - val_teo
    
    k1.metric("Merma SCADA", f"{val_waste:.2f}%", 
              delta=f"{delta_merma:+.2f}% vs Gianazza", delta_color="inverse")
    
    # 2. Acidez
    k2.metric("Acidez Proyectada", f"{last.get('pred_acidez', 0):.3f}%", 
              delta="Target 0.035%", delta_color="off")

    # 3. Soda
    real_naoh = last.get('caudal_naoh', 0)
    opt_naoh = last.get('opt_naoh', 0)
    k3.metric("NaOH (Real / Sug)", f"{real_naoh:.1f} / {opt_naoh:.1f}", 
             delta=f"{opt_naoh - real_naoh:+.1f} L/h")
    
    # 4. Agua
    real_agua = last.get('caudal_agua', 0)
    opt_agua = last.get('opt_agua', 0)
    k4.metric("Agua (Real / Sug)", f"{real_agua:.1f} / {opt_agua:.1f}",
             delta=f"{opt_agua - real_agua:+.1f} L/h")

    # --- GRÁFICO PRINCIPAL: CONTROL DE MERMA ---
    st.subheader("📉 Dinámica de Merma (Realidad vs Estándar)")
    
    fig_merma = go.Figure()
    
    if col_teo in df_view.columns:
        # Zona Segura
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view[col_teo]*1.15, 
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view[col_teo],
            mode='lines', line=dict(width=0), fill='tonexty', 
            fillcolor='rgba(40, 167, 69, 0.1)', name='Zona Óptima'
        ))
        # Linea Teórica
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view[col_teo], 
            mode='lines', name='Objetivo Gianazza (L)', line=dict(color='#28a745', dash='dash')
        ))
    
    if col_waste in df_view.columns:
        fig_merma.add_trace(go.Scatter(
            x=df_view.index, y=df_view[col_waste], 
            mode='lines+markers', name='Merma SCADA', line=dict(color='#ffc107', width=3)
        ))

    fig_merma.update_layout(height=350, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_merma, use_container_width=True)

    # --- GRÁFICOS DE CONTROL (DOSIS) ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### 🟠 Control de Soda (NaOH)")
        fig_soda = go.Figure()
        if 'caudal_naoh' in df_view.columns:
            fig_soda.add_trace(go.Scatter(x=df_view.index, y=df_view['caudal_naoh'], name='Real', line=dict(color='#fd7e14')))
        if 'opt_naoh' in df_view.columns:
            fig_soda.add_trace(go.Scatter(x=df_view.index, y=df_view['opt_naoh'], name='Sugerido', line=dict(color='#fd7e14', dash='dot')))
        fig_soda.update_layout(height=250, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_soda, use_container_width=True)

    with c2:
        st.markdown("##### 💧 Control de Agua")
        fig_agua = go.Figure()
        if 'caudal_agua' in df_view.columns:
            fig_agua.add_trace(go.Scatter(x=df_view.index, y=df_view['caudal_agua'], name='Real', line=dict(color='#0dcaf0')))
        if 'opt_agua' in df_view.columns:
            fig_agua.add_trace(go.Scatter(x=df_view.index, y=df_view['opt_agua'], name='Sugerido', line=dict(color='#0dcaf0', dash='dot')))
        fig_agua.update_layout(height=250, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_agua, use_container_width=True)

else:
    st.info("Esperando datos... Ejecuta el modelo para generar registros.")
