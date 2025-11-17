import streamlit as st
import gspread
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# -------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO CSS (MODERN UI 2026)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Neural Plant Ops",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definición de Paleta "Cyber-Industrial"
COLOR_REAL = "#FF6B35"       # Naranja vibrante (Acción)
COLOR_OPTIMO = "#2D7DD2"     # Azul técnico (Referencia)
COLOR_BG_CHART = "transparent"
COLOR_TEXT_SEC = "#8D99AE"

# CSS Personalizado para Look & Feel Minimalista
st.markdown("""
    <style>
    /* Tipografía General - Fuente limpia */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Títulos y Métricas */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Estilo de Tarjetas de Métricas (KPI Cards) */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.3);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #8D99AE;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Ajuste de Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0 16px;
        background-color: transparent;
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(45, 125, 210, 0.1);
        color: #2D7DD2;
        border: 1px solid rgba(45, 125, 210, 0.3);
    }

    /* Remover Padding superior excesivo */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. LÓGICA DE DATOS (Optimizada)
# -------------------------------------------------------------------

@st.cache_data(ttl=600)
def cargar_datos():
    try:
        creds_dict = st.secrets.get("google_credentials")
        if not creds_dict:
            return pd.DataFrame(), False
            
        gc = gspread.service_account_from_dict(creds_dict)
        # Ajusta nombres según tu sheet real
        spreadsheet = gc.open("Resultados_Planta")  
        worksheet = spreadsheet.worksheet("Resultados_Hibridos_RF")  

        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        columnas_numericas = [
            'ffa_pct_in', 'fosforo_ppm_in', 'caudal_aceite_in',  
            'caudal_acido_in', 'caudal_naoh_in', 'caudal_agua_in',  
            'temperatura_in', 'sim_acidez_HIBRIDA', 'sim_jabones_HIBRIDO',  
            'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh',  
            'sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L'  
        ]
        
        for col in columnas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True) # Asegurar orden cronológico
        else:
            return pd.DataFrame(), False

        df.dropna(inplace=True)
        return df, True

    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return pd.DataFrame(), False

df, loaded = cargar_datos()

# -------------------------------------------------------------------
# 3. SIDEBAR MINIMALISTA
# -------------------------------------------------------------------

if loaded and not df.empty:
    with st.sidebar:
        st.markdown("### 🎛️ Control Center")
        
        # Filtros de Fecha Compactos
        min_date = df.index.min().date()
        max_date = df.index.max().date()
        
        date_range = st.date_input(
            "Periodo de Análisis",
            (min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        else:
            start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

        st.markdown("---")
        st.markdown("### ⚙️ Specs")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            usl = st.number_input("Max Acidez", 0.045, format="%.3f", step=0.001)
            lsl = st.number_input("Min Acidez", 0.025, format="%.3f", step=0.001)
        with col_s2:
            costo_soda = st.number_input("Costo Soda", 0.5, format="%.2f")
            costo_agua = st.number_input("Costo Agua", 0.1, format="%.2f")
            
    # Filtrado
    df_f = df[(df.index >= start_date) & (df.index <= end_date)].copy()

    # -------------------------------------------------------------------
    # 4. HEADER & KPI DASHBOARD (HEADS-UP DISPLAY)
    # -------------------------------------------------------------------
    
    # Cálculos Rápidos
    df_f['Costo_Real'] = (df_f['caudal_naoh_in'] * costo_soda) + (df_f['caudal_agua_in'] * costo_agua)
    df_f['Costo_Opt'] = (df_f['opt_hibrida_naoh_Lh'] * costo_soda) + (df_f['opt_hibrida_agua_Lh'] * costo_agua)
    ahorro = (df_f['Costo_Real'] - df_f['Costo_Opt']).sum()
    
    acidez = df_f['sim_acidez_HIBRIDA']
    sigma = acidez.std()
    mu = acidez.mean()
    cpk = min((usl - mu)/(3*sigma), (mu - lsl)/(3*sigma)) if sigma > 0 else 0
    
    merma_extra = (df_f['sim_merma_ML_TOTAL'] - df_f['sim_merma_TEORICA_L']).mean()

    # Encabezado
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("Neutralización // Dashboard Operativo")
        st.markdown(f"<span style='color:{COLOR_TEXT_SEC}'>Última actualización: {datetime.now().strftime('%H:%M:%S')} • {len(df_f)} muestras analizadas</span>", unsafe_allow_html=True)
    with col_h2:
        # Espacio para un botón de acción o estado del sistema
        st.caption("🟢 Sistema En Línea")

    st.markdown("---")

    # KPIs Principales (Estilo Tarjetas)
    k1, k2, k3, k4 = st.columns(4)
    
    k1.metric(
        "Eficiencia de Costos (Gap)",
        f"-${ahorro:,.0f}",
        delta="Pérdida Operativa" if ahorro > 0 else "Óptimo",
        delta_color="inverse"
    )
    
    k2.metric(
        "Capacidad (Cpk)",
        f"{cpk:.2f}",
        delta="Estable" if cpk > 1.33 else "Revisar Proceso",
        delta_color="normal" if cpk > 1.33 else "inverse"
    )
    
    k3.metric(
        "Merma Extra (ML)",
        f"{merma_extra:.3f}%",
        delta="Sobre Teórico",
        delta_color="off"
    )
    
    # Última acidez registrada
    last_acidez = acidez.iloc[-1] if not acidez.empty else 0
    k4.metric(
        "Acidez Final (Live)",
        f"{last_acidez:.3f}%",
        delta=f"{last_acidez - mu:.3f} vs Avg",
        delta_color="off"
    )

    st.markdown("### ") # Espaciador

    # -------------------------------------------------------------------
    # 5. PESTAÑAS DE ANÁLISIS VISUAL
    # -------------------------------------------------------------------
    
    tab_control, tab_calidad, tab_costos, tab_data = st.tabs([
        "🎛️ Control de Dosificación", "⚗️ Calidad de Producto", "💰 Finanzas", "📋 Raw Data"
    ])

    # --- FUNCION HELPER PARA GRAFICOS ALTAIR MODERNOS ---
    def make_modern_chart(data, y_real, y_opt, title, y_label, color_real, color_opt):
        base = alt.Chart(data.reset_index()).encode(x=alt.X('timestamp', axis=alt.Axis(title=None, format='%H:%M', grid=False)))
        
        # Área degradada para el valor Real (Look moderno)
        area_real = base.mark_area(
            line={'color': color_real},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color=color_real, offset=0),
                       alt.GradientStop(color='white', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            ),
            opacity=0.5
        ).encode(y=alt.Y(y_real, title=y_label, axis=alt.Axis(grid=False)))
        
        # Línea sólida para el valor Óptimo
        line_opt = base.mark_line(
            color=color_opt,
            strokeDash=[5, 5],
            strokeWidth=2
        ).encode(y=alt.Y(y_opt))
        
        # Tooltips combinados
        chart = (area_real + line_opt).properties(
            height=250,
            title=alt.TitleParams(text=title, font='Inter', fontSize=14, color=COLOR_TEXT_SEC)
        ).interactive()
        
        return chart

    with tab_control:
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("##### Soda Cáustica (NaOH)")
            chart_soda = make_modern_chart(
                df_f, 'caudal_naoh_in', 'opt_hibrida_naoh_Lh', 
                "", "L/h", COLOR_REAL, COLOR_OPTIMO
            )
            st.altair_chart(chart_soda, use_container_width=True)
            
            # Mini indicador de error
            err_soda = (df_f['caudal_naoh_in'] - df_f['opt_hibrida_naoh_Lh']).mean()
            st.caption(f"Desviación media: **{err_soda:.2f} L/h**")

        with col_c2:
            st.markdown("##### Agua de Proceso")
            chart_agua = make_modern_chart(
                df_f, 'caudal_agua_in', 'opt_hibrida_agua_Lh', 
                "", "L/h", "#00A8E8", COLOR_OPTIMO
            )
            st.altair_chart(chart_agua, use_container_width=True)
            
            err_agua = (df_f['caudal_agua_in'] - df_f['opt_hibrida_agua_Lh']).mean()
            st.caption(f"Desviación media: **{err_agua:.2f} L/h**")

    with tab_calidad:
        c_q1, c_q2 = st.columns([2, 1])
        
        with c_q1:
            st.markdown("##### Evolución de Acidez")
            # Gráfico de Acidez con banda de límites
            base_acid = alt.Chart(df_f.reset_index()).encode(x='timestamp')
            line_acid = base_acid.mark_line(color="#2A9D8F", strokeWidth=2).encode(
                y=alt.Y('sim_acidez_HIBRIDA', title='% FFA', scale=alt.Scale(domain=[lsl*0.8, usl*1.2]))
            )
            
            # Bandas de especificación
            rule_u = base_acid.mark_rule(color='red', strokeDash=[2,2]).encode(y=alt.datum(usl))
            rule_l = base_acid.mark_rule(color='red', strokeDash=[2,2]).encode(y=alt.datum(lsl))
            
            st.altair_chart((line_acid + rule_u + rule_l).interactive(), use_container_width=True)
            
        with c_q2:
            st.markdown("##### Histograma")
            # Histograma simple y limpio
            hist = alt.Chart(df_f).mark_bar(color="#2A9D8F", opacity=0.7).encode(
                x=alt.X('sim_acidez_HIBRIDA', bin=alt.Bin(maxbins=20), title='% FFA'),
                y=alt.Y('count()', title=None)
            ).properties(height=300)
            st.altair_chart(hist, use_container_width=True)

    with tab_costos:
        st.markdown("##### Acumulación de Costos (Real vs Óptimo)")
        
        # 1. Calcular acumulados
        df_f['cum_real'] = df_f['Costo_Real'].cumsum()
        df_f['cum_opt'] = df_f['Costo_Opt'].cumsum()
        
        # 2. Transformar datos a formato largo (Long Format) con Pandas
        # Esto evita el error de Altair transform_fold
        df_cum = df_f.reset_index().melt(
            id_vars=['timestamp'],
            value_vars=['cum_real', 'cum_opt'],
            var_name='Tipo_Costo',
            value_name='Costo_Acumulado'
        )
        
        # 3. Renombrar para que la leyenda se vea bonita
        df_cum['Tipo_Costo'] = df_cum['Tipo_Costo'].replace({
            'cum_real': 'Real (Acumulado)', 
            'cum_opt': 'Óptimo (Acumulado)'
        })

        # 4. Crear el gráfico corregido
        chart_cum = alt.Chart(df_cum).mark_line(strokeWidth=3).encode(
            x=alt.X('timestamp', title=None, axis=alt.Axis(format='%H:%M', grid=False)),
            y=alt.Y('Costo_Acumulado', title='Costo Acumulado ($)', axis=alt.Axis(grid=False)),
            color=alt.Color('Tipo_Costo', 
                            scale=alt.Scale(domain=['Real (Acumulado)', 'Óptimo (Acumulado)'], 
                                            range=[COLOR_REAL, COLOR_OPTIMO]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[
                alt.Tooltip('timestamp', format='%H:%M', title='Hora'),
                alt.Tooltip('Tipo_Costo', title='Tipo'),
                alt.Tooltip('Costo_Acumulado', format='$,.2f')
            ]
        ).properties(
            height=350,
            title=alt.TitleParams(text="Divergencia de Costos", font='Inter', fontSize=14, color=COLOR_TEXT_SEC)
        ).interactive()
        
        st.altair_chart(chart_cum, use_container_width=True)

    with tab_data:
        st.dataframe(
            df_f.style.highlight_max(axis=0, color='#ffcccc'), 
            use_container_width=True,
            height=400
        )

else:
    # Pantalla de Estado Vacío Minimalista
    st.container()
    st.markdown("""
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>💤 Esperando Datos</h3>
            <p>Conecta la fuente de datos o ajusta los filtros de fecha.</p>
        </div>
    """, unsafe_allow_html=True)

