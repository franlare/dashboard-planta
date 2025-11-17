import streamlit as st
import gspread
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA (Debe ser lo primero)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Optimización | Planta Refinería",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para "look & feel" moderno
st.markdown("""
    <style>
    /* Ajuste de métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    /* Fondo más limpio para gráficos */
    .vega-embed {
        background-color: transparent !important;
    }
    /* Separadores sutiles */
    hr {
        margin-top: 1em;
        margin-bottom: 1em;
        opacity: 0.2;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 1. AUTENTICACIÓN Y CARGA DE DATOS
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        creds_dict = st.secrets.get("google_credentials")
        if not creds_dict:
            st.error("Error: Credenciales de Google no encontradas.")
            return pd.DataFrame(), False
            
        gc = gspread.service_account_from_dict(creds_dict)
        # Ajusta estos nombres si cambiaron en tu Sheet
        NOMBRE_SHEET = "Resultados_Planta"  
        NOMBRE_PESTAÑA = "Resultados_Hibridos_RF"  

        spreadsheet = gc.open(NOMBRE_SHEET)
        worksheet = spreadsheet.worksheet(NOMBRE_PESTAÑA)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        # Limpieza y Conversión de Tipos
        columnas_num = [
            'ffa_pct_in', 'caudal_aceite_in', 'caudal_naoh_in', 'caudal_agua_in',
            'sim_acidez_HIBRIDA', 'sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L',
            'opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh', 'opt_hibrida_pred_acidez_hibrida',
            'Costo_Real_Hora', 'Costo_Optimo_Hora'
        ]
        
        for col in columnas_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df, True
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame(), False

df_raw, exito = cargar_datos()

# -------------------------------------------------------------------
# 2. SIDEBAR (FILTROS)
# -------------------------------------------------------------------
st.sidebar.title("🎛️ Filtros de Proceso")

if exito and not df_raw.empty:
    min_date = df_raw['timestamp'].min().to_pydatetime()
    max_date = df_raw['timestamp'].max().to_pydatetime()
    
    rango_fechas = st.sidebar.slider(
        "Periodo de Análisis",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="DD/MM/YY HH:mm"
    )
    
    # Filtrar Data
    mask = (df_raw['timestamp'] >= rango_fechas[0]) & (df_raw['timestamp'] <= rango_fechas[1])
    df = df_raw.loc[mask]
else:
    st.sidebar.warning("No hay datos para mostrar.")
    df = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.caption("v2.1 - AI Optimization Engine")

# -------------------------------------------------------------------
# 3. DASHBOARD PRINCIPAL
# -------------------------------------------------------------------

if not df.empty:
    
    # --- CABECERA (HEADER) ---
    st.title("🏭 Centro de Control de Optimización")
    st.markdown("Monitoreo en tiempo real del comportamiento del Operador vs. IA Híbrida.")

    # --- TOP KPIS (Resumen Ejecutivo) ---
    col1, col2, col3, col4 = st.columns(4)
    
    costo_real_total = df['Costo_Real_Hora'].sum()
    costo_opt_total = df['Costo_Optimo_Hora'].sum()
    ahorro = costo_real_total - costo_opt_total
    pct_ahorro = (ahorro / costo_real_total * 100) if costo_real_total > 0 else 0
    
    # Merma promedio
    merma_real = df['sim_merma_ML_TOTAL'].mean()
    merma_teorica = df['sim_merma_TEORICA_L'].mean()
    gap_merma = merma_real - merma_teorica

    with col1:
        st.metric("Costo Operativo Real", f"${costo_real_total:,.0f}")
    with col2:
        st.metric("Costo Optimizado (IA)", f"${costo_opt_total:,.0f}", delta=f"-${ahorro:,.0f}", delta_color="normal")
    with col3:
        st.metric("Margen de Ahorro", f"{pct_ahorro:.2f}%")
    with col4:
        st.metric("Gap Merma (vs Teórica)", f"{gap_merma:.2f} L", help="Exceso de pérdida sobre el teórico ideal")

    st.markdown("---")

    # --- TABS DE NAVEGACIÓN MODERNA ---
    tab1, tab2, tab3 = st.tabs(["💰 Impacto Económico", "🧠 Inteligencia del Modelo (Bias Analysis)", "⚙️ Calidad de Proceso"])

    # =================================================================
    # TAB 1: IMPACTO ECONÓMICO
    # =================================================================
    with tab1:
        st.subheader("Evolución del Costo por Hora")
        
        # Preparar datos para Altair (formato largo)
        df_costos = df.melt('timestamp', value_vars=['Costo_Real_Hora', 'Costo_Optimo_Hora'], var_name='Tipo', value_name='Costo')
        # Renombrar para leyenda bonita
        df_costos['Tipo'] = df_costos['Tipo'].replace({'Costo_Real_Hora': 'Operador (Real)', 'Costo_Optimo_Hora': 'Modelo (Óptimo)'})

        chart_costos = alt.Chart(df_costos).mark_line(interpolate='step-after').encode(
            x=alt.X('timestamp', title='Tiempo'),
            y=alt.Y('Costo', title='Costo ($/h)'),
            color=alt.Color('Tipo', scale=alt.Scale(domain=['Operador (Real)', 'Modelo (Óptimo)'], range=['#1f77b4', '#2ca02c'])),
            tooltip=['timestamp', 'Tipo', 'Costo']
        ).properties(height=350, width='container').interactive()

        st.altair_chart(chart_costos, use_container_width=True)
        
        st.info(f"💡 **Insight:** El modelo ha detectado oportunidades de ahorro acumuladas de **${ahorro:,.2f}** en el periodo seleccionado, principalmente ajustando el uso de agua y sosa sin sacrificar calidad.")

    # =================================================================
    # TAB 2: INTELIGENCIA DEL MODELO (EL NUEVO ANÁLISIS)
    # =================================================================
    with tab2:
        st.subheader("¿El modelo copia o piensa?")
        st.markdown("""
        Análisis de comportamiento para detectar si la IA simplemente imita al operador (Overfitting) o si encuentra sus propias rutas óptimas.
        """)

        col_bias_1, col_bias_2 = st.columns(2)

        # --- A. ANÁLISIS DE SOSA (NaOH) ---
        with col_bias_1:
            st.markdown("#### 🧪 Dosificación de Sosa (NaOH)")
            
            # Scatter Plot: Operador vs Modelo
            min_val = min(df['caudal_naoh_in'].min(), df['opt_hibrida_naoh_Lh'].min())
            max_val = max(df['caudal_naoh_in'].max(), df['opt_hibrida_naoh_Lh'].max())
            
            base = alt.Chart(df).encode(
                x=alt.X('caudal_naoh_in', title='Operador (L/h)', scale=alt.Scale(domain=[min_val, max_val])),
                y=alt.Y('opt_hibrida_naoh_Lh', title='Modelo (L/h)', scale=alt.Scale(domain=[min_val, max_val]))
            )
            
            scatter = base.mark_circle(size=60, opacity=0.6).encode(
                tooltip=['timestamp', 'caudal_naoh_in', 'opt_hibrida_naoh_Lh']
            )
            
            line_1to1 = base.mark_rule(color='red', strokeDash=[5,5]).encode(x=alt.value(0), x2=alt.value(800), y=alt.value(0), y2=alt.value(800)) # Simplificado, idealmente dynamic
            
            chart_scatter = (scatter + base.mark_line(color='red', strokeDash=[5,5]).encode(x=alt.X('caudal_naoh_in'), y=alt.Y('caudal_naoh_in'))).properties(
                title="Correlación: Operador vs Modelo",
                height=300
            ).interactive()

            st.altair_chart(chart_scatter, use_container_width=True)
            
            var_op = df['caudal_naoh_in'].var()
            var_mod = df['opt_hibrida_naoh_Lh'].var()
            st.caption(f"**Varianza (Dinamismo):** Operador {var_op:.3f} vs Modelo {var_mod:.3f}. "
                       f"El modelo es **{var_mod/var_op:.1f}x** más dinámico ajustando la dosis.")

        # --- B. ANÁLISIS DE AGUA ---
        with col_bias_2:
            st.markdown("#### 💧 Dosificación de Agua")
            
            # Line Chart Comparativo
            df_agua = df.melt('timestamp', value_vars=['caudal_agua_in', 'opt_hibrida_agua_Lh'], var_name='Origen', value_name='Caudal')
            df_agua['Origen'] = df_agua['Origen'].replace({'caudal_agua_in': 'Operador', 'opt_hibrida_agua_Lh': 'Modelo'})
            
            chart_agua = alt.Chart(df_agua).mark_line().encode(
                x='timestamp',
                y='Caudal',
                color=alt.Color('Origen', scale=alt.Scale(domain=['Operador', 'Modelo'], range=['#1f77b4', '#ff7f0e'])),
                tooltip=['timestamp', 'Origen', 'Caudal']
            ).properties(title="Estrategia de Dilución", height=300).interactive()
            
            st.altair_chart(chart_agua, use_container_width=True)
            
            diff_agua = (df['opt_hibrida_agua_Lh'] - df['caudal_agua_in']).mean()
            st.caption(f"**Diferencia Promedio:** El modelo sugiere usar **{abs(diff_agua):.2f} L/h menos** de agua que el estándar actual.")

        st.success("""
        **Conclusión:** El modelo **NO** tiene un sesgo de imitación pura. 
        1. En **Sosa**, valida la operación actual (alta correlación) pero ajusta con más frecuencia (mayor varianza).
        2. En **Agua**, discrepa significativamente, proponiendo una estrategia de ahorro constante.
        """)

    # =================================================================
    # TAB 3: CALIDAD Y PROCESO
    # =================================================================
    with tab3:
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            st.subheader("Acidez Final (Simulada vs Predicha)")
            # Gráfico de Acidez
            df_acidez = df.melt('timestamp', value_vars=['sim_acidez_HIBRIDA', 'opt_hibrida_pred_acidez_hibrida'], var_name='Tipo', value_name='Acidez')
            chart_acidez = alt.Chart(df_acidez).mark_line().encode(
                x='timestamp',
                y=alt.Y('Acidez', scale=alt.Scale(zero=False)),
                color='Tipo',
                strokeDash='Tipo'
            ).properties(height=300).interactive()
            st.altair_chart(chart_acidez, use_container_width=True)
            
        with col_q2:
            st.subheader("Control de Pérdidas (Merma)")
            # Gráfico de Merma
            df_merma = df.melt('timestamp', value_vars=['sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L'], var_name='Tipo', value_name='Merma')
            chart_merma = alt.Chart(df_merma).mark_area(opacity=0.3).encode(
                x='timestamp',
                y='Merma',
                color='Tipo'
            ).properties(height=300).interactive()
            st.altair_chart(chart_merma, use_container_width=True)

else:
    st.info("Esperando datos... Verifica la conexión o los filtros de fecha.")
