import streamlit as st
import gspread
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de Optimización | Planta Refinería",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 2rem; }
    .vega-embed { background-color: transparent !important; }
    hr { margin-top: 1em; margin-bottom: 1em; opacity: 0.2; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 1. AUTENTICACIÓN Y CARGA DE DATOS (ROBUSTA)
# -------------------------------------------------------------------
@st.cache_data(ttl=600)
def cargar_datos():
    try:
        creds_dict = st.secrets.get("google_credentials")
        if not creds_dict:
            st.error("Error: Credenciales de Google no encontradas.")
            return pd.DataFrame(), False
            
        gc = gspread.service_account_from_dict(creds_dict)
        
        # --- CONFIGURACIÓN DE HOJA ---
        NOMBRE_SHEET = "Resultados_Planta"  
        NOMBRE_PESTAÑA = "Resultados_Hibridos_RF"  

        spreadsheet = gc.open(NOMBRE_SHEET)
        worksheet = spreadsheet.worksheet(NOMBRE_PESTAÑA)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        if df.empty:
            return df, False

        # --- AUTOCORRECCIÓN DE COLUMNAS FALTANTES ---
        # Si el Sheet no tiene los costos, los calculamos aquí (Precios: Sosa $0.5, Agua $0.1)
        if 'Costo_Real_Hora' not in df.columns:
            # Asegurarnos que los inputs sean numéricos primero
            for col in ['caudal_naoh_in', 'caudal_agua_in']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
            df['Costo_Real_Hora'] = (df['caudal_naoh_in'] * 0.5) + (df['caudal_agua_in'] * 0.1)

        if 'Costo_Optimo_Hora' not in df.columns:
             # Asegurarnos que los inputs sean numéricos
            for col in ['opt_hibrida_naoh_Lh', 'opt_hibrida_agua_Lh']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    df[col] = 0 # Si no hay predicción, asumimos 0
            
            df['Costo_Optimo_Hora'] = (df['opt_hibrida_naoh_Lh'] * 0.5) + (df['opt_hibrida_agua_Lh'] * 0.1)

        # Limpieza y Conversión de Tipos General
        columnas_num = [
            'ffa_pct_in', 'caudal_aceite_in', 
            'sim_acidez_HIBRIDA', 'sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L',
            'opt_hibrida_pred_acidez_hibrida'
        ]
        
        for col in columnas_num:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0 # Rellenar con 0 si falta alguna columna secundaria

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
    # Si falla la carga, mostramos un mensaje amigable
    st.sidebar.warning("Esperando datos...")
    df = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.caption("v2.2 - Auto-Healing Enabled")

# -------------------------------------------------------------------
# 3. DASHBOARD PRINCIPAL
# -------------------------------------------------------------------

if not df.empty:
    
    # --- CABECERA ---
    st.title("🏭 Centro de Control de Optimización")
    st.markdown("Monitoreo en tiempo real del comportamiento del Operador vs. IA Híbrida.")

    # --- TOP KPIS ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Cálculos seguros
    costo_real_total = df['Costo_Real_Hora'].sum()
    costo_opt_total = df['Costo_Optimo_Hora'].sum()
    ahorro = costo_real_total - costo_opt_total
    pct_ahorro = (ahorro / costo_real_total * 100) if costo_real_total > 0 else 0
    
    merma_real = df['sim_merma_ML_TOTAL'].mean() if 'sim_merma_ML_TOTAL' in df.columns else 0
    merma_teorica = df['sim_merma_TEORICA_L'].mean() if 'sim_merma_TEORICA_L' in df.columns else 0
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

    # --- TABS DE NAVEGACIÓN ---
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Impacto Económico", "🧠 Inteligencia (Bias Analysis)", "⚙️ Calidad", "🐛 Debug Data"])

    # TAB 1: ECONOMÍA
    with tab1:
        st.subheader("Evolución del Costo por Hora")
        if 'timestamp' in df.columns and 'Costo_Real_Hora' in df.columns:
            df_costos = df.melt('timestamp', value_vars=['Costo_Real_Hora', 'Costo_Optimo_Hora'], var_name='Tipo', value_name='Costo')
            df_costos['Tipo'] = df_costos['Tipo'].replace({'Costo_Real_Hora': 'Operador (Real)', 'Costo_Optimo_Hora': 'Modelo (Óptimo)'})

            chart_costos = alt.Chart(df_costos).mark_line(interpolate='step-after').encode(
                x=alt.X('timestamp', title='Tiempo'),
                y=alt.Y('Costo', title='Costo ($/h)'),
                color=alt.Color('Tipo', scale=alt.Scale(domain=['Operador (Real)', 'Modelo (Óptimo)'], range=['#1f77b4', '#2ca02c'])),
                tooltip=['timestamp', 'Tipo', 'Costo']
            ).properties(height=350, width='container').interactive()
            st.altair_chart(chart_costos, use_container_width=True)
            st.info(f"💡 **Insight:** Ahorro acumulado de **${ahorro:,.2f}** detectado.")
        else:
            st.warning("Datos de costo insuficientes para graficar.")

    # TAB 2: INTELIGENCIA
    with tab2:
        st.subheader("¿El modelo copia o piensa?")
        col_bias_1, col_bias_2 = st.columns(2)

        # A. SOSA (NaOH)
        with col_bias_1:
            st.markdown("#### 🧪 Dosificación de Sosa (NaOH)")
            if 'caudal_naoh_in' in df.columns and 'opt_hibrida_naoh_Lh' in df.columns:
                min_val = min(df['caudal_naoh_in'].min(), df['opt_hibrida_naoh_Lh'].min())
                max_val = max(df['caudal_naoh_in'].max(), df['opt_hibrida_naoh_Lh'].max())
                
                base = alt.Chart(df).encode(
                    x=alt.X('caudal_naoh_in', title='Operador (L/h)', scale=alt.Scale(domain=[min_val, max_val])),
                    y=alt.Y('opt_hibrida_naoh_Lh', title='Modelo (L/h)', scale=alt.Scale(domain=[min_val, max_val]))
                )
                scatter = base.mark_circle(size=60, opacity=0.6).encode(
                    tooltip=['timestamp', 'caudal_naoh_in', 'opt_hibrida_naoh_Lh']
                )
                chart_scatter = (scatter + base.mark_line(color='red', strokeDash=[5,5]).encode(x='caudal_naoh_in', y='caudal_naoh_in')).properties(height=300).interactive()
                st.altair_chart(chart_scatter, use_container_width=True)
                
                var_op = df['caudal_naoh_in'].var()
                var_mod = df['opt_hibrida_naoh_Lh'].var()
                ratio = var_mod/var_op if var_op > 0 else 0
                st.caption(f"**Dinamismo:** El modelo es **{ratio:.1f}x** más dinámico que el operador.")

        # B. AGUA
        with col_bias_2:
            st.markdown("#### 💧 Dosificación de Agua")
            if 'caudal_agua_in' in df.columns and 'opt_hibrida_agua_Lh' in df.columns:
                df_agua = df.melt('timestamp', value_vars=['caudal_agua_in', 'opt_hibrida_agua_Lh'], var_name='Origen', value_name='Caudal')
                df_agua['Origen'] = df_agua['Origen'].replace({'caudal_agua_in': 'Operador', 'opt_hibrida_agua_Lh': 'Modelo'})
                
                chart_agua = alt.Chart(df_agua).mark_line().encode(
                    x='timestamp', y='Caudal',
                    color=alt.Color('Origen', scale=alt.Scale(domain=['Operador', 'Modelo'], range=['#1f77b4', '#ff7f0e'])),
                    tooltip=['timestamp', 'Origen', 'Caudal']
                ).properties(height=300).interactive()
                st.altair_chart(chart_agua, use_container_width=True)
                
                diff_agua = (df['opt_hibrida_agua_Lh'] - df['caudal_agua_in']).mean()
                st.caption(f"**Diferencia:** El modelo sugiere usar **{abs(diff_agua):.2f} L/h** {'menos' if diff_agua < 0 else 'más'} de agua.")

    # TAB 3: CALIDAD
    with tab3:
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.subheader("Acidez Final")
            if 'sim_acidez_HIBRIDA' in df.columns:
                chart_acidez = alt.Chart(df).mark_line().encode(
                    x='timestamp', y='sim_acidez_HIBRIDA'
                ).properties(height=300).interactive()
                st.altair_chart(chart_acidez, use_container_width=True)
        
        with col_q2:
            st.subheader("Control de Merma")
            if 'sim_merma_ML_TOTAL' in df.columns:
                chart_merma = alt.Chart(df).mark_area(opacity=0.3).encode(
                    x='timestamp', y='sim_merma_ML_TOTAL'
                ).properties(height=300).interactive()
                st.altair_chart(chart_merma, use_container_width=True)

    # TAB 4: DEBUG (Solo visible si hay problemas)
    with tab4:
        st.write("### 🛠️ Diagnóstico de Datos")
        st.write("Columnas detectadas en Google Sheets:", list(df.columns))
        st.write("Muestra de datos:", df.head())

else:
    st.info("⏳ Conectando a Google Sheets... (Si esto tarda, revisa las credenciales)")
