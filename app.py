import streamlit as st
import gspread
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# 1. AUTENTICACIÓN Y ACCESO A GOOGLE SHEETS
# -------------------------------------------------------------------

@st.cache_data(ttl=600) # Cachea los datos por 10 minutos (600 seg)
def cargar_datos():
    try:
        creds_dict = st.secrets["google_credentials"]
        gc = gspread.service_account_from_dict(creds_dict)
        
        NOMBRE_SHEET = "Resultados_Planta" 
        NOMBRE_PESTAÑA = "Resultados_Hibridos_RF" 

        spreadsheet = gc.open(NOMBRE_SHEET)
        worksheet = spreadsheet.worksheet(NOMBRE_PESTAÑA)

        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        # --- LIMPIEZA DE DATOS (CRÍTICO) ---
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
            else:
                st.warning(f"Advertencia: No se encontró la columna '{col}'.")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True) 
        else:
            st.error("Error crítico: No se encontró la columna 'timestamp'.")
            return pd.DataFrame(), False

        df.dropna(inplace=True)
        return df, True

    except Exception as e:
        st.error(f"Ocurrió un error cargando datos: {e}")
        return pd.DataFrame(), False

# Cargar los datos
df, data_loaded_successfully = cargar_datos()

# -------------------------------------------------------------------
# 2. INTERFAZ DE STREAMLIT (¡REDISEÑADA!)
# -------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("📊 Dashboard de Control y Optimización (Modelo Híbrido)")

if data_loaded_successfully and not df.empty:

    # -------------------------------------------------
    # BARRA LATERAL (SIDEBAR)
    # -------------------------------------------------
    st.sidebar.header("Filtros de Análisis")
    
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Selecciona el rango de fechas:",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

    st.sidebar.header("Límites de Especificación (Cpk)")
    usl = st.sidebar.number_input("Acidez - Límites Superior (USL)", value=0.04, format="%.3f")
    lsl = st.sidebar.number_input("Acidez - Límite Inferior (LSL)", value=0.015, format="%.3f")

    st.sidebar.header("Costos de Insumos")
    costo_soda = st.sidebar.number_input("Costo Soda ($/L)", value=0.5, format="%.2f")
    costo_agua = st.sidebar.number_input("Costo Agua ($/L)", value=0.1, format="%.2f")

    # Aplicar filtros al DataFrame
    df_filtrado = df[
        (df.index >= start_date) & 
        (df.index <= end_date)
    ]
    
    if df_filtrado.empty:
        st.warning("No hay datos para el rango de fechas seleccionado.")
    else:
        
        # -------------------------------------------------
        # CÁLCULOS PREVIOS (Para limpiar el código)
        # -------------------------------------------------
        
        # --- Errores de Dosificación ---
        df_filtrado['Error_Dosificacion_Soda'] = df_filtrado['caudal_naoh_in'] - df_filtrado['opt_hibrida_naoh_Lh']
        df_filtrado['Error_Dosificacion_Agua'] = df_filtrado['caudal_agua_in'] - df_filtrado['opt_hibrida_agua_Lh']
        mae_soda = np.mean(np.abs(df_filtrado['Error_Dosificacion_Soda']))
        mae_agua = np.mean(np.abs(df_filtrado['Error_Dosificacion_Agua']))

        # --- Acidez y Cpk ---
        acidez_data = df_filtrado['sim_acidez_HIBRIDA'].dropna()
        media = acidez_data.mean()
        std_dev = acidez_data.std()
        cpk = 0
        if std_dev > 0:
            cpu = (usl - media) / (3 * std_dev)
            cpl = (media - lsl) / (3 * std_dev)
            cpk = min(cpu, cpl)

        # --- Costos ---
        df_filtrado['Costo_Real_Hora'] = (df_filtrado['caudal_naoh_in'] * costo_soda) + (df_filtrado['caudal_agua_in'] * costo_agua)
        df_filtrado['Costo_Optimo_Hora'] = (df_filtrado['opt_hibrida_naoh_Lh'] * costo_soda) + (df_filtrado['opt_hibrida_agua_Lh'] * costo_agua)
        costo_total_real = df_filtrado['Costo_Real_Hora'].sum()
        costo_total_optimo = df_filtrado['Costo_Optimo_Hora'].sum()
        ahorro_potencial = costo_total_real - costo_total_optimo

        # --- Merma ---
        df_filtrado['Merma_Extra_ML'] = df_filtrado['sim_merma_ML_TOTAL'] - df_filtrado['sim_merma_TEORICA_L']
        merma_extra_media = df_filtrado['Merma_Extra_ML'].mean()
        
        # -------------------------------------------------
        # ¡NUEVO DISEÑO CON PESTAÑAS!
        # -------------------------------------------------
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Resumen Gerencial", 
            "🎯 Análisis de Dosificación", 
            "🔬 Calidad de Producto", 
            "💸 Costos y Merma", 
            "🗃️ Datos Crudos"
        ])

        # --- Pestaña 1: Resumen Gerencial (KPIs) ---
        with tab1:
            st.subheader("Indicadores Clave de Rendimiento (KPIs)")
            
            col1, col2, col3 = st.columns(3)
            
            # KPI de Costo
            col1.metric("Ahorro Potencial Perdido", f"${ahorro_potencial:,.2f}", 
                         help="Costo extra pagado por no seguir la dosificación óptima en el período.",
                         delta_color="inverse")
            
            # KPI de Calidad
            cpk_text = f"{cpk:.2f}"
            if cpk < 0.7: cpk_text += " 🔴 (No Capaz)"
            elif cpk < 1.33: cpk_text += " 🟡 (Aceptable)"
            else: cpk_text += " 🟢 (Capaz)"
            col2.metric("Capacidad de Proceso (Cpk)", cpk_text,
                         help="Mide qué tan bien el proceso se ajusta a los límites de especificación de acidez.")
            
            # KPI de Eficiencia
            col3.metric("Merma Operativa Extra (Promedio)", f"{merma_extra_media:.3f}%",
                         help="Merma promedio predicha por ML por encima de la merma teórica.")

            st.divider()
            
            st.subheader("Errores de Seguimiento (Promedio)")
            col4, col5, col6 = st.columns(3)
            col4.metric("Error Absoluto Soda (MAE)", f"{mae_soda:.2f} L/h")
            col5.metric("Error Absoluto Agua (MAE)", f"{mae_agua:.2f} L/h")
            col6.metric("N° de Muestras Analizadas", f"{len(df_filtrado)}")

        # --- Pestaña 2: Análisis de Dosificación (Error) ---
        with tab2:
            st.subheader("Análisis de Error: Dosificación de Soda")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Seguimiento Real vs. Óptimo (Soda)")
                st.line_chart(df_filtrado, y=['caudal_naoh_in', 'opt_hibrida_naoh_Lh'])
            with col2:
                st.markdown("##### Distribución del Error (Soda)")
                hist_soda_error, bins_soda = np.histogram(df_filtrado['Error_Dosificacion_Soda'].dropna(), bins=20)
                st.bar_chart(pd.DataFrame(hist_soda_error, index=bins_soda[:-1]))

            st.divider()
            
            st.subheader("Análisis de Error: Dosificación de Agua")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("##### Seguimiento Real vs. Óptimo (Agua)")
                st.line_chart(df_filtrado, y=['caudal_agua_in', 'opt_hibrida_agua_Lh'])
            with col4:
                st.markdown("##### Distribución del Error (Agua)")
                hist_agua_error, bins_agua = np.histogram(df_filtrado['Error_Dosificacion_Agua'].dropna(), bins=20)
                st.bar_chart(pd.DataFrame(hist_agua_error, index=bins_agua[:-1]))

        # --- Pestaña 3: Calidad de Producto ---
        with tab3:
            st.subheader("Análisis de Acidez Final (%FFA)")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Acidez Final Simulada (Híbrida)")
                st.line_chart(df_filtrado, y='sim_acidez_HIBRIDA')
            with col2:
                st.markdown("##### Distribución de Acidez Final")
                hist_acidez, bins_acidez = np.histogram(acidez_data, bins=20)
                st.bar_chart(pd.DataFrame(hist_acidez, index=bins_acidez[:-1]))

            st.markdown("##### Gráfico de Control de Acidez (SPC)")
            spc_df = pd.DataFrame({
                'Acidez': acidez_data,
                'Media': media,
                'Límite Superior (UCL)': media + (3 * std_dev),
                'Límite Inferior (LCL)': media - (3 * std_dev)
            })
            st.line_chart(spc_df)
            
            st.divider()
            
            st.subheader("Análisis de Jabones Finales (ppm)")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("##### Jabones Simulados (Híbrido)")
                st.line_chart(df_filtrado, y='sim_jabones_HIBRIDO')
            with col4:
                st.markdown("##### Distribución de Jabones")
                hist_jabon, bins_jabon = np.histogram(
                    df_filtrado['sim_jabones_HIBRIDO'].dropna(), bins=20
                )
                st.bar_chart(pd.DataFrame(hist_jabon, index=bins_jabon[:-1]))
        
        # --- Pestaña 4: Costos y Merma ---
        with tab4:
            st.subheader("Análisis de Costo por Hora")
            st.line_chart(df_filtrado, y=['Costo_Real_Hora', 'Costo_Optimo_Hora'])
            
            # --- ¡LÍNEA 1 CORREGIDA! ---
            st.info(f"El 'Ahorro Potencial Perdido' en este período fue de ${ahorro_potencial:,.2f}.")
            
            st.divider()
            
            st.subheader("Análisis de Merma (ML vs. Teórica)")
            st.line_chart(df_filtrado, y=['sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L'])
            
            # --- ¡LÍNEA 2 CORREGIDA! ---
            st.info(f"La 'Merma Operativa Extra' (diferencia entre ambas líneas) promedió {merma_extra_media:.3f}%.")

        # --- Pestaña 5: Datos Crudos ---
        with tab5:
            st.subheader("Datos Crudos Filtrados")
            st.dataframe(df_filtrado)

else:
    if not data_loaded_successfully:
        st.error("La carga de datos falló. Revisa la configuración y el archivo de secretos.")
    elif df.empty and data_loaded_successfully:
        st.error("La hoja de Google Sheets está vacía o no se pudieron cargar datos (posiblemente por formato incorrecto o filtro).")
