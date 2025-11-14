import streamlit as st
import gspread
import pandas as pd
import numpy as np 

# -------------------------------------------------------------------
# 1. AUTENTICACIÓN Y ACCESO A GOOGLE SHEETS
# -------------------------------------------------------------------

@st.cache_data(ttl=600) 
def cargar_datos():
    try:
        creds_dict = st.secrets["google_credentials"]
        gc = gspread.service_account_from_dict(creds_dict)
        
        # -----------------
        # ! REEMPLAZA ESTOS VALORES !
        NOMBRE_SHEET = "Resultados_Planta" # <- ASEGÚRATE DE QUE SEA TU NOMBRE REAL
        NOMBRE_PESTAÑA = "Hoja 1" # <- Y ESTE EL DE TU PESTAÑA
        # -----------------

        spreadsheet = gc.open(NOMBRE_SHEET)
        worksheet = spreadsheet.worksheet(NOMBRE_PESTAÑA)

        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        # --- LIMPIEZA DE DATOS (CRÍTICO) ---
        columnas_numericas = [
            'ffa_pct_in', 
            'fosforo_ppm_in', 
            'caudal_aceite_in', 
            'caudal_acido_in', 
            'caudal_naoh_in', 
            'caudal_agua_in', 
            'temperatura_in', 
            'sim_acidez_final_pct', 
            'sim_jabones_ppm_fisico', 
            'opt_caudal_naoh_Lh', 
            'opt_caudal_agua_Lh' 
        ]
        
        for col in columnas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Advertencia: No se encontró la columna '{col}'.")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True) # Asegurarse de que esté ordenado por tiempo
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
# 2. CONSTRUCCIÓN DE LA INTERFAZ DE STREAMLIT
# -------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("📊 Dashboard de Control y Optimización de Neutralización")

if data_loaded_successfully and not df.empty:

    # -------------------------------------------------
    # BARRA LATERAL (SIDEBAR) CON FILTROS
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

    # --- INPUTS DE CÁLCULO EN SIDEBAR ---
    st.sidebar.header("Límites de Especificación (Cpk)")
    usl = st.sidebar.number_input("Acidez - Límite Superior (USL)", value=0.15)
    lsl = st.sidebar.number_input("Acidez - Límite Inferior (LSL)", value=0.05)

    st.sidebar.header("Costos de Insumos")
    # --- CAMBIOS AQUÍ ---
    costo_soda_kg = st.sidebar.number_input("Costo Soda ($/kg)", value=0.75, format="%.2f")
    densidad_soda = st.sidebar.number_input("Densidad Soda (kg/L)", value=1.52, format="%.2f")
    costo_agua = st.sidebar.number_input("Costo Agua ($/L)", value=0.1, format="%.2f")
    # --- FIN DE CAMBIOS ---


    # Aplicar filtros al DataFrame
    df_filtrado = df[
        (df.index >= start_date) & 
        (df.index <= end_date)
    ].copy() # Usamos .copy() para evitar warnings de pandas
    
    if df_filtrado.empty:
        st.warning("No hay datos para el rango de fechas seleccionado.")
    else:
        
        # --- CÁLCULOS PRELIMINARES DE COSTO ---
        # Convertir costo de $/kg a $/L usando la densidad
        costo_soda_litro = costo_soda_kg * densidad_soda
        
        # Calcular los costos POR HORA (para el gráfico de línea)
        df_filtrado['Costo_Real_Hora'] = (df_filtrado['caudal_naoh_in'] * costo_soda_litro) + \
                                         (df_filtrado['caudal_agua_in'] * costo_agua)
        
        df_filtrado['Costo_Optimo_Hora'] = (df_filtrado['opt_caudal_naoh_Lh'] * costo_soda_litro) + \
                                          (df_filtrado['opt_caudal_agua_Lh'] * costo_agua)

        # --- NUEVO CÁLCULO DE COSTO TOTAL PRECISO ---
        # 1. Calcular el delta de tiempo (duración de la fila) en horas
        #    diff() calcula la diferencia con la fila anterior.
        #    .dt.total_seconds() / 3600 lo convierte a horas
        df_filtrado['duracion_horas'] = df_filtrado.index.to_series().diff().dt.total_seconds() / 3600
        
        # 2. Rellenar el primer valor (que será NaN) con el promedio de duración
        #    Esto es una buena aproximación para la primera medición
        promedio_duracion = df_filtrado['duracion_horas'].mean()
        df_filtrado['duracion_horas'].fillna(promedio_duracion, inplace=True)
        
        # 3. Calcular el costo real de esa medición (Costo/Hora * Horas)
        df_filtrado['Costo_Real_Medicion'] = df_filtrado['Costo_Real_Hora'] * df_filtrado['duracion_horas']
        df_filtrado['Costo_Optimo_Medicion'] = df_filtrado['Costo_Optimo_Hora'] * df_filtrado['duracion_horas']

        # 4. Calcular los totales sumando los costos de cada medición
        costo_total_real = df_filtrado['Costo_Real_Medicion'].sum()
        costo_total_optimo = df_filtrado['Costo_Optimo_Medicion'].sum()
        ahorro_potencial = costo_total_real - costo_total_optimo
        # --- FIN DE CÁLCULO PRECISO ---


        # -------------------------------------------------
        # SECCIÓN 1: ANÁLISIS DE ERROR - DOSIFICACIÓN DE SODA
        # ... (Esta sección no cambia) ...
        # -------------------------------------------------
        st.header("1. Análisis de Error: Dosificación de Soda (Óptima vs. Real)")
        df_filtrado['Error_Dosificacion_Soda'] = df_filtrado['caudal_naoh_in'] - df_filtrado['opt_caudal_naoh_Lh']
        col1, col2, col3 = st.columns(3)
        mae_soda = np.mean(np.abs(df_filtrado['Error_Dosificacion_Soda']))
        col1.metric("Error Absoluto Medio (MAE)", f"{mae_soda:.2f} L/h")
        bias_soda = np.mean(df_filtrado['Error_Dosificacion_Soda'])
        col2.metric("Error Medio (Bias)", f"{bias_soda:.2f} L/h", help="Positivo = La dosificación REAL de soda fue MAYOR que la óptima. Negativo = La dosificación REAL fue MENOR.")
        col3.metric("N° de Muestras", f"{len(df_filtrado)}")
        st.line_chart(df_filtrado, y=['caudal_naoh_in', 'opt_caudal_naoh_Lh'])
        st.scatter_chart(df_filtrado, x='opt_caudal_naoh_Lh', y='caudal_naoh_in')


        # -------------------------------------------------
        # SECCIÓN 1.1: ANÁLISIS DE ERROR - DOSIFICACIÓN DE AGUA
        # ... (Esta sección no cambia) ...
        # -------------------------------------------------
        st.header("1.1. Análisis de Error: Dosificación de Agua (Óptima vs. Real)")
        df_filtrado['Error_Dosificacion_Agua'] = df_filtrado['caudal_agua_in'] - df_filtrado['opt_caudal_agua_Lh']
        col1_agua, col2_agua, col3_agua = st.columns(3)
        mae_agua = np.mean(np.abs(df_filtrado['Error_Dosificacion_Agua']))
        col1_agua.metric("Error Absoluto Medio (MAE)", f"{mae_agua:.2f} L/h")
        bias_agua = np.mean(df_filtrado['Error_Dosificacion_Agua'])
        col2_agua.metric("Error Medio (Bias)", f"{bias_agua:.2f} L/h", help="Positivo = La dosificación REAL de agua fue MAYOR que la óptima. Negativo = La dosificación REAL fue MENOR.")
        st.line_chart(df_filtrado, y=['caudal_agua_in', 'opt_caudal_agua_Lh'])
        st.scatter_chart(df_filtrado, x='opt_caudal_agua_Lh', y='caudal_agua_in')


        # -------------------------------------------------
        # SECCIÓN 2: VARIABLES DE PROCESO (ENTRADA)
        # ... (Esta sección no cambia) ...
        # -------------------------------------------------
        st.header("2. Variables de Proceso (Entrada)")
        st.line_chart(df_filtrado, y=['ffa_pct_in', 'caudal_aceite_in'])
        st.scatter_chart(df_filtrado, x='caudal_aceite_in', y='ffa_pct_in')


        # -------------------------------------------------
        # SECCIÓN 3: ANÁLISIS DE RESULTADOS (SIMULADOS)
        # ... (Esta sección no cambia) ...
        # -------------------------------------------------
        st.header("3. Resultados Simulados por el Modelo")
        
        st.subheader("Acidez Final Simulada (%FFA)")
        st.line_chart(df_filtrado, y='sim_acidez_final_pct')
        acidez_data = df_filtrado['sim_acidez_final_pct'].dropna()
        st.subheader("Distribución de Acidez Final Simulada")
        hist_values_acidez, bin_edges_acidez = np.histogram(acidez_data, bins=20)
        hist_df_acidez = pd.DataFrame(hist_values_acidez, index=bin_edges_acidez[:-1])
        st.bar_chart(hist_df_acidez)
        
        st.subheader("Jabones Simulados (ppm)")
        st.line_chart(df_filtrado, y='sim_jabones_ppm_fisico')
        st.subheader("Distribución de Jabones Simulados")
        hist_values_jabon, bin_edges_jabon = np.histogram(df_filtrado['sim_jabones_ppm_fisico'].dropna(), bins=20)
        hist_df_jabon = pd.DataFrame(hist_values_jabon, index=bin_edges_jabon[:-1])
        st.bar_chart(hist_df_jabon)


        # -------------------------------------------------
        # SECCIÓN 4: SPC Y CAPACIDAD DE PROCESO (Cpk)
        # ... (Esta sección no cambia) ...
        # -------------------------------------------------
        st.header("4. Análisis de Estabilidad (SPC) y Capacidad (Cpk)")

        st.subheader("Gráfico de Control de Acidez Final (SPC)")
        media = acidez_data.mean()
        std_dev = acidez_data.std()
        ucl = media + (3 * std_dev) 
        lcl = media - (3 * std_dev) 
        spc_df = pd.DataFrame({'Acidez': acidez_data, 'Media': media, 'Límite Superior (UCL)': ucl, 'Límite Inferior (LCL)': lcl})
        st.line_chart(spc_df)
        st.info(f"Media: {media:.3f} | Límite Superior (UCL): {ucl:.3f} | Límite Inferior (LCL): {lcl:.3f}")

        st.subheader("Análisis de Capacidad del Proceso (Cpk)")
        if std_dev > 0: 
            cpu = (usl - media) / (3 * std_dev)
            cpl = (media - lsl) / (3 * std_dev)
            cpk = min(cpu, cpl)
            col1_cpk, col2_cpk = st.columns(2)
            col1_cpk.metric("Valor Cpk", f"{cpk:.2f}")
            if cpk < 1.0:
                col2_cpk.error("No Capaz: El proceso produce defectos.")
            elif cpk < 1.33:
                col2_cpk.warning("Aceptable, pero requiere control estricto.")
            else:
                col2_cpk.success("¡Capaz! El proceso es robusto.")
            st.markdown(f"""- **Media del Proceso:** `{media:.3f}` | **Límites de Especificación:** `{lsl}` (LSL) a `{usl}` (USL)""")
        else:
            st.warning("No se puede calcular Cpk: Se necesita más de un punto de dato.")


        # -------------------------------------------------
        # SECCIÓN 5: ANÁLISIS DE COSTOS (ACTUALIZADA)
        # -------------------------------------------------
        st.header("5. Análisis de Costo/Beneficio")
        
        # Las métricas ahora usan los valores totales precisos
        col1_costo, col2_costo, col3_costo = st.columns(3)
        col1_costo.metric("Costo Real Total (Período)", f"${costo_total_real:,.2f}")
        col2_costo.metric("Costo Óptimo Total (Período)", f"${costo_total_optimo:,.2f}")
        col3_costo.metric("Ahorro Potencial Perdido", f"${ahorro_potencial:,.2f}", 
                           delta_color="inverse")

        st.info(f"""
        El 'Ahorro Potencial Perdido' es el costo extra pagado por no seguir la dosificación óptima en el período filtrado.
        Cálculo basado en: Costo Soda = ${costo_soda_kg}/kg ({costo_soda_litro:.2f} $/L) y Costo Agua = ${costo_agua}/L.
        """)
        
        # El gráfico de línea sigue mostrando el "costo por hora"
        # Esto es bueno para ver la *tasa* de gasto en el tiempo
        st.subheader("Tasa de Gasto por Hora (Real vs. Óptimo)")
        st.line_chart(df_filtrado, y=['Costo_Real_Hora', 'Costo_Optimo_Hora'])
        

        # -------------------------------------------------
        # SECCIÓN 6: DATOS CRUDOS
        # -------------------------------------------------
        st.header("6. Datos Crudos Filtrados")
        st.dataframe(df_filtrado)

else:
    if not data_loaded_successfully:
        st.error("La carga de datos falló. Revisa la configuración y el archivo de secretos.")
    elif df.empty and data_loaded_successfully:
        st.error("La hoja de Google Sheets está vacía o no se pudieron cargar datos (posiblemente por formato incorrecto o filtro).")

