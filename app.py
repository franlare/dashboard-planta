import streamlit as st
import gspread
import pandas as pd
import numpy as np # Asegúrate de que numpy esté importado

# -------------------------------------------------------------------
# 1. AUTENTICACIÓN Y ACCESO A GOOGLE SHEETS
# -------------------------------------------------------------------

@st.cache_data(ttl=600) # Cachea los datos por 10 minutos (600 seg)
def cargar_datos():
    try:
        creds_dict = st.secrets["google_credentials"]
        gc = gspread.service_account_from_dict(creds_dict)
        
        # -----------------
        # ¡CORREGIDO! Apuntando al nombre del archivo y la pestaña correctos
        NOMBRE_SHEET = "Resultados_Planta" 
        NOMBRE_PESTAÑA = "Resultados_Hibridos_RF" # <-- ¡CAMBIO 1!
        # -----------------

        spreadsheet = gc.open(NOMBRE_SHEET)
        worksheet = spreadsheet.worksheet(NOMBRE_PESTAÑA)

        data = worksheet.get_all_records()
        df = pd.DataFrame(data)

        # --- LIMPIEZA DE DATOS (CRÍTICO) ---
        # ¡CORREGIDO! Usando los nuevos nombres de columnas del modelo híbrido
        columnas_numericas = [
            'ffa_pct_in', 
            'fosforo_ppm_in', 
            'caudal_aceite_in', 
            'caudal_acido_in', 
            'caudal_naoh_in', 
            'caudal_agua_in', 
            'temperatura_in', 
            'sim_acidez_HIBRIDA',         # <-- ¡CAMBIO 2!
            'sim_jabones_HIBRIDO',        # <-- ¡CAMBIO 3!
            'opt_hibrida_naoh_Lh',      # <-- ¡CAMBIO 4!
            'opt_hibrida_agua_Lh',      # <-- ¡CAMBIO 5!
            'sim_merma_ML_TOTAL',       # <-- ¡NUEVO!
            'sim_merma_TEORICA_L'       # <-- ¡NUEVO!
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

        # Eliminar filas donde cualquier conversión a número o fecha falló
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
st.title("📊 Dashboard de Control y Optimización de Neutralización (Modelo Híbrido)")

if data_loaded_successfully and not df.empty:

    # -------------------------------------------------
    # BARRA LATERAL (SIDEBAR) CON FILTROS
    # -------------------------------------------------
    st.sidebar.header("Filtros de Análisis")
    
    # Filtro de Rango de Fechas (usa el índice que creamos)
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

    # --- NUEVOS INPUTS EN SIDEBAR ---
    st.sidebar.header("Límites de Especificación (Cpk)")
    # ! CAMBIA ESTOS VALORES POR DEFECTO !
    usl = st.sidebar.number_input("Acidez - Límite Superior (USL)", value=0.04)
    lsl = st.sidebar.number_input("Acidez - Límite Inferior (LSL)", value=0.015)

    st.sidebar.header("Costos de Insumos")
    costo_soda = st.sidebar.number_input("Costo Soda ($/L)", value=0.5, format="%.2f")
    costo_agua = st.sidebar.number_input("Costo Agua ($/L)", value=0.1, format="%.2f")
    # --- FIN DE NUEVOS INPUTS ---


    # Aplicar filtros al DataFrame
    df_filtrado = df[
        (df.index >= start_date) & 
        (df.index <= end_date)
    ]
    
    if df_filtrado.empty:
        st.warning("No hay datos para el rango de fechas seleccionado.")
    else:
        # -------------------------------------------------
        # SECCIÓN 1: ANÁLISIS DE ERROR - DOSIFICACIÓN DE SODA
        # -------------------------------------------------
        st.header("1. Análisis de Error: Dosificación de Soda (Óptima vs. Real)")
        
        # ¡CORREGIDO!
        df_filtrado['Error_Dosificacion_Soda'] = df_filtrado['caudal_naoh_in'] - df_filtrado['opt_hibrida_naoh_Lh']
        
        col1, col2, col3 = st.columns(3)
        mae_soda = np.mean(np.abs(df_filtrado['Error_Dosificacion_Soda']))
        col1.metric("Error Absoluto Medio (MAE)", f"{mae_soda:.2f} L/h")
        
        bias_soda = np.mean(df_filtrado['Error_Dosificacion_Soda'])
        col2.metric("Error Medio (Bias)", f"{bias_soda:.2f} L/h", 
                     help="Positivo = La dosificación REAL de soda fue MAYOR que la óptima. Negativo = La dosificación REAL fue MENOR.")
        
        col3.metric("N° de Muestras", f"{len(df_filtrado)}")
        
        st.subheader("Seguimiento Temporal - Dosificación de Soda")
        # ¡CORREGIDO!
        st.line_chart(
            df_filtrado, 
            y=['caudal_naoh_in', 'opt_hibrida_naoh_Lh']
        )
        
        st.subheader("Correlación - Dosificación de Soda (Óptima vs. Real)")
        # ¡CORREGIDO!
        st.scatter_chart(
            df_filtrado,
            x='opt_hibrida_naoh_Lh', 
            y='caudal_naoh_in'      
        )

        # -------------------------------------------------
        # SECCIÓN 1.1: ANÁLISIS DE ERROR - DOSIFICACIÓN DE AGUA
        # -------------------------------------------------
        st.header("1.1. Análisis de Error: Dosificación de Agua (Óptima vs. Real)")
        
        # ¡CORREGIDO!
        df_filtrado['Error_Dosificacion_Agua'] = df_filtrado['caudal_agua_in'] - df_filtrado['opt_hibrida_agua_Lh']
        
        col1_agua, col2_agua, col3_agua = st.columns(3)
        mae_agua = np.mean(np.abs(df_filtrado['Error_Dosificacion_Agua']))
        col1_agua.metric("Error Absoluto Medio (MAE)", f"{mae_agua:.2f} L/h")
        
        bias_agua = np.mean(df_filtrado['Error_Dosificacion_Agua'])
        col2_agua.metric("Error Medio (Bias)", f"{bias_agua:.2f} L/h", 
                     help="Positivo = La dosificación REAL de agua fue MAYOR que la óptima. Negativo = La dosificación REAL fue MENOR.")
        
        col3_agua.metric("N° de Muestras", f"{len(df_filtrado)}")
        
        st.subheader("Seguimiento Temporal - Dosificación de Agua")
        # ¡CORREGIDO!
        st.line_chart(
            df_filtrado, 
            y=['caudal_agua_in', 'opt_hibrida_agua_Lh']
        )
        
        st.subheader("Correlación - Dosificación de Agua (Óptima vs. Real)")
        # ¡CORREGIDO!
        st.scatter_chart(
            df_filtrado,
            x='opt_hibrida_agua_Lh', 
            y='caudal_agua_in'      
        )


        # -------------------------------------------------
        # SECCIÓN 2: VARIABLES DE PROCESO (ENTRADA)
        # -------------------------------------------------
        st.header("2. Variables de Proceso (Entrada)")
        
        st.subheader("Seguimiento de Acidez (FFA) vs. Caudal de Aceite")
        st.line_chart(
            df_filtrado,
            y=['ffa_pct_in', 'caudal_aceite_in']
        )
        
        st.subheader("Correlación Acidez (FFA) vs. Caudal de Aceite")
        st.scatter_chart(
            df_filtrado,
            x='caudal_aceite_in',
            y='ffa_pct_in'
        )

        # -------------------------------------------------
        # SECCIÓN 3: ANÁLISIS DE RESULTADOS (SIMULADOS)
        # -------------------------------------------------
        st.header("3. Resultados Simulados por el Modelo")
        
        # Gráfico de Acidez Final Simulada
        st.subheader("Acidez Final Simulada (%FFA)")
        # ¡CORREGIDO!
        st.line_chart(
            df_filtrado,
            y='sim_acidez_HIBRIDA'
        )
        
        # Guardar la serie de datos de acidez para usarla en SPC y Cpk
        # ¡CORREGIDO!
        acidez_data = df_filtrado['sim_acidez_HIBRIDA'].dropna()

        # Histograma de Acidez Final Simulada
        st.subheader("Distribución de Acidez Final Simulada")
        hist_values_acidez, bin_edges_acidez = np.histogram(acidez_data, bins=20)
        hist_df_acidez = pd.DataFrame(hist_values_acidez, index=bin_edges_acidez[:-1])
        st.bar_chart(hist_df_acidez)
        
        # Gráfico de Jabones Simulados
        st.subheader("Jabones Simulados (ppm)")
        # ¡CORREGIDO!
        st.line_chart(
            df_filtrado,
            y='sim_jabones_HIBRIDO'
        )

        # Histograma de Jabones Simulados
        st.subheader("Distribución de Jabones Simulados")
        # ¡CORREGIDO!
        hist_values_jabon, bin_edges_jabon = np.histogram(
            df_filtrado['sim_jabones_HIBRIDO'].dropna(), bins=20
        )
        hist_df_jabon = pd.DataFrame(hist_values_jabon, index=bin_edges_jabon[:-1])
        st.bar_chart(hist_df_jabon)


        # -------------------------------------------------
        # SECCIÓN 4: SPC Y CAPACIDAD DE PROCESO (NUEVA)
        # -------------------------------------------------
        st.header("4. Análisis de Estabilidad (SPC) y Capacidad (Cpk)")

        # --- NUEVO GRÁFICO DE CONTROL (SPC) ---
        st.subheader("Gráfico de Control de Acidez Final (SPC)")
        
        # Calcular estadísticas para el gráfico de control
        media = acidez_data.mean()
        std_dev = acidez_data.std()
        
        # Límites de control (3-sigma)
        ucl = media + (3 * std_dev) # Upper Control Limit
        lcl = media - (3 * std_dev) # Lower Control Limit
        
        # Crear un DataFrame para el gráfico
        spc_df = pd.DataFrame({
            'Acidez': acidez_data,
            'Media': media,
            'Límite Superior (UCL)': ucl,
            'Límite Inferior (LCL)': lcl
        })
        
        # Graficar
        st.line_chart(spc_df)
        st.info(f"Media: {media:.3f} | Límite Superior (UCL): {ucl:.3f} | Límite Inferior (LCL): {lcl:.3f}")

        # --- NUEVO ANÁLISIS DE CAPACIDAD (Cpk) ---
        st.subheader("Análisis de Capacidad del Proceso (Cpk)")

        # Calcular Cpk
        if std_dev > 0: # Evitar división por cero si solo hay un dato
            cpu = (usl - media) / (3 * std_dev)
            cpl = (media - lsl) / (3 * std_dev)
            
            # El Cpk es el PEOR de los dos lados
            cpk = min(cpu, cpl)
            
            col1_cpk, col2_cpk = st.columns(2)
            col1_cpk.metric("Valor Cpk", f"{cpk:.2f}")
            
            # Interpretar el Cpk
            if cpk < 0.7:
                col2_cpk.error("No Capaz: El proceso produce defectos.")
            elif cpk < 1.33:
                col2_cpk.warning("Aceptable, pero requiere control estricto.")
            else:
                col2_cpk.success("¡Capaz! El proceso es robusto.")
            
            st.markdown(f"""
            - **Media del Proceso:** `{media:.3f}`
            - **Límites de Especificación:** `{lsl}` (LSL) a `{usl}` (USL)
            - *Un Cpk > 1.33 es considerado excelente.*
            """)
        else:
            st.warning("No se puede calcular Cpk: Se necesita más de un punto de dato o la desviación estándar es cero.")


        # -------------------------------------------------
        # SECCIÓN 5: ANÁLISIS DE COSTOS (NUEVA)
        # -------------------------------------------------
        st.header("5. Análisis de Costo/Beneficio")
        
        # Calcular costos por hora (usando los valores del sidebar)
        df_filtrado['Costo_Real_Hora'] = (df_filtrado['caudal_naoh_in'] * costo_soda) + \
                                         (df_filtrado['caudal_agua_in'] * costo_agua)
        
        # ¡CORREGIDO!
        df_filtrado['Costo_Optimo_Hora'] = (df_filtrado['opt_hibrida_naoh_Lh'] * costo_soda) + \
                                          (df_filtrado['opt_hibrida_agua_Lh'] * costo_agua)

        
        costo_total_real = df_filtrado['Costo_Real_Hora'].sum()
        costo_total_optimo = df_filtrado['Costo_Optimo_Hora'].sum()
        ahorro_potencial = costo_total_real - costo_total_optimo
        
        col1_costo, col2_costo, col3_costo = st.columns(3)
        col1_costo.metric("Costo Real Total (Período)", f"${costo_total_real:,.2f}")
        col2_costo.metric("Costo Óptimo Total (Período)", f"${costo_total_optimo:,.2f}")
        col3_costo.metric("Ahorro Potencial Perdido", f"${ahorro_potencial:,.2f}", 
                           delta_color="inverse")

        st.info("El 'Ahorro Potencial Perdido' es el costo extra pagado por no seguir la dosificación óptima en el período filtrado.")
        
        st.subheader("Costo por Hora (Real vs. Óptimo)")
        st.line_chart(df_filtrado, y=['Costo_Real_Hora', 'Costo_Optimo_Hora'])
        

        # -------------------------------------------------
        # SECCIÓN 6: ¡NUEVA! ANÁLISIS DE MERMA
        # -------------------------------------------------
        st.header("6. Análisis de Merma (ML vs. Teórica)")

        st.subheader("Seguimiento Temporal de Merma (%)")
        st.line_chart(
            df_filtrado,
            y=['sim_merma_ML_TOTAL', 'sim_merma_TEORICA_L']
        )
        st.info("La Merma ML TOTAL es la predicción del modelo (Random Forest) de lo que realmente ocurrirá. La Merma TEÓRICA L es el cálculo base (L=TL).")
        
        # Calcular el promedio de merma extra
        df_filtrado['Merma_Extra_ML'] = df_filtrado['sim_merma_ML_TOTAL'] - df_filtrado['sim_merma_TEORICA_L']
        merma_extra_media = df_filtrado['Merma_Extra_ML'].mean()
        
        st.metric("Merma Operativa Extra (Promedio)", f"{merma_extra_media:.3f}%",
                  help="Este es el % de merma promedio que el modelo ML predice POR ENCIMA de la merma teórica.")


        # -------------------------------------------------
        # SECCIÓN 7: DATOS CRUDOS (antes Sección 4)
        # -------------------------------------------------
        st.header("7. Datos Crudos Filtrados")
        st.dataframe(df_filtrado)

else:
    if not data_loaded_successfully:
        st.error("La carga de datos falló. Revisa la configuración y el archivo de secretos.")
    elif df.empty and data_loaded_successfully:
        st.error("La hoja de Google Sheets está vacía o no se pudieron cargar datos (posiblemente por formato incorrecto o filtro).")
