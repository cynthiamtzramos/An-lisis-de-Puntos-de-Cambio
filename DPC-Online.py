import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import ruptures as rpt
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Detección de puntos de cambio",
    page_icon="🔍",
    layout="centered",
)

# Custom
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #87cefa;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


st.markdown(
"""
<div style="text-align: justify;">

<h3>📊 Análisis de Puntos de Cambio</h3>

💡 <b>Propósito:</b> Esta implementación está diseñada para <b>uso educativo y de libre acceso</b>, con el objetivo de que estudiantes y profesionales puedan aprender y experimentar con técnicas de detección de puntos de cambio.  

La metodología y el código aquí presentados son desarrollados por <b>M.C. Cynthia Mtz Ramos</b>, quien conserva la autoría de esta implementación. Se agradece citarla si se utiliza en trabajos derivados.

Este sistema implementa <b>tres técnicas principales de detección de puntos de cambio</b>:

<ol>
<li><b>Ventaneo Móvil:</b> Analiza la serie de datos utilizando una ventana deslizante de tamaño fijo y detecta cambios basándose en un modelo seleccionado por el usuario.</li>
<li><b>Segmentación Binaria:</b> Divide la serie en segmentos optimizando un criterio de costo, con número de puntos de cambio definido manualmente.</li>
<li><b>PELT (Pruned Exact Linear Time):</b> Metodología óptima que minimiza un costo total y permite controlar la sensibilidad mediante una penalización.</li>
</ol>

<b>Nota sobre los modelos disponibles:</b>

<ul>
<li><b>l1:</b> Coste basado en la norma L1 (robusto a outliers).</li>
<li><b>l2:</b> Coste basado en la norma L2 (minimiza errores cuadrados).</li>
<li><b>rbf:</b> Radial Basis Function, captura cambios no lineales.</li>
<li><b>linear:</b> Ajuste lineal por segmentos.</li>
<li><b>normal:</b> Para series con distribución gaussiana.</li>
<li><b>ar:</b> Modela series temporales como un proceso autorregresivo.</li>
</ul>

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Elige un archivo", type=["txt", "xlsx"])
st.write("📌 Instrucciones: El archivo deberá de ser en formato .TXT o .XLSX. Puede contener:")
st.write("- 1 columna: será interpretada como **Y**, y el eje X será generado automáticamente (empezando en 1).")
st.write("- 2 columnas: la primera se tomará como **X** y la segunda como **Y**.")

if uploaded_file is not None:
    data = None
    if uploaded_file.name.endswith(".txt"):
        try:
            
            data = pd.read_csv(uploaded_file, delimiter=None, engine="python")
        except Exception as e:
            st.error(f"❌ No se pudo leer el archivo TXT: {e}")
            data = None
    elif uploaded_file.name.endswith(".xlsx"):
        try:
            data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ No se pudo leer el archivo Excel: {e}")
            data = None

    
    if data is not None:
        try:
            numeric_data = data.apply(pd.to_numeric, errors="raise")

            
            if numeric_data.shape[1] == 1:
                
                Y = numeric_data.iloc[:, 0].values
                X = np.arange(1, len(Y) + 1)  
                st.success("✅ Archivo cargado correctamente (1 columna detectada: se generó el eje X automáticamente empezando en 1).")

            elif numeric_data.shape[1] == 2:
                
                X = numeric_data.iloc[:, 0].values
                Y = numeric_data.iloc[:, 1].values
                st.success("✅ Archivo cargado correctamente (2 columnas detectadas: primera=X, segunda=Y).")

            else:
                st.error("❌ El archivo debe contener **1 o 2 columnas** de valores numéricos.")
                X, Y = None, None

            
            if "X" in locals() and "Y" in locals() and X is not None and Y is not None:
                
                test_sum = np.nansum(Y) + np.nansum(X)
                if np.isnan(test_sum) or np.isinf(test_sum):
                    raise ValueError("Los datos contienen valores no numéricos o inválidos.")

                
                preview_df = pd.DataFrame({"X": X[:10], "Y": Y[:10]})
                st.write("Vista previa de los datos (primeras 10 filas):")
                st.dataframe(preview_df)

                #⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘

                metodo = st.radio(
                    "Selecciona el método de detección de puntos de cambio:",
                    ["Ventaneo Móvil", "Segmentación binaria", "PELT"]
                )

                st.write(f"Has seleccionado el método: **{metodo}**")

                #⫘⫘⫘⫘⫘⫘
                
                if metodo == "Ventaneo Móvil":
                    model_options=["L1","l2", "rbf", "linear", "normal", "ar"]
                    selected_model=st.selectbox("Seleciona el tipo de modelo para Ventaneo Móvil:", model_options)

                    st.write(f"Has seleccionado el modelo **{selected_model}**")
                    
                    st.markdown("Parámetros")
                    n_points=len(X)
                    window_suggest= max(1, int(n_points/20))

                    window_size = st.number_input("Tamaño de ventana:", min_value=1, value=20)
                    st.caption(f"💡 Sugerencia de tamaño de ventana basada en la longitud de los datos: ~{window_suggest}")
                    
                    n_bkps=st.number_input("Número de Puntos de cambio", min_value=1, value=3)
                    
                    if st.button("Ejecutar Ventaneo Móvil"):
                        st.write(f"Ejecutando Ventaneo Móvil con modelo `{selected_model}`, window={window_size}, n_bkps={n_bkps}")

                        feature_matrix=np.column_stack((X,Y))

                        #Para RBF
                        if selected_model.lower()== "rbf":
                            scaler=StandardScaler()
                            feature_matrix=scaler.fit_transform(feature_matrix)

                        algo= rpt.Window(width=window_size, model=selected_model.lower()).fit(feature_matrix)
                        bkps=algo.predict(n_bkps=n_bkps)

                        if not bkps or (len(bkps)==1 and bkps[0]==len(X)):
                            st.warning("No se detectaron puntos de cambio")
                        else:
                            st.write(f"Indices de punto de cambio detectados: {bkps}")

                            fig, ax=plt.subplots(figsize=(12,6))
                            ax.plot(X, Y, color="blue", label="Y")
                            for i, bkpt in enumerate(bkps[:-1]):
                                ax.axvline(x=X[bkpt], color="red", linestyle="--", label="Breakpoint" if i == 0 else "")
                            ax.set_xlabel("X")
                            ax.set_ylabel("Y")
                            ax.set_title(f"Ventaneo Móvil ({selected_model}, window={window_size}, n_bkps={n_bkps})")
                            ax.grid(True)
                            ax.legend()
                            st.pyplot(fig)
    

                #⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘

                elif metodo == "Segmentación binaria":
                    
                    model_options=["L1","l2", "rbf", "linear", "normal", "ar"]
                    selected_model=st.selectbox("Seleciona el tipo de modelo para Ventaneo Móvil:", model_options)
                    st.write(f"Has seleccionado el modelo **{selected_model}**")
                    
                   
                    if "X" in locals() and "Y" in locals() and X is not None and Y is not None:
                        feature_matrix = np.column_stack((X, Y))

                        n_bkps = st.number_input("Número de puntos de cambio (n_bkps):", min_value=1, value=3)
            

                        if st.button("Ejecutar Segmentación Binaria"):
                            # Para RBF
                            if selected_model.lower() == "rbf":
                                scaler = StandardScaler()
                                feature_matrix = scaler.fit_transform(feature_matrix)

                            
                            algo = rpt.Binseg(model=selected_model.lower()).fit(feature_matrix)                       
                            bkps = algo.predict(n_bkps=n_bkps)

                            
                            if not bkps or (len(bkps) == 1 and bkps[0] == len(X)):
                                st.warning("⚠️ No se detectaron puntos de cambio.")
                            else:
                                st.write(f"Indices de puntos de cambio detectados: {bkps}")

                                
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.plot(X, Y, color="blue", label="Y")
                                for i, bkpt in enumerate(bkps[:-1]):
                                    ax.axvline(x=X[bkpt], color="red", linestyle="--", label="Breakpoint" if i == 0 else "")
                                ax.set_xlabel("X")
                                ax.set_ylabel("Y")
                                ax.set_title(f"Segmentación Binaria ({selected_model}, n_bkps={n_bkps})")
                                ax.grid(True)
                                ax.legend()
                                st.pyplot(fig)                            


                #⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘⫘
                #     
                elif metodo == "PELT":
                    
                    model_options = ["L1", "L2", "rbf", "linear", "normal", "ar"]
                    selected_model = st.selectbox("Selecciona el tipo de modelo para PELT:", model_options)
                    st.write(f"Has seleccionado el modelo **{selected_model}**")

                    if "X" in locals() and "Y" in locals() and X is not None and Y is not None:
                        feature_matrix = np.column_stack((X, Y))

                            
                        pen_value = st.number_input("Penalización (pen):", min_value=1, value=10, step=1)
                        st.markdown(
                            """
                            💡 **Sobre la penalización (pen):**  
                            La penalización controla cuán estricta es la detección de puntos de cambio.  
                            - Valores altos → detecta **menos puntos**, evitando sobreajuste.  
                            - Valores bajos → detecta **más puntos**, puede generar sobreajuste.  
                            """
                        )
                        if st.button("Ejecutar PELT"):
                                # Para RBF
                            if selected_model.lower() == "rbf":
                                scaler = StandardScaler()
                                feature_matrix = scaler.fit_transform(feature_matrix)

                                
                            algo = rpt.Pelt(model=selected_model.lower()).fit(feature_matrix)

                                
                            bkps = algo.predict(pen=pen_value)

                            if not bkps or (len(bkps) == 1 and bkps[0] == len(X)):
                                st.warning("⚠️ No se detectaron puntos de cambio.")
                            else:
                                st.write(f"Índices de puntos de cambio detectados: {bkps}")

                                    
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.plot(X, Y, color="blue", label="Y")
                                for i, bkpt in enumerate(bkps[:-1]):
                                        ax.axvline(x=X[bkpt], color="red", linestyle="--", label="Breakpoint" if i == 0 else "")
                                ax.set_xlabel("X")
                                ax.set_ylabel("Y")
                                ax.set_title(f"PELT ({selected_model}, pen={pen_value})")
                                ax.grid(True)
                                ax.legend()
                                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: El archivo debe contener únicamente valores numéricos.\nDetalles: {e}")
