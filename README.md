# Benchmark de modelos de aprendizaje automático para la detección de volatilidad extrema en el crudo West Texas Intermediate

**Autores:** Luis Espino y Jhoselyn Torres
**Tutor:** Prof. Pablo Hidalgo
**Programa:** Maestría en Data Science y Business Analytics
**Instituciones:** Universidad Metropolitana de Educación, Ciencia y Tecnología (UMECIT) e IMMUNE Technology Institute

---

## 1. Presentación de la investigación

Este repositorio reúne el código, la configuración experimental, los datos procesados y los resultados del Trabajo de Fin de Máster del mismo título. La investigación compara, bajo un protocolo temporal común, un modelo econométrico de volatilidad condicional y cuatro arquitecturas de redes neuronales recurrentes en la tarea de identificar, con un día de anticipación, episodios de volatilidad extrema en los futuros del crudo West Texas Intermediate (WTI).

La finalidad del repositorio es documental y de reproducibilidad: permite consultar los artefactos que sustentan los resultados reportados en la tesis y, si se desea, volver a ejecutar el flujo de trabajo. No constituye un sistema de alerta en operación ni aporta evidencia sobre rentabilidad.

## 2. Problema y objetivos

El problema se formula como una **clasificación binaria con horizonte de un día**: a partir de la información disponible en el día *t*, se determina si el día *t+1* pertenecerá a un régimen de volatilidad extrema. La investigación no predice la dirección del precio ni evalúa estrategias de inversión.

La investigación se articula en torno a tres objetivos generales:

1. Construir el conjunto de datos históricos del WTI y definir el marco de etiquetado de eventos de volatilidad extrema.
2. Implementar y calibrar los modelos GARCH y LSTM bajo un protocolo estandarizado y reproducible.
3. Contrastar su rendimiento mediante métricas sensibles al desbalance de clases y pruebas de significancia.

Las hipótesis verificadas en el manuscrito contrastan la ausencia de diferencias significativas entre GARCH y los modelos de aprendizaje profundo (H0), la superioridad de estos últimos en F1 (H1) y la equivalencia del modelo econométrico por parsimonia (H2).

## 3. Diseño metodológico

| Elemento | Especificación |
|---|---|
| Instrumento | Futuros continuos del WTI, símbolo `CL=F` (Yahoo Finance) |
| Período analizado | 22/09/2000 – 27/02/2026 (6.385 observaciones diarias) |
| Entrenamiento | Hasta el 31/12/2018 · 4.584 observaciones |
| Validación | 2019–2020 · 505 observaciones, 483 válidas tras excluir 22 ventanas contaminadas |
| Prueba (test) | 2021 – febrero de 2026 · 1.296 observaciones · 27 días extremos |
| Volatilidad | Desviación estándar de los retornos logarítmicos en ventana móvil de 21 días |
| Evento extremo | Volatilidad superior al percentil 95 del entrenamiento limpio: `0.04299581565948056` (≈ 4,30 %) |
| Horizonte | Etiqueta del día *t+1* con información hasta *t* |
| Desbalance | Ponderación de clases (`class_weight`); no se emplea sobremuestreo sintético |
| Semilla | 42 |

La partición es estrictamente cronológica. El percentil 95 que define el evento extremo se calcula únicamente con el conjunto de entrenamiento y se aplica sin recalibración a validación y prueba. Este umbral de volatilidad es distinto del umbral de decisión que cada modelo aplica a su puntuación para emitir una alerta (por ejemplo, una probabilidad de 0,99 en LSTM-Simple-Uni).

La serie descargada comienza el 23/08/2000; la primera observación del conjunto procesado es del 22/09/2000, una vez completada la ventana de 21 días.

## 4. Preparación y control de calidad de los datos

A partir de los precios de cierre se calculan los retornos logarítmicos y su volatilidad realizada de 21 días. La descarga original se conserva congelada en [`NB01_EDA/wti_raw_frozen.csv`](NB01_EDA/wti_raw_frozen.csv) para que el experimento no dependa de consultas posteriores a la fuente.

**Tratamiento del precio negativo del 20 de abril de 2020.** Ese día el contrato cerró en −37,63 USD, valor para el que el retorno logarítmico no está definido. El procedimiento aplicado es el siguiente:

- El precio original se conserva en la columna `wti_raw`.
- Para el cálculo se aplica un piso técnico de 0,01 USD (`clip_floor`).
- Ese piso genera dos retornos artificialmente extremos, que se identifican mediante la marca `bad_return`.
- Las 22 ventanas de volatilidad afectadas (20/04/2020 – 19/05/2020) se marcan con `bad_vol21` y se excluyen de la etiqueta.

El episodio no se elimina de la serie; se documenta y se impide que un artefacto de cálculo determine la definición de evento extremo.

## 5. Modelos evaluados

El benchmark formal comprende cinco modelos:

| Modelo | Descripción |
|---|---|
| GJR-GARCH(1,1)-t | Modelo econométrico con asimetría y errores t de Student, seleccionado por AIC/BIC entre cuatro especificaciones. Pronostica la volatilidad de *t+1* con ventana expansiva |
| LSTM-Simple-Uni | LSTM(64) → Dropout(0,2) → Dense(1, sigmoide). Entrada: `vol21_clean` |
| BiDir-LSTM-Uni | LSTM bidireccional (64 y 32 unidades). Entrada: `vol21_clean` |
| LSTM-Simple-Multi | Arquitectura simple con ocho variables de entrada |
| BiDir-LSTM-Multi | Arquitectura bidireccional con ocho variables de entrada |

Todas las redes reciben secuencias de 21 observaciones. Las ocho variables de la versión multivariada son `vol21_clean`, `returns`, `ret_abs_lag1`, `hl_spread`, `co_spread`, `var_rolling_252_lag1`, `ma_cross` y `volumen_ratio_21`.

Como referencias de comparación se emplean dos reglas sin entrenamiento, calibradas con el conjunto de entrenamiento:

- **B1 (referencia ingenua).** Emite una alerta cuando el retorno absoluto del día anterior supera el percentil 95 de los retornos absolutos de entrenamiento. Este percentil no es el P95 de la volatilidad realizada que define el evento extremo.
- **B5 (referencia cuantitativa).** Emite una alerta cuando el cociente entre la volatilidad de 21 días y el valor en riesgo móvil, ambos rezagados un día, supera el percentil 95 de ese cociente en entrenamiento (excedencia ajustada por riesgo; Christoffersen, 1998).

## 6. Protocolo de entrenamiento y validación

- **Escalado.** `StandardScaler` ajustado con el conjunto de entrenamiento y aplicado a validación y prueba.
- **Secuencias.** Ventanas de 21 días alineadas con la etiqueta de *t+1*; los pares que cruzan la frontera entre particiones se descartan.
- **Entrenamiento de las LSTM.** Optimizador Adam (tasa 10⁻³), entropía cruzada binaria, lotes de 32, parada temprana sobre la pérdida de validación (paciencia de 15 épocas) y orden temporal preservado (`shuffle=False`).
- **GJR-GARCH.** Estimación por máxima verosimilitud y reestimación diaria con historia estrictamente anterior a *t*.
- **Umbral de decisión.** Optimizado exclusivamente en validación y aplicado en prueba sin reoptimización.
- **Métricas.** F1 como métrica principal; precisión, recall, AUC-ROC y precisión media (AP, *Average Precision*) como complementarias. La AP se calcula con `average_precision_score` de scikit-learn y aparece rotulada como «AUC-PR» en los notebooks, las figuras y la configuración. La exactitud se descarta por el desbalance de clases.

Estas medidas reducen el riesgo de fuga de información; no constituyen una garantía absoluta de su ausencia.

## 7. Resultados empíricos

Resultados en el conjunto de prueba (2021 – febrero de 2026), con los umbrales de decisión calibrados en validación. Fuentes: [`config_estudio_FINAL.json`](config_estudio_FINAL.json), las predicciones de [`NB02_Benchmark/predicciones_nb02/`](NB02_Benchmark/predicciones_nb02) y, para B1 y B5, las salidas guardadas de NB01.

| Modelo | F1 | Precisión | Recall | AUC-ROC | AP | Secuencias |
|---|---:|---:|---:|---:|---:|---:|
| **LSTM-Simple-Uni** | **0,783** | 0,947 | 0,667 | 0,999 | 0,947 | 1.275 |
| BiDir-LSTM-Uni | 0,735 | 0,610 | 0,926 | 0,996 | 0,940 | 1.275 |
| BiDir-LSTM-Multi | 0,598 | 0,433 | 0,963 | 0,995 | 0,876 | 1.180 |
| LSTM-Simple-Multi | 0,397 | 0,253 | 0,926 | 0,989 | 0,848 | 1.180 |
| GJR-GARCH(1,1)-t | 0,000 | 0,000 | 0,000 | 0,987 | 0,730 | 1.295 |
| *B5 (referencia cuantitativa)* | 0,547 | 0,382 | 0,963 | 0,998 | — | 1.296 |
| *B1 (referencia ingenua)* | 0,207 | 0,150 | 0,333 | 0,768 | — | 1.296 |

Para LSTM-Simple-Uni, la matriz de confusión sobre 1.275 secuencias registra 18 verdaderos positivos, 1 falso positivo, 9 falsos negativos y 1.247 verdaderos negativos. El intervalo de confianza *bootstrap* al 95 % de su F1 es [0,628; 0,898] y se solapa con el de BiDir-LSTM-Uni [0,606; 0,843].

El valor F1 = 0,487 que la configuración registra para B1 corresponde a validación; en prueba es 0,207. Del mismo modo, B5 obtiene F1 = 0,742 en validación y 0,547 en prueba.

## 8. Análisis de los resultados

**Modelo con mejor F1.** En el protocolo y el período evaluados, LSTM-Simple-Uni obtuvo el mayor F1 del benchmark formal. Su punto operativo es conservador: 18 de las 19 alertas emitidas fueron correctas y se detectaron 18 de los 27 días extremos evaluados, de modo que nueve quedaron sin detectar. Esta afirmación se limita al benchmark formal y no se extiende a los modelos complementarios descritos en la sección 12.

**Precisión frente a recall.** Las variantes bidireccionales y multivariadas detectan más días extremos a costa de un número mayor de falsas alarmas. La elección entre ambos perfiles depende del coste relativo de cada tipo de error y no se resuelve en este trabajo.

**Desempeño del GJR-GARCH.** El modelo presenta una AUC-ROC elevada (0,987), es decir, ordena adecuadamente los días según su riesgo. Sin embargo, con el umbral de volatilidad calibrado en validación no emitió ninguna alerta en prueba, de ahí su F1 nulo. El resultado se asocia a la transferencia del umbral entre períodos con distinta distribución de la volatilidad; el diseño no permite atribuirlo exclusivamente al cambio de régimen.

**Capacidad discriminante y clasificación.** Los cinco modelos formales presentan AUC-ROC entre 0,987 y 0,999, mientras que su F1 varía entre 0,000 y 0,783. Una alta capacidad de ordenamiento no garantiza un buen clasificador una vez fijado el umbral.

**Precauciones de interpretación.** Los 27 positivos de la prueba son días extremos concentrados en marzo y abril de 2022, no 27 episodios independientes. El número de secuencias evaluadas difiere entre modelos: el GJR-GARCH pronostica desde el 04/01/2021, mientras que las LSTM comienzan el 02/02/2021 por la ventana de 21 días; los 27 días extremos están presentes en todas las muestras.

## 9. Estructura real del repositorio

```
├── 1._WTI_Preparacion_Datos.ipynb                       NB01 · datos, limpieza, variables, P95 y particiones
├── 2 _WTI_ Benchmark de Modelos_CPU__.ipynb             NB02-CPU · benchmark canónico
├── 2 _WTI_ Benchmark_Optimización Bayesiana GPU.ipynb   NB02-GPU · corrida experimental
├── 3A_WTI__Sentimiento_FinBERT.ipynb                    NB03A · módulo complementario
├── 3B_WTI_ Market Signals.ipynb                         NB03B · módulo complementario
├── config_estudio_FINAL.json                            Configuración congelada y resultados registrados
├── requirements.txt · requirements_eda.txt              Registro de versiones del entorno original (contenido equivalente)
├── utils/
│   └── figure_utils.py                                  Estilo y utilidades de figuras
├── NB01_EDA/                                            Datos congelados, datos procesados y figuras
├── NB02_Benchmark/                                      Modelos, predicciones y figuras del benchmark
├── Optimización_Bayesiana_GPU/                          Salidas de la corrida experimental
└── NB03_FinBERT/ · NB03B_MarketSignals/                 Salidas de los módulos complementarios
```

Accesos directos:

- Notebooks: [NB01](1._WTI_Preparacion_Datos.ipynb) · [NB02-CPU](2%20_WTI_%20Benchmark%20de%20Modelos_CPU__.ipynb) · [NB02-GPU](2%20_WTI_%20Benchmark_Optimizaci%C3%B3n%20Bayesiana%20GPU.ipynb) · [NB03A](3A_WTI__Sentimiento_FinBERT.ipynb) · [NB03B](3B_WTI_%20Market%20Signals.ipynb)
- Configuración: [`config_estudio_FINAL.json`](config_estudio_FINAL.json)
- Datos: [`NB01_EDA/`](NB01_EDA)
- Resultados del benchmark: [`NB02_Benchmark/predicciones_nb02/`](NB02_Benchmark/predicciones_nb02) · [`NB02_Benchmark/modelos_nb02/`](NB02_Benchmark/modelos_nb02) · [`NB02_Benchmark/Fig_NB02/`](NB02_Benchmark/Fig_NB02)
- Utilidades: [`utils/figure_utils.py`](utils/figure_utils.py)

**Nota sobre las rutas de la configuración.** `config_estudio_FINAL.json` registra las rutas históricas `data/raw/` y `data/processed/`. En el repositorio esos archivos se encuentran en `NB01_EDA/`. Los notebooks no leen esas claves: construyen las rutas de los CSV a partir de la variable `BASE_PATH` de su celda de configuración.

## 10. Consulta y ejecución en un equipo local (VS Code)

Los cinco notebooks conservan las salidas de la ejecución original, por lo que los resultados, las tablas y las figuras pueden consultarse sin ejecutar ninguna celda.

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Jcarol03/Benchmark-de-modelos-para-deteccion-volatilidad-WTI.git
   cd Benchmark-de-modelos-para-deteccion-volatilidad-WTI
   ```
2. Abrir la carpeta en VS Code (*Archivo → Abrir carpeta*) con las extensiones *Python* y *Jupyter* instaladas.
3. Abrir [NB01](1._WTI_Preparacion_Datos.ipynb) o [NB02-CPU](2%20_WTI_%20Benchmark%20de%20Modelos_CPU__.ipynb) para revisar el código y las salidas guardadas.

La ejecución es opcional y queda a criterio del usuario. Para ella se requiere un entorno de Python con las librerías que importan los notebooks. [`requirements.txt`](requirements.txt) registra las versiones del entorno original, organizadas por notebook; las del benchmark canónico son numpy 2.0.2, pandas 2.2.2, scipy 1.16.3, statsmodels 0.14.6, scikit-learn 1.6.1, matplotlib 3.10.0, seaborn 0.13.2, yfinance 0.2.66, arch 8.0.0 y tensorflow 2.19.0. NB02-CPU importa además `xgboost` en la sección de modelos complementarios, cuya versión no figura en ese registro. El archivo incluye también los paquetes del módulo de sentimiento, entre ellos una compilación de PyTorch específica para GPU, que no son necesarios para el benchmark.

Fuera de Google Colab, la celda de configuración de cada notebook asigna rutas relativas al directorio de trabajo, que en VS Code es habitualmente la carpeta del notebook. En consecuencia:

- NB02-CPU lee `config_estudio_FINAL.json` y `NB01_EDA/wti_volatilidad_MODEL_BASE.csv` desde la raíz del repositorio, y guarda figuras, predicciones y modelos en las carpetas `Fig_NB02/`, `predicciones_nb02/` y `modelos_nb02/` de esa raíz, no en `NB02_Benchmark/`.
- NB01 busca `wti_raw_frozen.csv` en el directorio de trabajo; si no lo encuentra, descarga la serie de Yahoo Finance, por lo que los datos pueden diferir de los congelados. Sus archivos de salida se escriben en el directorio de trabajo.

Los autores ejecutaron los notebooks en Google Colab. El comportamiento local descrito se deduce de la lectura del código y no se ha verificado mediante una ejecución completa.

## 11. Ejecución en Google Colab

La ejecución original se realizó en Google Colab con los archivos del proyecto almacenados en Google Drive. Al detectar ese entorno, la celda de configuración de cada notebook monta Google Drive y utiliza la carpeta de trabajo indicada en la variable `BASE_PATH`.

Para ejecutar los notebooks en Colab es necesario:

1. Copiar el contenido del repositorio a una carpeta de Google Drive, conservando la estructura de subcarpetas.
2. Abrir el notebook en Colab y autorizar el montaje de Drive cuando se solicite.
3. Ajustar, en la celda de configuración de rutas de cada notebook, el valor de `BASE_PATH` y de las rutas definidas junto a ella para que apunten a la carpeta elegida.
4. Ejecutar en orden NB01 y NB02-CPU. Los notebooks NB03A, NB03B y NB02-GPU son opcionales.

En esta modalidad, las salidas se guardan en la estructura de carpetas del proyecto (`NB01_EDA/`, `NB02_Benchmark/`, etc.). Una nueva ejecución sobrescribe la configuración y los artefactos existentes en esa carpeta.

## 12. Experimentos complementarios

Los siguientes componentes no forman parte del ranking formal ni de la verificación de hipótesis:

| Componente | Notebook | Alcance | Requisitos adicionales |
|---|---|---|---|
| GRU, TCN y XGBoost | NB02-CPU (sección «Modelos complementarios no canónicos») | Complementario | `xgboost` |
| Optimización bayesiana (Optuna) | NB02-GPU | Experimental | `optuna`; GPU recomendada |
| Sentimiento con FinBERT | NB03A | Complementario | `transformers`, `torch`, `gdeltdoc`; conexión a internet |
| Señales de mercado | NB03B | Complementario | `pandas-datareader`; clave propia de la API de EIA |

Resultados en prueba de los modelos complementarios de NB02-CPU, comprobados con los archivos de [`NB02_Benchmark/predicciones_nb02/`](NB02_Benchmark/predicciones_nb02):

| Modelo | F1 | VP | FP | FN | Observaciones |
|---|---:|---:|---:|---:|---:|
| XGBoost-Tabular | 0,946 | 26 | 2 | 1 | 1.289 |
| GRU-Simple-Uni | 0,900 | 27 | 6 | 0 | 1.275 |
| BiDir-GRU-Uni | 0,857 | 21 | 1 | 6 | 1.275 |
| TCN-Uni | 0,000 | 0 | 4 | 27 | 1.275 |

XGBoost-Tabular y las variantes GRU obtuvieron un F1 puntual superior al de LSTM-Simple-Uni. Estos resultados no modifican el benchmark formal por las razones que documenta el propio proyecto:

- **Alcance de las hipótesis.** Las hipótesis contrastan el modelo GARCH con las redes LSTM. NB02-CPU presenta estos modelos como un bloque complementario cuyo objetivo «no es redefinir el benchmark principal».
- **Carácter provisional de XGBoost.** El manuscrito clasifica su resultado como provisional por un riesgo de autocorrelación estructural no cuantificado: las variables rezagadas de `vol21_clean` comparten 20 de los 21 días de la ventana con el objetivo del día siguiente. La medición de ese efecto requiere un experimento con ventanas disjuntas que quedó fuera del alcance del estudio.
- **Evidencia estadística no concluyente.** Según NB02-CPU, la ventaja de XGBoost-Tabular frente a LSTM-Simple-Uni es significativa en la prueba de Diebold-Mariano, pero no alcanza significancia al 5 % en la prueba de McNemar.
- **Diseño no equivalente.** XGBoost es un modelo tabular con variables rezagadas explícitas, evaluado sobre 1.289 observaciones, frente a las 1.275 secuencias de las LSTM univariadas.

La clave de la API de EIA se lee de la variable de entorno `EIA_API_KEY` o de los secretos de Colab. El repositorio no contiene credenciales.

## 13. Limitaciones y alcance

- El conjunto de prueba contiene 27 días extremos, concentrados en marzo y abril de 2022.
- El modelo con mejor F1 omite 9 de esos 27 días.
- Los resultados son sensibles a los umbrales de decisión y a los cambios de régimen entre validación y prueba.
- Se emplea una única semilla y una única partición temporal; no se estima la variabilidad entre ejecuciones.
- El número de secuencias evaluadas difiere entre familias de modelos.
- La serie corresponde a un único contrato y a una única fuente de datos.

Los resultados describen el comportamiento histórico de los modelos en el período analizado. Su posible utilidad para el seguimiento del riesgo de mercado es potencial: no se ha evaluado su estabilidad en otros períodos, ni su rentabilidad, ni su despliegue en un entorno operativo.

## 14. Citación y atribución académica

> Espino, L. y Torres, J. (2026). *Benchmark de modelos de aprendizaje automático para la detección de volatilidad extrema en el crudo West Texas Intermediate* [Trabajo de Fin de Máster, Universidad Metropolitana de Educación, Ciencia y Tecnología].

**Datos de terceros.** Las series de precios proceden de Yahoo Finance; los titulares y enlaces de noticias, de GDELT; los inventarios, de la U.S. Energy Information Administration. Su reutilización está sujeta a las condiciones de cada proveedor.

**Condiciones de reutilización.** El repositorio no incluye todavía un archivo de licencia. Hasta que los autores lo incorporen, el material se publica con fines de consulta académica.

**Archivos serializados.** El repositorio contiene archivos `.pkl` y `.keras` generados por los autores. Como con cualquier archivo de este tipo, se recomienda cargarlos únicamente si se confía en su procedencia.

## 15. Estado de reproducibilidad

| Aspecto | Estado |
|---|---|
| Consulta de resultados, predicciones y figuras guardadas | Verificable directamente con los archivos del repositorio |
| Métricas de los modelos formales y complementarios | Coinciden con los recuentos de los archivos de predicciones |
| Ejecución en Google Colab con Google Drive | Realizada por los autores; requiere adaptar la carpeta de trabajo |
| Ejecución local (VS Code) | No verificada mediante una ejecución completa |
| NB02-GPU, NB03A y NB03B | Requieren GPU, conexión a internet o credenciales propias |
| Reproducción exacta de las métricas | No garantizada: depende de las versiones, del hardware y de la disponibilidad de la fuente de datos |
