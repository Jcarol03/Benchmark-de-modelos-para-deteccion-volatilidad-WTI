# Benchmark de modelos de aprendizaje automático para la detección de volatilidad extrema en el crudo WTI

Código, configuración y resultados del Trabajo de Fin de Máster presentado para optar al grado de **Magíster en Data Science y Business Analytics** (programa en convenio UMECIT · IMMUNE Technology Institute).

- **Autores:** Luis Espino · Jhoselyn Torres
- **Tutor:** Prof. Pablo Hidalgo
- **Universidad:** Universidad Metropolitana de Educación, Ciencia y Tecnología (UMECIT), Panamá

## Qué hace este proyecto

Compara un modelo econométrico (GJR-GARCH) con redes neuronales recurrentes (LSTM) en una tarea de **clasificación binaria**: con la información disponible en el día `t`, anticipar si el día `t+1` será un día de **volatilidad extrema** en los futuros del WTI (`CL=F`).

- Un día es *extremo* cuando su volatilidad realizada de 21 días supera el **percentil 95 calculado solo con el conjunto de entrenamiento** (≈ 4,30 %).
- No predice la dirección del precio ni evalúa estrategias de inversión.
- Es un ejercicio académico: no existe un sistema de alertas en producción.

## Datos

| Elemento | Valor |
|---|---|
| Instrumento | Futuros continuos del WTI, `CL=F` (Yahoo Finance, vía `yfinance`) |
| Período | 22/09/2000 – 27/02/2026 (6.385 observaciones diarias) |
| Entrenamiento | hasta 31/12/2018 · 4.584 obs. |
| Validación | 2019–2020 · 505 obs. (483 válidas) |
| Test | 2021 – feb. 2026 · 1.296 obs. · 27 días extremos |

La descarga está congelada en `NB01_EDA/wti_raw_frozen.csv` para que el experimento sea reproducible.

**Precio negativo del 20/04/2020 (−USD 37,63):** el valor original se conserva (`wti_raw`); para calcular retornos logarítmicos se aplica un piso técnico de USD 0,01, y se marcan los 2 retornos y las 22 ventanas de volatilidad afectadas, que quedan fuera de la etiqueta.

## Modelos del benchmark formal

| Modelo | Descripción |
|---|---|
| GJR-GARCH(1,1)-t | Elegido por AIC/BIC entre 4 especificaciones; pronóstico a 1 paso con ventana expansiva |
| LSTM-Simple-Uni | LSTM(64) → Dropout(0,2) → Dense(1); entrada: `vol21_clean` |
| BiDir-LSTM-Uni | BiLSTM(64) → BiLSTM(32); entrada: `vol21_clean` |
| LSTM-Simple-Multi | Igual que la simple, con 8 variables |
| BiDir-LSTM-Multi | Igual que la bidireccional, con 8 variables |

Referencias de comparación: **B1** (|retorno t−1| > P95) y **B5** (vol21 t−1 / VaR > P95).
GRU, TCN y XGBoost son modelos **complementarios**, fuera del ranking formal.

## Resultados en test (2021 – feb. 2026)

Umbrales de decisión calibrados en validación y aplicados en test sin reoptimizar.

| Modelo | F1 | Precisión | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| **LSTM-Simple-Uni** | **0,783** | 0,947 | 0,667 | 0,999 | 0,947 |
| BiDir-LSTM-Uni | 0,735 | 0,610 | 0,926 | 0,996 | 0,940 |
| BiDir-LSTM-Multi | 0,598 | 0,433 | 0,963 | 0,995 | 0,876 |
| B5 (referencia cuantitativa) | 0,547 | 0,382 | 0,963 | 0,998 | — |
| LSTM-Simple-Multi | 0,397 | 0,253 | 0,926 | 0,989 | 0,848 |
| B1 (referencia ingenua) | 0,207 | 0,150 | 0,333 | 0,768 | — |
| GJR-GARCH(1,1)-t | 0,000 | 0,000 | 0,000 | 0,987 | 0,730 |

LSTM-Simple-Uni: 18 verdaderos positivos, 1 falso positivo, 9 falsos negativos y 1.247 verdaderos negativos sobre 1.275 secuencias. IC 95 % bootstrap del F1: [0,628; 0,898].

**Lectura:** casi todos los modelos ordenan bien el riesgo (AUC-ROC ≈ 0,99), pero eso no garantiza un buen clasificador operativo. El GJR-GARCH conserva alta capacidad discriminante y, aun así, su umbral calibrado en validación no produjo alertas correctas en test.

## Estructura del repositorio

```
├─ 1._WTI_Preparacion_Datos.ipynb                      NB01: datos, limpieza, variables, P95, particiones
├─ 2 _WTI_ Benchmark de Modelos_CPU__.ipynb            NB02: benchmark canónico (fuente de los resultados)
├─ 2 _WTI_ Benchmark_Optimización Bayesiana GPU.ipynb  Experimental (Optuna), fuera del ranking formal
├─ 3A_WTI__Sentimiento_FinBERT.ipynb                   Complementario, fuera del alcance formal
├─ 3B_WTI_ Market Signals.ipynb                        Complementario, fuera del alcance formal
├─ config_estudio_FINAL.json                           Configuración congelada y resultados registrados
├─ requirements.txt · requirements_eda.txt             Versiones de librerías
├─ utils/figure_utils.py                               Estilo de figuras
├─ NB01_EDA/                                           Datos procesados y figuras de NB01
├─ NB02_Benchmark/                                     Modelos, predicciones y figuras de NB02
├─ NB03_FinBERT/ · NB03B_MarketSignals/                Salidas de los módulos complementarios
└─ Optimización_Bayesiana_GPU/                         Salidas de la búsqueda experimental
```

## Cómo reproducir el benchmark

Los notebooks se ejecutaron en **Google Colab** (Python 3.10+), con semilla `42`.

1. Copia el contenido de este repositorio a tu Google Drive, en la carpeta
   `Mi unidad/2410VDSO Trabajo de Grado/Modelos/` (es la ruta base que esperan los notebooks; si usas otra, ajusta la ruta base en la primera celda de configuración).
2. Abre `1._WTI_Preparacion_Datos.ipynb` en Colab e instala las dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Ejecuta **NB01** completo. Si existe `NB01_EDA/wti_raw_frozen.csv`, se usan los datos congelados en lugar de descargarlos de nuevo. NB01 genera los CSV procesados y actualiza `config_estudio_FINAL.json`.
4. Ejecuta **NB02-CPU** (`2 _WTI_ Benchmark de Modelos_CPU__.ipynb`) completo, sin GPU. Produce los modelos, las predicciones y las métricas del benchmark canónico.

Librerías principales: `yfinance 0.2.66`, `pandas 2.2.2`, `numpy 2.0.2`, `scipy 1.16.3`, `statsmodels 0.14.6`, `scikit-learn 1.6.1`, `arch 8.0.0`, `tensorflow 2.19.0`, `matplotlib 3.10.0`, `seaborn 0.13.2`.
`xgboost` se usa solo en los modelos complementarios de NB02 y no figura en `requirements.txt`.
El notebook 3B necesita una clave propia de la API de EIA, configurada como secreto `EIA_API_KEY` en Colab; el repositorio no contiene ninguna clave.

## Medidas para reducir el riesgo de fuga de información

- Percentil 95 calculado solo con entrenamiento.
- `StandardScaler` ajustado en entrenamiento y aplicado a validación y test.
- Etiqueta `t+1` construida con `shift(-1)`; los pares que cruzan particiones se descartan.
- GARCH reestimado cada día con historia estrictamente anterior.
- Entrenamiento con `shuffle=False`; desbalance tratado con `class_weight` (sin SMOTE).
- Umbral de decisión optimizado solo en validación.

## Limitaciones

- Solo 27 días extremos en test, concentrados en marzo y abril de 2022.
- 9 falsos negativos del modelo ganador.
- Sensibilidad a los umbrales y a los cambios de régimen.
- Una única semilla y una única partición temporal.
- Los modelos univariados y multivariados evalúan un número distinto de secuencias (1.275 y 1.180).

## Cómo citar

Espino, L. y Torres, J. (2026). *Benchmark de modelos de aprendizaje automático para la detección de volatilidad extrema en el crudo West Texas Intermediate* [Trabajo de Fin de Máster]. Universidad Metropolitana de Educación, Ciencia y Tecnología (UMECIT).
