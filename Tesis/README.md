# Estimación del nivel de estrés hídrico en *Pinus radiata* mediante algoritmos de machine learning

Códigos de la tesis del Magíster en Data Science (Universidad Católica del Maule), autor: Leonel Torres.

El estudio compara seis algoritmos de machine learning (Ridge, Lasso, Random Forest, SVM, red neuronal y XGBoost) para estimar dos indicadores del estado de la vegetación, el **NDVI** y el **CWSI**, a partir de bandas e índices de vegetación derivados de imágenes multiespectrales obtenidas con drones.

## Contenido

| Archivo | Descripción |
|---|---|
| `pipeline_modelos_ndvi.R` | Preprocesamiento, validación cruzada, partición final y análisis de observaciones extremas para el NDVI. |
| `pipeline_modelos_cwsi_vf.R` | Mismo flujo para el CWSI, más el diagnóstico del rango de la variable, el filtrado al intervalo [0, 1] y el análisis descriptivo de ambos indicadores. |

Cada script es independiente y puede ejecutarse completo de principio a fin.

## Datos

Los datos pertenecen a Forestal Mininco y **no se incluyen en este repositorio**. Pueden solicitarse al profesor a cargo del proyecto.

Los scripts esperan un archivo Excel llamado `data indices estres.xlsx` con un registro por árbol y las columnas `codigo` (familia genética), `Site`, `Bloque`, las bandas espectrales, los índices de vegetación, las variables térmicas, `NDVI` y `CWSI`. Los nombres se normalizan con `janitor::clean_names()`.

Correspondencia entre los nombres de las columnas y los usados en la tesis:

| Columna | Nombre en la tesis |
|---|---|
| `codigo` | Código de la familia genética |
| `blue_480nm`, `green_560nm`, `red_670nm`, `red_edge_720nm`, `nir_840nm` | Bandas azul, verde, rojo, borde rojo e infrarrojo cercano |
| `ndre`, `ci_green`, `cire`, `gndvi`, `osavi`, `sr`, `rvi`, `ndwi` | NDRE, CI_green, CIRE, GNDVI, OSAVI, SR, RVI, NDWI |
| `cci_25` | CCI |
| `cci_28` | CCCI |
| `rendvi`, `ig` | RENDVI e IG (excluidos del modelamiento) |

## Requisitos

- R 4.4.1
- Paquetes: `tidyverse`, `readxl`, `openxlsx`, `janitor`, `glmnet`, `ranger`, `xgboost`, `e1071`, `nnet`, `caret`, `Metrics`, `patchwork`, `knitr`, `kableExtra`

```r
install.packages(c("tidyverse", "readxl", "openxlsx", "janitor", "glmnet", "ranger",
                   "xgboost", "e1071", "nnet", "caret", "Metrics", "patchwork",
                   "knitr", "kableExtra"))
```

## Ejecución

1. Copiar `data indices estres.xlsx` en una carpeta de trabajo.
2. Editar la variable `ruta` al inicio de cada script para que apunte a esa carpeta.
3. Ejecutar el script completo (`source("pipeline_modelos_ndvi.R")` o el equivalente para el CWSI).

## Diseño de la evaluación

- **Predictores.** NDVI: bandas azul, verde y borde rojo, más NDRE, CI_green y CIRE (se excluyen los índices que reutilizan el par NIR–Rojo). CWSI: cinco bandas y los índices de vegetación, excluyendo las variables térmicas, IG (función directa del CWSI) y RENDVI (idéntico a NDRE).
- **Filtrado del CWSI.** Se conservan solo las observaciones con CWSI en [0, 1].
- **Validación cruzada.** 10 particiones agrupadas por familia genética: todas las observaciones de una familia quedan en el mismo pliegue. Los seis modelos usan las mismas particiones.
- **Conjunto de prueba.** División 70/30 por familia, idéntica para los seis modelos.
- **Hiperparámetros.** En Ridge y Lasso, λ se elige con `cv.glmnet` (10 particiones internas agrupadas por familia, criterio `lambda.min`). Los demás modelos usan valores fijos, documentados en la sección 2.4.2 de la tesis y en los argumentos de cada función.
- **Escalado.** SVM y la red neuronal se centran y escalan dentro de cada partición con los parámetros del entrenamiento. `glmnet` estandariza internamente. Random Forest y XGBoost usan la escala original.
- **Métricas.** R² = 1 − SSE/SST, RMSE y MAE.
- **Comparación entre modelos.** Prueba de Friedman y Wilcoxon pareado con ajuste de Holm sobre las métricas por partición.
- **Observaciones extremas.** Residuos fuera de [Q1 − 1.5·RIQ, Q3 + 1.5·RIQ] en al menos un modelo. Las métricas se recalculan sin ellas, sobre el mismo subconjunto para todos los modelos.
- **Reproducibilidad.** Todas las particiones y modelos usan la semilla 42.
