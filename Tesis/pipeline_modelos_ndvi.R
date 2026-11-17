rm(list = ls())

ruta = "C:/Users/19990772/Desktop/Mg Data Science/03. semestre/Proyecto de Data Science I"
#ruta = "C:/Users/Leo Tactics/OneDrive/Escritorio/Magister en Data Science/Semestre 3/Proyecto de Data Science"
setwd(ruta)

library(tidyverse)
library(readxl)
library(openxlsx)
library(glmnet)
library(janitor)
library(ranger) # ranger es más rápido que randomForest
library(caret)
library(Metrics)
library(patchwork)
library(e1071)
library(nnet)
library(xgboost)
library(knitr) 
library(kableExtra)


# 1. Carga y limpieza -----
df = tibble(read_excel("data indices estres.xlsx"))
df = df %>% filter(codigo != "RE")
df = df %>% mutate(
  Site   = as.factor(Site),
  Bloque = as.factor(Bloque),
  codigo = as.factor(codigo)
)

columnas_a_eliminar = c("Formula texto", "T_wet", "T_dry",
                        "Temp_mean °C", "Temp_min °C", "Temp_max °C")

df = df %>% select(-all_of(columnas_a_eliminar))
df = df %>% mutate(across(4:ncol(.), as.numeric))
df = df %>% clean_names()


# 2. Definir conjuntos de features ----
band_cols = c("blue_480nm", "green_560nm", "red_edge_720nm")


vi_cols = c("ndre", "ci_green", "cire") # hay que tener ojo con el osavi 


# 3. Función CV con Group K-Fold por planta (Ridge o Lasso) ----
cv_glmnet_grupo = function(data, features, alpha, k = 10, seed = 42) {
  set.seed(seed)
  
  plantas = levels(data$codigo)
  folds   = split(sample(plantas), ceiling(seq_along(plantas) / (length(plantas) / k)))
  
  resultados = data.frame()
  
  for (i in seq_len(k)) {
    
    test_idx   = data$codigo %in% folds[[i]]
    train_data = data[!test_idx, ]
    test_data  = data[test_idx, ]
    
    # Matrices X e Y
    X_train = train_data %>%
      select(all_of(features)) %>%
      model.matrix(~ ., data = .) %>%
      .[, -1]                          # quita intercept
    
    X_test = test_data %>%
      select(all_of(features)) %>%
      model.matrix(~ ., data = .) %>%
      .[, -1]
    
    y_train = train_data$ndvi
    y_test  = test_data$ndvi
    
    # Ajuste con CV interna para lambda (solo sobre train)
    cv_model   = cv.glmnet(X_train, y_train, alpha = alpha)
    opt_lambda = cv_model$lambda.min
    
    pred = predict(cv_model, s = opt_lambda, newx = X_test) %>% as.numeric()
    
    R2   = 1 - (sum((y_test - pred)^2) / sum((y_test - mean(y_test))^2))
    RMSE = sqrt(mean((y_test - pred)^2))
    MAE  = mean(abs(y_test - pred))
    
    cat(sprintf("Fold %d | Train: %d | Test: %d | R²: %.4f\n",
                i, nrow(train_data), nrow(test_data), R2))
    
    resultados = rbind(resultados,
                       data.frame(Fold = i, R2 = R2, RMSE = RMSE, MAE = MAE))
  }
  
  cat(sprintf("─── R² medio: %.4f ± %.4f ───\n\n", mean(resultados$R2), sd(resultados$R2)))
  return(resultados)
}


# 4. Función de evaluación con Group K-Fold por arbol ----
# agrupar por 'codigo' (arbol) para evitar data leakage
# Arbol completas van a train o test, nunca divididas

eval_grupo = function(df, features, k = 10, seed = 42) {
  set.seed(seed)
  
  plantas = levels(df$codigo)
  folds   = split(sample(plantas), ceiling(seq_along(plantas) / (length(plantas) / k)))
  
  r2_folds = numeric(k)
  rmse_fols = numeric(k)
  mae_folds = numeric(k)
  resultados = data.frame()
  
  for (i in seq_len(k)) {
    test_idx = df$codigo %in% folds[[i]]
    train_df = df[!test_idx, ]
    test_df = df[test_idx, ]
    
    X_train = train_df %>% select(all_of(features))
    y_train = train_df$ndvi
    X_test = test_df  %>% select(all_of(features))
    y_test = test_df$ndvi
    
    modelo = ranger(
      x = X_train,
      y = y_train,
      num.trees = 500,
      mtry = max(1, floor(sqrt(length(features)))),
      min.node.size = 5,
      seed = seed
    )
    
    preds = predict(modelo, X_test)$predictions
    rmse_fols = sqrt(mean((y_test - preds)^2))
    mae_folds = mean(abs(y_test - preds))
    r2_folds = 1 - (sum((y_test - preds)^2) / sum((y_test - mean(y_test))^2))
    
    resultados = rbind(resultados,
                       data.frame(
                         Fold = i,
                         R2 = r2_folds,
                         RMSE = rmse_fols,
                         MAE = mae_folds
                       ))
  }
  
  #resultados = data.frame()
  
  #cat(sprintf("R² por fold: %s\n", paste(round(r2_folds, 3), collapse = ", ")))
  #cat(sprintf("R² medio:    %.4f ± %.4f\n\n", mean(r2_folds), sd(r2_folds)))
  return(resultados)
}


# 5. Función CV SVM -----------
# Se usa kernel radial (RBF), que es el más habitual para regresión con datos continuos.
# Los hiperparámetros cost y gamma se dejan en sus defaults de e1071 (cost=1, gamma=1/n_features).
# Para tne, agregar tune.svm() dentro del fold, pero tiene un costo computacional enorme bastante.
cv_svm_grupo = function(df, features, k = 10, seed = 42, cost = 1, gamma = NULL, epsilon = 0.1) {
  set.seed(seed)
  
  plantas = levels(df$codigo)
  folds   = split(sample(plantas), ceiling(seq_along(plantas) / (length(plantas) / k)))
  
  resultados = data.frame()
  
  for (i in seq_len(k)) {
    test_idx = df$codigo %in% folds[[i]]
    train_df = df[!test_idx, ]
    test_df  = df[test_idx, ]
    
    X_train = train_df %>% select(all_of(features))
    y_train = train_df$ndvi
    X_test  = test_df  %>% select(all_of(features))
    y_test  = test_df$ndvi
    
    # Escalar features (importante para SVM)
    scaler  = preProcess(X_train, method = c("center", "scale"))
    X_train_sc = predict(scaler, X_train)
    X_test_sc  = predict(scaler, X_test)
    
    # Gamma por defecto: 1 / n_features si no se especifica
    g = if (is.null(gamma)) 1 / length(features) else gamma
    
    modelo = svm(
      x       = X_train_sc,
      y       = y_train,
      type    = "eps-regression",
      kernel  = "radial",
      cost    = cost,
      gamma   = g,
      epsilon = epsilon
    )
    
    preds = predict(modelo, X_test_sc)
    
    R2   = 1 - (sum((y_test - preds)^2) / sum((y_test - mean(y_test))^2))
    RMSE = sqrt(mean((y_test - preds)^2))
    MAE  = mean(abs(y_test - preds))
    
    cat(sprintf("Fold %d | Train: %d | Test: %d | R²: %.4f\n",
                i, nrow(train_df), nrow(test_df), R2))
    
    resultados = rbind(resultados,
                       data.frame(Fold = i, R2 = R2, RMSE = RMSE, MAE = MAE))
  }
  
  cat(sprintf("─── R² medio: %.4f ± %.4f ───\n\n", mean(resultados$R2), sd(resultados$R2)))
  return(resultados)
}


# 6. Función CV red neuronal --------
# nnet implementa una red con UNA capa oculta (feedforward).
# size  = número de neuronas en la capa oculta (regla práctica: (n_inputs + 1) / 2)
# decay = regularización L2 para evitar sobreajuste (equivalente a weight decay)
# maxit = máximo de iteraciones del optimizador BFGS
# Los features se escalan antes de entrenar (nnet es sensible a la escala).
cv_nnet_grupo = function(df, features, k = 10, seed = 42, size = NULL, decay = 0.01, maxit = 500) {
  set.seed(seed)
  
  plantas = levels(df$codigo)
  folds   = split(sample(plantas), ceiling(seq_along(plantas) / (length(plantas) / k)))
  
  resultados = data.frame()
  
  # size por defecto: (n_features + 1) / 2, mínimo 3
  n_size = if (is.null(size)) max(3, floor((length(features) + 1) / 2)) else size
  
  for (i in seq_len(k)) {
    test_idx = df$codigo %in% folds[[i]]
    train_df = df[!test_idx, ]
    test_df  = df[test_idx, ]
    
    X_train = train_df %>% select(all_of(features))
    y_train = train_df$ndvi
    X_test  = test_df  %>% select(all_of(features))
    y_test  = test_df$ndvi
    
    # Escalar (nnet es sensible a la escala)
    scaler     = preProcess(X_train, method = c("center", "scale"))
    X_train_sc = predict(scaler, X_train)
    X_test_sc  = predict(scaler, X_test)
    
    # Escalar también y (linout = TRUE requiere que y esté en rango razonable)
    y_mean  = mean(y_train); y_sd = sd(y_train)
    y_train_sc = (y_train - y_mean) / y_sd
    
    modelo = nnet(
      x      = X_train_sc,
      y      = y_train_sc,
      size   = n_size,
      decay  = decay,
      maxit  = maxit,
      linout = TRUE,    # linout=TRUE para regresión (salida lineal)
      trace  = FALSE
    )
    
    preds_sc = predict(modelo, X_test_sc) %>% as.numeric()
    preds    = preds_sc * y_sd + y_mean   # desescalar predicciones
    
    R2   = 1 - (sum((y_test - preds)^2) / sum((y_test - mean(y_test))^2))
    RMSE = sqrt(mean((y_test - preds)^2))
    MAE  = mean(abs(y_test - preds))
    
    cat(sprintf("Fold %d | Train: %d | Test: %d | R²: %.4f\n",
                i, nrow(train_df), nrow(test_df), R2))
    
    resultados = rbind(resultados,
                       data.frame(Fold = i, R2 = R2, RMSE = RMSE, MAE = MAE))
  }
  
  cat(sprintf("─── R² medio: %.4f ± %.4f ───\n\n", mean(resultados$R2), sd(resultados$R2)))
  return(resultados)
}

# 7. Función CV XGBoost ----
# nrounds   = número de árboles (iteraciones de boosting)
# max_depth = profundidad máxima de cada árbol base
# eta       = learning rate (tasa de aprendizaje)
# subsample = fracción de filas muestreadas por árbol (bagging)
# colsample = fracción de columnas muestreadas por árbol
# Los hiperparámetros usados aquí son valores razonables para un dataset mediano;
# para tuning más fino se puede usar xgb.cv() o caret dentro del fold.
cv_xgb_grupo = function(df, features, k = 10, seed = 42, nrounds = 200, max_depth = 4, eta = 0.05, subsample = 0.8, colsample = 0.8) {
  set.seed(seed)
  
  plantas = levels(df$codigo)
  folds   = split(sample(plantas), ceiling(seq_along(plantas) / (length(plantas) / k)))
  
  resultados = data.frame()
  
  params = list(
    objective        = "reg:squarederror",
    max_depth        = max_depth,
    eta              = eta,
    subsample        = subsample,
    colsample_bytree = colsample,
    seed             = seed
  )
  
  for (i in seq_len(k)) {
    test_idx = df$codigo %in% folds[[i]]
    train_df = df[!test_idx, ]
    test_df  = df[test_idx, ]
    
    X_train = train_df %>% select(all_of(features)) %>% as.matrix()
    y_train = train_df$ndvi
    X_test  = test_df  %>% select(all_of(features)) %>% as.matrix()
    y_test  = test_df$ndvi
    
    dtrain = xgb.DMatrix(data = X_train, label = y_train)
    dtest  = xgb.DMatrix(data = X_test,  label = y_test)
    
    modelo = xgb.train(
      params  = params,
      data    = dtrain,
      nrounds = nrounds,
      verbose = 0
    )
    
    preds = predict(modelo, dtest)
    
    R2   = 1 - (sum((y_test - preds)^2) / sum((y_test - mean(y_test))^2))
    RMSE = sqrt(mean((y_test - preds)^2))
    MAE  = mean(abs(y_test - preds))
    
    cat(sprintf("Fold %d | Train: %d | Test: %d | R²: %.4f\n",
                i, nrow(train_df), nrow(test_df), R2))
    
    resultados = rbind(resultados,
                       data.frame(Fold = i, R2 = R2, RMSE = RMSE, MAE = MAE))
  }
  
  cat(sprintf("─── R² medio: %.4f ± %.4f ───\n\n", mean(resultados$R2), sd(resultados$R2)))
  return(resultados)
}

# 8. Evaluar CV ----
cat("=== RIDGE — ALPHA = 0 ===\n")
ridge_completo = cv_glmnet_grupo(df, c(band_cols, vi_cols), alpha = 0)
ridge_completo = cbind(rep("Ridge", nrow(ridge_completo)), ridge_completo)
names(ridge_completo)[1] = "Modelo"
ridge_completo
#write.xlsx(ridge_completo, "resultados_ridge_cv.xlsx")


cat("=== LASSO — ALPHA = 1 ===\n")
lasso_completo = cv_glmnet_grupo(df, c(band_cols, vi_cols), alpha = 1)
lasso_completo = cbind(rep("Lasso", nrow(lasso_completo)), lasso_completo)
names(lasso_completo)[1] = "Modelo"
lasso_completo
#write.xlsx(lasso_completo, "resultados_lasso_cv.xlsx")


cat("=== RANDOM FOREST ===\n")
r2_completo = eval_grupo(df, c(band_cols, vi_cols))
r2_completo = cbind(rep("RF", nrow(r2_completo)), r2_completo)
names(r2_completo)[1] = "Modelo"
r2_completo
#write.xlsx(r2_completo, "resultados_cv_rf.xlsx")

cat("=== SVM (kernel radial) ===\n")
svm_completo = cv_svm_grupo(df, c(band_cols, vi_cols))
svm_completo = cbind(rep("SVM", nrow(svm_completo)), svm_completo)
names(svm_completo)[1] = "Modelo"
svm_completo
#write.xlsx(svm_completo, "resultados_cv_vsm.xlsx")

cat("=== RED NEURONAL (nnet, 1 capa oculta) ===\n")
nnet_completo = cv_nnet_grupo(df, c(band_cols, vi_cols))
nnet_completo = cbind(rep("Red neuronal", nrow(nnet_completo)), nnet_completo)
names(nnet_completo)[1] = "Modelo"
nnet_completo
#write.xlsx(nnet_completo, "resultados_cv_red.xlsx")

cat("=== XGBOOST ===\n")
xgb_completo = cv_xgb_grupo(df, c(band_cols, vi_cols))
xgb_completo = cbind(rep("XGBOOST", nrow(xgb_completo)), xgb_completo)
names(xgb_completo)[1] = "Modelo"
xgb_completo
#write.xlsx(xgb_completo, "resultados_cv_xgboost.xlsx")


# 9. Resumen comparativo -----


resumen_cv_completo = rbind(ridge_completo, lasso_completo, r2_completo, svm_completo, nnet_completo, xgb_completo)
#write.xlsx(resumen_cv_completo, "resumen_cv_completo.xlsx")

# Ver estadísticas por fold

res_folds = bind_rows(
  ridge_completo %>% mutate(Modelo = "Ridge"),
  lasso_completo %>% mutate(Modelo = "Lasso"),
  r2_completo    %>% mutate(Modelo = "Random Forest"),
  svm_completo   %>% mutate(Modelo = "SVM"),
  nnet_completo  %>% mutate(Modelo = "Red Neuronal"),
  xgb_completo   %>% mutate(Modelo = "XGBoost")
) %>% mutate(Modelo = factor(Modelo), Fold = factor(Fold))

res_folds %>%
  group_by(Modelo) %>%
  summarise(across(c(R2, RMSE, MAE), list(media = mean, sd = sd)))

# Test sobre métricas del fold

# ¿Hay diferencias globales entre los modelos?
friedman.test(R2 ~ Modelo | Fold, data = res_folds)
friedman.test(RMSE ~ Modelo | Fold, data = res_folds)
friedman.test(MAE ~ Modelo | Fold, data = res_folds)

# ¿Entre cuáles pares? (Wilcoxon pareado con corrección de Holm)
pairwise.wilcox.test(res_folds$RMSE, res_folds$Modelo,
                     paired = TRUE, p.adjust.method = "holm")

pairwise.wilcox.test(res_folds$R2,  res_folds$Modelo, paired = TRUE, p.adjust.method = "holm")
pairwise.wilcox.test(res_folds$MAE, res_folds$Modelo, paired = TRUE, p.adjust.method = "holm")

resumen_cv = data.frame(
  Modelo = c("Ridge", "Lasso", "Random Forest", "SVM", "Red Neuronal", "XGBoost"),
  R2_medio   = c(mean(ridge_completo$R2), mean(lasso_completo$R2), mean(r2_completo$R2),
                 mean(svm_completo$R2),   mean(nnet_completo$R2),  mean(xgb_completo$R2)),
  RMSE_medio = c(mean(ridge_completo$RMSE), mean(lasso_completo$RMSE), mean(r2_completo$RMSE),
                 mean(svm_completo$RMSE),   mean(nnet_completo$RMSE),  mean(xgb_completo$RMSE)),
  MAE_medio  = c(mean(ridge_completo$MAE), mean(lasso_completo$MAE), mean(r2_completo$MAE),
                 mean(svm_completo$MAE),   mean(nnet_completo$MAE),  mean(xgb_completo$MAE))
)

print(resumen_cv)
#write.xlsx(resumen_cv, "resultados_grupo.xlsx")

# Orden fijo de los modelos (ajusta abreviaciones a tu gusto)
orden_modelos = c("Ridge", "Lasso", "Random Forest", "SVM", "Red Neuronal", "XGBoost")

resumen_cv = resumen_cv %>%
  mutate(Modelo = factor(Modelo, levels = orden_modelos))

resumen_cv = resumen_cv %>%
  mutate(color_barra = ifelse(R2_medio == max(R2_medio), "#6FA8DC", "gray"))

resumen_cv = resumen_cv %>%
  mutate(color_barra_rmse = ifelse(RMSE_medio == min(RMSE_medio), "#6FA8DC", "gray"))

resumen_cv = resumen_cv %>%
  mutate(color_barra_mae = ifelse(MAE_medio == min(MAE_medio), "#6FA8DC", "gray"))

r2 = ggplot(resumen_cv,
       aes(x = Modelo,
           y = R2_medio,
           fill = color_barra)) +
  geom_col(color = "black",
           linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.4f", R2_medio)),
            hjust = -0.15,
            size = 6,
            fontface = "bold") +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  scale_fill_identity()+
  labs(
    title = expression(bold("Comparación del " * R^2 * " medio por modelo respecto al NDVI")),
    x = "",
    y = expression(bold(R^2 ~ "medio"))
  ) +
  theme_classic(base_size = 12)+
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title.x = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold", size = 15)
  )
r2

rmse = ggplot(resumen_cv,
            aes(x = Modelo,
                y = RMSE_medio,
                fill = color_barra_rmse)) +
  geom_col(color = "black",
           linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.4f", RMSE_medio)),
            hjust = -0.15,
            size = 6,
            fontface = "bold") +
  coord_flip() +
  scale_y_continuous(limits = c(0, 0.05)) +
  scale_fill_identity()+
  labs(
    title = expression(bold("Comparación del " * RMSE * " medio por modelo respecto al NDVI")),
    x = "",
    y = expression(bold(RMSE ~ "medio"))
  ) +
  theme_classic(base_size = 12)+
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title.x = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold", size = 15)
  )
rmse

mae = ggplot(resumen_cv,
              aes(x = Modelo,
                  y = MAE_medio,
                  fill = color_barra_mae)) +
  geom_col(color = "black",
           linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.4f", MAE_medio)),
            hjust = -0.15,
            size = 6,
            fontface = "bold") +
  coord_flip() +
  scale_y_continuous(limits = c(0, 0.03)) +
  scale_fill_identity()+
  labs(
    title = expression(bold("Comparación del " * MAE * " medio por modelo respecto al NDVI")),
    x = "",
    y = expression(bold(MAE ~ "medio"))
  ) +
  theme_classic(base_size = 12)+
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title.x = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold", size = 15)
  )
mae

# 1800 y 750

(r2 | rmse) /
  (mae | plot_spacer())


# 10. Modelo final: split 70/30 por planta ---------------
# Modelo final Ridge/Lasso
modelo_final_grupo = function(data, features, alpha, seed = 42) {
  set.seed(seed)
  
  # Separar plantas completas en 70/30
  plantas    = levels(data$codigo)
  n_train    = round(0.7 * length(plantas))
  plantas_train = sample(plantas, n_train)
  
  train_data = data[data$codigo %in% plantas_train, ]
  test_data  = data[!data$codigo %in% plantas_train, ]
  
  cat(sprintf("Plantas train: %d | Plantas test: %d\n",
              length(plantas_train), length(plantas) - n_train))
  cat(sprintf("Obs train: %d    | Obs test: %d\n",
              nrow(train_data), nrow(test_data)))
  
  # Matrices X e Y
  X_train = train_data %>%
    select(all_of(features)) %>%
    model.matrix(~ ., data = .) %>%
    .[, -1]
  
  X_test = test_data %>%
    select(all_of(features)) %>%
    model.matrix(~ ., data = .) %>%
    .[, -1]
  
  y_train = train_data$ndvi
  y_test  = test_data$ndvi
  
  # Ajuste
  cv_model = cv.glmnet(X_train, y_train, alpha = alpha)
  
  pred = predict(cv_model, s = "lambda.min", newx = X_test) %>% as.numeric()
  
  return(data.frame(observado = y_test, predicho = pred))
}

# Modelo final Random Forest
modelo_final_rf_grupo = function(data, features, seed = 42) {
  set.seed(seed)
  
  # Separar plantas completas en 70/30
  plantas       = levels(data$codigo)
  n_train       = round(0.7 * length(plantas))
  plantas_train = sample(plantas, n_train)
  
  train_data = data[data$codigo %in% plantas_train, ]
  test_data  = data[!data$codigo %in% plantas_train, ]
  
  cat(sprintf("Plantas train: %d | Plantas test: %d\n",
              length(plantas_train), length(plantas) - n_train))
  cat(sprintf("Obs train: %d    | Obs test: %d\n",
              nrow(train_data), nrow(test_data)))
  
  X_train = train_data %>% select(all_of(features))
  y_train = train_data$ndvi
  X_test  = test_data  %>% select(all_of(features))
  y_test  = test_data$ndvi
  
  modelo = ranger(
    x             = X_train,
    y             = y_train,
    num.trees     = 500,
    mtry          = max(1, floor(sqrt(length(features)))),
    min.node.size = 5,
    seed          = seed
  )
  
  pred = predict(modelo, X_test)$predictions
  
  return(data.frame(observado = y_test, predicho = pred))
}


# SVM
# Mismos hiperparámetros que en CV (cost=1, gamma=1/n_features, epsilon=0.1).
# Escalar con los parámetros del train para no filtrar info del test.
modelo_final_svm_grupo = function(data, features, seed = 42, cost = 1, gamma = NULL, epsilon = 0.1) {
  set.seed(seed)
  
  plantas       = levels(data$codigo)
  n_train       = round(0.7 * length(plantas))
  plantas_train = sample(plantas, n_train)
  
  train_data = data[data$codigo %in% plantas_train, ]
  test_data  = data[!data$codigo %in% plantas_train, ]
  
  cat(sprintf("Plantas train: %d | Plantas test: %d\n",
              length(plantas_train), length(plantas) - n_train))
  cat(sprintf("Obs train: %d    | Obs test: %d\n", nrow(train_data), nrow(test_data)))
  
  X_train = train_data %>% select(all_of(features))
  y_train = train_data$ndvi
  X_test  = test_data  %>% select(all_of(features))
  y_test  = test_data$ndvi
  
  scaler     = preProcess(X_train, method = c("center", "scale"))
  X_train_sc = predict(scaler, X_train)
  X_test_sc  = predict(scaler, X_test)
  
  g = if (is.null(gamma)) 1 / length(features) else gamma
  
  modelo = svm(
    x       = X_train_sc,
    y       = y_train,
    type    = "eps-regression",
    kernel  = "radial",
    cost    = cost,
    gamma   = g,
    epsilon = epsilon
  )
  
  pred = predict(modelo, X_test_sc)
  return(data.frame(observado = y_test, predicho = pred))
}


# --- Red Neuronal ---
# Escalar X e y con los parámetros del train (desescalar las predicciones al final).
modelo_final_nnet_grupo = function(data, features, seed = 42, size = NULL, decay = 0.01, maxit = 500) {
  set.seed(seed)
  
  plantas       = levels(data$codigo)
  n_train       = round(0.7 * length(plantas))
  plantas_train = sample(plantas, n_train)
  
  train_data = data[data$codigo %in% plantas_train, ]
  test_data  = data[!data$codigo %in% plantas_train, ]
  
  cat(sprintf("Plantas train: %d | Plantas test: %d\n",
              length(plantas_train), length(plantas) - n_train))
  cat(sprintf("Obs train: %d    | Obs test: %d\n", nrow(train_data), nrow(test_data)))
  
  X_train = train_data %>% select(all_of(features))
  y_train = train_data$ndvi
  X_test  = test_data  %>% select(all_of(features))
  y_test  = test_data$ndvi
  
  n_size = if (is.null(size)) max(3, floor((length(features) + 1) / 2)) else size
  
  scaler     = preProcess(X_train, method = c("center", "scale"))
  X_train_sc = predict(scaler, X_train)
  X_test_sc  = predict(scaler, X_test)
  
  y_mean     = mean(y_train); y_sd = sd(y_train)
  y_train_sc = (y_train - y_mean) / y_sd
  
  modelo = nnet(
    x      = X_train_sc,
    y      = y_train_sc,
    size   = n_size,
    decay  = decay,
    maxit  = maxit,
    linout = TRUE,
    trace  = FALSE
  )
  
  preds_sc = predict(modelo, X_test_sc) %>% as.numeric()
  pred     = preds_sc * y_sd + y_mean
  
  return(data.frame(observado = y_test, predicho = pred))
}


# --- XGBoost ---
# nrounds tuneado con los mismos hiperparámetros de la CV.
modelo_final_xgb_grupo = function(data, features, seed = 42, nrounds = 200, max_depth = 4, eta = 0.05, subsample = 0.8, colsample = 0.8) {
  set.seed(seed)
  
  plantas       = levels(data$codigo)
  n_train       = round(0.7 * length(plantas))
  plantas_train = sample(plantas, n_train)
  
  train_data = data[data$codigo %in% plantas_train, ]
  test_data  = data[!data$codigo %in% plantas_train, ]
  
  cat(sprintf("Plantas train: %d | Plantas test: %d\n",
              length(plantas_train), length(plantas) - n_train))
  cat(sprintf("Obs train: %d    | Obs test: %d\n", nrow(train_data), nrow(test_data)))
  
  X_train = train_data %>% select(all_of(features)) %>% as.matrix()
  y_train = train_data$ndvi
  X_test  = test_data  %>% select(all_of(features)) %>% as.matrix()
  y_test  = test_data$ndvi
  
  dtrain = xgb.DMatrix(data = X_train, label = y_train)
  dtest  = xgb.DMatrix(data = X_test,  label = y_test)
  
  params = list(
    objective        = "reg:squarederror",
    max_depth        = max_depth,
    eta              = eta,
    subsample        = subsample,
    colsample_bytree = colsample,
    seed             = seed
  )
  
  modelo = xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  pred = predict(modelo, dtest)
  return(data.frame(observado = y_test, predicho = pred))
}


metricas_modelo_final = function(df, nombre) {
  data.frame(
    Nombre_modelo = nombre,
    R2 = 1 - (sum((df$observado - df$predicho)^2) / sum((df$observado - mean(df$observado))^2)),
    #R2   = cor(df$observado, df$predicho)^2,
    RMSE = rmse(df$observado, df$predicho),
    MAE  = mae(df$observado, df$predicho)
  )
}


plot_obs_pred = function(df, titulo) {
  
  metricas = metricas_modelo_final(df, titulo)
  label    = sprintf("R² = %.3f\nRMSE = %.3f\nMAE = %.3f",
                     metricas$R2, metricas$RMSE, metricas$MAE)
  
  ggplot(df, aes(x = observado, y = predicho)) +
    geom_point(alpha = 0.4, color = "#2C7BB6", size = 1.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
    annotate("text", x = -Inf, y = Inf, label = label,
             hjust = -0.1, vjust = 1.3, size = 3.5, family = "mono") +
    labs(title = titulo, x = "NDVI Observado", y = "NDVI Predicho") +
    theme_bw()+
    theme(plot.title = element_text(hjust = 0.5, face = "bold")
    )
}




res_rf_completo = modelo_final_rf_grupo(df, c(band_cols, vi_cols))
res_rf_completo = cbind(rep("RF", nrow(res_rf_completo)), res_rf_completo)
names(res_rf_completo)[1] = "Modelo"

res_ridge_completo = modelo_final_grupo(df, c(band_cols, vi_cols), alpha = 0)
res_ridge_completo = cbind(rep("Ridge", nrow(res_ridge_completo)), res_ridge_completo)
names(res_ridge_completo)[1] = "Modelo"

res_lasso_completo = modelo_final_grupo(df, c(band_cols, vi_cols), alpha = 1)
res_lasso_completo = cbind(rep("Lasso", nrow(res_lasso_completo)), res_lasso_completo)
names(res_lasso_completo)[1] = "Modelo"

res_svm_completo = modelo_final_svm_grupo(df, c(band_cols, vi_cols))
res_svm_completo = cbind(rep("SVM", nrow(res_svm_completo)), res_svm_completo)
names(res_svm_completo)[1] = "Modelo"

res_nnet_completo = modelo_final_nnet_grupo(df, c(band_cols, vi_cols))
res_nnet_completo = cbind(rep("Red neuronal", nrow(res_nnet_completo)), res_nnet_completo)
names(res_nnet_completo)[1] = "Modelo"

res_xgb_completo = modelo_final_xgb_grupo(df, c(band_cols, vi_cols))
res_xgb_completo = cbind(rep("XGboost", nrow(res_xgb_completo)), res_xgb_completo)
names(res_xgb_completo)[1] = "Modelo"

resumen_particion_final = rbind(res_rf_completo, res_ridge_completo, res_lasso_completo, res_svm_completo, res_nnet_completo, res_xgb_completo)
#write.xlsx(resumen_particion_final, "resumen_datos_partición_final.xlsx")

p1 = plot_obs_pred(res_rf_completo, "RF")
p2 = plot_obs_pred(res_ridge_completo, "Ridge")
p3 = plot_obs_pred(res_lasso_completo, "Lasso")
p4 = plot_obs_pred(res_svm_completo, "SVM")
p5 = plot_obs_pred(res_nnet_completo, "Red Neuronal")
p6 = plot_obs_pred(res_xgb_completo, "XGBoost")


(p1 | p2 | p3)/(p4 | p5| p6)


resumen_final = rbind(
  metricas_modelo_final(res_ridge_completo, "Ridge"),
  metricas_modelo_final(res_lasso_completo, "Lasso"),
  metricas_modelo_final(res_rf_completo, "Random Forest"),
  metricas_modelo_final(res_svm_completo, "SVM"),
  metricas_modelo_final(res_nnet_completo, "Red Neuronal"),
  metricas_modelo_final(res_xgb_completo, "XGBoost")
)

print(resumen_final)

# Ver valores atipicos
resumen_particion_final = tibble(read_excel("resumen_datos_partición_final.xlsx"))
res = resumen_particion_final %>% mutate(residuo = observado - predicho)

res %>%
  group_by(Modelo) %>%
  mutate(q1 = quantile(residuo, .25), q3 = quantile(residuo, .75),
         iqr = q3 - q1,
         atipico = residuo < q1 - 1.5*iqr | residuo > q3 + 1.5*iqr) %>%
  summarise(n_atipicos = sum(atipico),
            pct = round(100*mean(atipico), 1),
            max_abs = max(abs(residuo)))

res = res %>%
  group_by(Modelo) %>%
  mutate(id = row_number(),
         atipico = residuo < quantile(residuo, .25) - 1.5*IQR(residuo) |
           residuo > quantile(residuo, .75) + 1.5*IQR(residuo)) %>%
  ungroup()

# a) ¿Son las mismas observaciones en todos los modelos?
res %>% group_by(id) %>% summarise(n_modelos = sum(atipico)) %>% count(n_modelos)

# b) ¿Dónde se concentran? (rango de NDVI y plantas)
res %>% filter(atipico) %>% count(Modelo, tramo = observado > 0.7)
#res %>% filter(atipico) %>% count(Modelo, codigo, sort = TRUE) %>% head(15)

# Mayoria sobre 0.7, lo esperado en plantas sanas
res %>%
  group_by(Modelo, tramo = ifelse(observado > 0.7, "> 0.7", "≤ 0.7")) %>%
  summarise(n = n(), n_atip = sum(atipico),
            pct = round(100*mean(atipico), 1),
            RMSE = sqrt(mean(residuo^2)), .groups = "drop")

# c) Sensibilidad: métricas sin las observaciones extremas (mismo conjunto para todos)
ids_extremos = res %>% filter(atipico) %>% pull(id) %>% unique()
res %>% filter(!id %in% ids_extremos) %>% group_by(Modelo) %>%
  summarise(RMSE = sqrt(mean(residuo^2)), MAE = mean(abs(residuo)),
            R2 = cor(observado, predicho)^2)


tabla_tramo = res %>%
  mutate(tramo = ifelse(observado > 0.7, "alto", "bajo_medio")) %>%
  group_by(Modelo, tramo) %>%
  summarise(n = n(),
            RMSE = sqrt(mean(residuo^2)),
            MAE  = mean(abs(residuo)), .groups = "drop")

tabla_tramo

# Formato ancho: ¿cuántas veces mayor es el error en NDVI ≤ 0.7?
tabla_tramo %>%
  select(Modelo, tramo, RMSE) %>%
  pivot_wider(names_from = tramo, values_from = RMSE) %>%
  mutate(cociente = bajo_medio / alto)

# Modificar gráficos en función de los valores atipicos ---------


res = res %>% group_by(id) %>% 
  mutate(extremo_comun = any(atipico)) %>% 
  ungroup()


hist(res$residuo) # usando el 1.5*IQR se asume que los residuos son simetricos, se confirma con el grafico


met = function(o,p){
  c(R2 = 1 - sum((o - p)^2) / sum((o - mean(o))^2),
    RMSE = sqrt(mean((o - p)^2)),
    MAE = mean(abs(o - p))
  )
}

# Grafico por modelo
# criterio          : "comun" (recomendado) o "propio" (extremos de ese modelo)
# solo_sin_atipicos : FALSE = muestra todos los puntos con los extremos resaltados en rojo
#                     TRUE  = muestra solo los puntos no extremos
plot_obs_pred_atip = function(res, modelo, titulo = modelo, criterio = "comun",
                              solo_sin_atipicos = FALSE, etq = "NDVI", lims = NULL,
                              leg_texto = 12, leg_punto = 4, leg_clave = 1.5, met_texto = 3.5) {
  
  d = res %>%
    filter(Modelo == modelo) %>%
    mutate(extremo = if (criterio == "comun") extremo_comun else atipico)
  
  d_ok  = d %>% filter(!extremo)
  m_all = met(d$observado, d$predicho)
  m_ok  = met(d_ok$observado, d_ok$predicho)
  
  label = sprintf("R² todas las observaciones = %.3f\nR² sin extremos = %.3f",
                  m_all["R2"], m_ok["R2"])
  
  datos = if (solo_sin_atipicos) d_ok else d
  
  g = ggplot(datos, aes(x = observado, y = predicho, color = extremo)) +
    geom_point(alpha = 0.45, size = 1.4) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
    scale_color_manual(values = c(`FALSE` = "#2C7BB6", `TRUE` = "#D7191C"),
                       labels = c(`FALSE` = "Dentro del rango", `TRUE` = "Extremo (1.5×IQR)"),
                       name = NULL) +
    annotate("text", x = -Inf, y = Inf, label = label,
             hjust = -0.05, vjust = 1.1, size = met_texto, family = "mono", fontface = "bold") +
    labs(title = titulo, x = paste(etq, "Observado"), y = paste(etq, "Predicho")) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "bottom",
          legend.text     = element_text(size = leg_texto),
          legend.key.size = unit(leg_clave, "lines")) +
    guides(color = guide_legend(override.aes = list(size = leg_punto, alpha = 1)))
  
  if (!is.null(lims)) g = g + coord_equal(xlim = lims, ylim = lims)
  if (solo_sin_atipicos) g = g + theme(legend.position = "none")
  g
}

# 3) Figura con los seis modelos ----
orden   = c("RF", "Ridge", "Lasso", "SVM", "Red neuronal", "XGboost")   # etiquetas tal como están en res$Modelo
titulos = c(RF = "Random Forest", Ridge = "Ridge", Lasso = "Lasso", SVM = "SVM",
            `Red neuronal` = "Red Neuronal", XGboost = "XGBoost")
lims_comunes = range(c(res$observado, res$predicho))   # mismos ejes en todos los paneles

# 3a) Todas las observaciones, extremos resaltados (versión recomendada para la tesis)
paneles = lapply(orden, function(m) {
  plot_obs_pred_atip(res, m, titulo = titulos[[m]], lims = lims_comunes, etq = "NDVI",
                     leg_texto = 12, leg_punto = 5, leg_clave = 2, met_texto = 3.8)
})

fig_resaltado = wrap_plots(paneles, ncol = 3) +
  plot_layout(guides = "collect") +
  plot_annotation(theme = theme(legend.position = "bottom"))
fig_resaltado
ggsave("particion_final_ndvi_resaltado.png", fig_resaltado, width = 14, height = 9, dpi = 300)

# 3b) Solo observaciones no extremas (complemento)
paneles_sin = lapply(orden, function(m) {
  plot_obs_pred_atip(res, m, titulo = titulos[[m]], lims = lims_comunes, etq = "NDVI",
                     solo_sin_atipicos = TRUE)
})
fig_sin = wrap_plots(paneles_sin, ncol = 3)
fig_sin
# ggsave("particion_final_cwsi_sin_extremos.png", fig_sin, width = 14, height = 9, dpi = 300)


# 4) Tabla comparativa: con y sin observaciones extremas ----
tabla_atipicos = res %>%
  group_by(Modelo) %>%
  summarise(
    n            = n(),
    n_extremos   = sum(extremo_comun),
    pct_extremos = round(100 * mean(extremo_comun), 1),
    R2_todas     = 1 - sum(residuo^2) / sum((observado - mean(observado))^2),
    RMSE_todas   = sqrt(mean(residuo^2)),
    MAE_todas    = mean(abs(residuo)),
    R2_sin       = 1 - sum(residuo[!extremo_comun]^2) /
      sum((observado[!extremo_comun] - mean(observado[!extremo_comun]))^2),
    RMSE_sin     = sqrt(mean(residuo[!extremo_comun]^2)),
    MAE_sin      = mean(abs(residuo[!extremo_comun])),
    .groups = "drop"
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

print(tabla_atipicos)

tabla_metricas = tabla_atipicos %>%
  transmute(Modelo,
            `N extremos (%)` = sprintf("%d (%.1f%%)", n_extremos, pct_extremos),
            RMSE_todas, RMSE_sin,
            MAE_todas,  MAE_sin)

print(tabla_metricas)

kable(tabla_metricas, format = "latex", booktabs = TRUE, digits = 4,
      col.names = c("Modelo", "Extremos", "Todas", "Sin extremos", "Todas", "Sin extremos"),
      caption = "RMSE y MAE de la partición final con y sin observaciones extremas.",
      label = "metricas_atipicos_cwsi") %>%
  add_header_above(c(" " = 2, "RMSE" = 2, "MAE" = 2))
















