# Proyecto de Consolidación: Sistema de Scoring Crediticio
## Enhanced German Credit Data - Clasificación y Preprocesado

---

## 📋 Descripción del Proyecto

Este proyecto integrador combina los conocimientos de los módulos:
- **Preprocesado, Creación y Selección de Características**
- **Principales Algoritmos para Clasificación en Aprendizaje Máquina**

Los estudiantes trabajarán en equipos de 3-5 personas para desarrollar un **sistema completo de evaluación de riesgo crediticio** utilizando el dataset Enhanced German Credit Data.

---

## 🎯 Objetivos del Proyecto

1. **Aplicar técnicas avanzadas de preprocesado de datos**
2. **Realizar feature engineering y selección de características**
3. **Comparar múltiples algoritmos de clasificación**
4. **Desarrollar un pipeline reproducible y libre de data leakage**
5. **Analizar e interpretar los resultados desde una perspectiva de negocio**

---

## 📊 Sobre el Dataset

### Información General
- **Nombre**: Enhanced German Credit Data
- **Instancias**: 1,250 (aumento del 25% respecto al original)
- **Features**: 28 + 1 target
- **Objetivo**: Predecir si un cliente es "bueno" (1) o "malo" (2) para otorgar crédito

### Estructura de Variables

Explicada en el documento "german_credit_data_description.txt"
---

## 📝 Entregables del Proyecto

### 1. **Análisis Exploratorio de Datos (EDA)** - 20%
- Análisis descriptivo completo de todas las variables
- Visualizaciones de distribuciones y relaciones
- Identificación de problemas de calidad de datos
- Análisis de correlaciones
- Detección visual de outliers y valores faltantes

**Notebook**: `01_EDA.ipynb`

### 2. **Pipeline de Preprocesado** - 30%

Implementar y justificar:

#### A. Tratamiento de Valores Faltantes
- Eliminación de filas (si procede)
- Estrategia de imputación 

#### B. Detección y Tratamiento de Outliers
- Identificar outliers usando al menos 2 métodos
- Aplicar tratamiento apropiado

#### C. Encoding de Variables Categóricas
- One-Hot Encoding para nominales
- Ordinal Encoding donde corresponda
- Target Encoding (opcional, si es apropiado)
- Manejo de categorías raras

#### D. Transformación de Variables
- Normalización/Estandarización según el algoritmo
- Transformaciones para normalidad (si es necesario)
- Justificar elecciones

#### E. Feature Engineering
- Crear al menos 3 features nuevas adicionales
- Justificar su utilidad potencial

#### F. Selección de Features
- Eliminar variables (si procede)
- Evaluar eliminación de variable/s correlacionada/s (si procede)
- Aplicar al menos 2 métodos de selección de variables:
  - Métodos de filtro (correlación, chi-cuadrado, mutual information)
  - Métodos wrapper (RFE)
  - Feature importance de modelos

**Notebook**: `02_Preprocesado.ipynb`

### 3. **Modelado y Comparación de Algoritmos** - 30%

Deben implementar y comparar **al menos 6 algoritmos**:

#### Algoritmos Obligatorios (mínimo 5):
1. Regresión Logística (baseline)
2. Decision Tree
3. Random Forest
4. XGBoost o LightGBM
5. SVM (con al menos 2 kernels diferentes)

#### Algoritmos Opcionales (elegir al menos 1):
6. K-Nearest Neighbors
7. Naive Bayes
8. Ensemble con Voting o Stacking
9. Otros (justificar)

#### Requisitos del Modelado:
- ✅ Usar validación cruzada estratificada (5-10 folds)
- ✅ Ajustar hiperparámetros (GridSearchCV o RandomizedSearchCV)
- ✅ Evitar data leakage (usar Pipelines de scikit-learn)
- ✅ Considerar el desbalanceo de clases:
  - SMOTE, ADASYN, undersampling
  - Class weights
  - Threshold tuning
- ✅ Considerar la matriz de costos (EXTRA: Opcional)
  - Costo de clasificar malo como bueno: 5
  - Costo de clasificar bueno como malo: 1

**Notebook**: `03_Modelado.ipynb`

### 4. **Evaluación y Análisis de Resultados** - 15%

#### Métricas de Evaluación:
- Accuracy (baseline)
- Precision, Recall, F1-Score (por clase)
- ROC-AUC
- Confusion Matrix
- **Costo total** (considerando la matriz de costos)
- Curvas de aprendizaje

#### Análisis Requerido:
- Comparación de algoritmos (tabla resumen)
- Análisis de feature importance
- Interpretación de errores (FP y FN)
- Análisis de costos de negocio
- Recomendaciones de umbrales de decisión

**Notebook**: `04_Evaluacion.ipynb`

### 5. **Presentación Final y Reporte** - 5%

**Formato**: PDF o presentación (máximo 15 slides)

Contenido:
1. Introducción y contexto del problema
2. Resumen del preprocesado (decisiones clave)
3. Features más importantes identificadas
4. Comparación de algoritmos (tabla y gráficos)
5. Modelo recomendado y justificación
6. Análisis de costos y ROI potencial
7. Limitaciones y trabajo futuro
8. Conclusiones

**Archivo**: `Reporte_Equipo_X.pdf`

---

## 🔧 Estructura de Archivos propuesta

```
proyecto_credit_scoring/
│
├── data/
│   ├── german_credit_data.txt             # Dataset original
│   └── german_credit_data_description.txt # Descripción
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesado.ipynb
│   ├── 03_Modelado.ipynb
│   └── 04_Evaluacion.ipynb
│
├── src/                                 # (Opcional) Scripts reutilizables
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── modeling.py
│
├── outputs/
│   ├── figures/                         # Gráficos generados
│   ├── models/                          # Modelos entrenados (.pkl)
│   └── results/                         # Tablas de resultados
│
├── README.md                            # Este archivo
├── requirements.txt                     # Dependencias
└── Reporte_Equipo_X.pdf                # Reporte final
```

---

## Evaluación

| Criterio | Peso | Descripción |
|----------|------|-------------|
| **EDA** | 20% | Completitud, visualizaciones, insights |
| **Preprocesado** | 30% | Justificación de decisiones, implementación correcta, ausencia de data leakage |
| **Modelado** | 30% | Diversidad de algoritmos, optimización, validación correcta |
| **Evaluación** | 15% | Métricas apropiadas, interpretación, análisis de negocio |
| **Presentación** | 5% | Claridad, estructura, conclusiones |

### Criterios Específicos de Evaluación:

**Excelente (9-10)**
- Todas las técnicas implementadas correctamente
- Justificaciones sólidas basadas en análisis
- Pipeline reproducible sin data leakage
- Análisis profundo de resultados
- Insights valiosos de negocio

**Bueno (7-8)**
- Mayoría de técnicas implementadas
- Justificaciones razonables
- Pipeline funcional con mínimos errores
- Análisis completo de métricas
- Interpretación correcta

**Suficiente (5-6)**
- Técnicas básicas implementadas
- Algunas justificaciones presentes
- Pipeline con algunos errores
- Métricas básicas reportadas
- Interpretación superficial

**Insuficiente (<5)**
- Técnicas incompletas o incorrectas
- Falta de justificación
- Data leakage presente
- Métricas inadecuadas
- Interpretación incorrecta

---

## 💡 Consejos y Buenas Prácticas

### ✅ DO
- Documentar todas las decisiones de preprocesado
- Usar sklearn Pipelines para evitar data leakage
- Validar con datos de test independientes
- Considerar el contexto de negocio al interpretar
- Guardar modelos y transformadores entrenados
- Usar control de versiones (Git)
- Realizar análisis de sensibilidad de hiperparámetros

### ❌ DON'T
- Aplicar fit() en datos de test
- Eliminar datos sin justificación
- Ignorar el desbalanceo de clases
- Usar solo accuracy como métrica
- Copiar código sin entender
- Olvidar la reproducibilidad (semillas aleatorias)
- Ignorar la matriz de costos del problema

---

## 📖 Referencias Útiles

### Documentación
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Feature-engine](https://feature-engine.readthedocs.io/)

### Tutoriales Recomendados
- [Handling Missing Data](https://scikit-learn.org/stable/modules/impute.html)
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Pipeline and ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [Imbalanced-learn](https://imbalanced-learn.org/stable/)

### Papers Relevantes
- SMOTE: Synthetic Minority Over-sampling Technique
- Random Forest: Breiman (2001)
- XGBoost: Chen & Guestrin (2016)

---

## ❓ Preguntas Frecuentes

**P: ¿Podemos usar otras librerías además de scikit-learn?**
R: Sí, pueden usar pandas, numpy, matplotlib, seaborn, feature-engine, imbalanced-learn, xgboost, lightgbm, catboost, etc. Deben incluir todas las dependencias en requirements.txt.

**P: ¿Cuántos features debemos crear?**
R: Mínimo 3 features nuevas además de las ya incluidas. La calidad es más importante que la cantidad - justificar por qué cada feature podría ser útil.

**P: ¿Debemos eliminar las variables de ruido identificadas?**
R: Sí, es parte del ejercicio de selección de features. Deben demostrar que identifican y eliminan features irrelevantes.

**P: ¿Cómo manejamos variables correlacionadas?**
R: Analicen la correlación, evalúen el impacto en los modelos con y sin ella, y tomen una decisión justificada.

**P: ¿Es obligatorio usar la matriz de costos?**
R: No, es opcional, aunque recomendable. Pueden considerar los costos diferentes de FP y FN en su análisis final y recomendaciones.

**P: ¿Podemos usar deep learning?**
R: El enfoque debe estar en los algoritmos vistos en clase (regresión logística, árboles, random forest, xgboost, svm, knn, naive bayes). No debe emplearse Deep learning ni ninguna otra te´cnica no vista en clase.

---

## 📧 Contacto y Soporte

Para dudas sobre el proyecto:
- Consultar el syllabus de los módulos
- Revisar la descripción detallada del dataset
- Contactar al instructor durante las sesiones

---

## 🎓 Créditos

**Dataset Original**: German Credit Data, UCI Machine Learning Repository  
**Dataset Ampliado**: Versión educativa con features adicionales para práctica de preprocesado y Clasificación
**Curso**: Máster en Inteligencia Artificial - MBIT School
**Módulos**: Preprocesado y Algoritmos de Clasificación
**Profesor**: Juan José Garcés Iniesta; jjgarcesiniesta@gmail.com 
---

**¡Buena suerte con el proyecto! 🚀**

---
# EOF (End Of File)