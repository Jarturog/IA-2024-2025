import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score

RANDOM_STATE = 42

"""
# Práctica Bloc III - Análisis del Forest Cover Type Dataset

## Introducción
Este notebook presenta un análisis detallado del Forest Cover Type Dataset utilizando diferentes técnicas de aprendizaje automático. El objetivo es predecir el tipo de cobertura forestal basándonos en variables cartográficas.

## Carga y exploración inicial de datos
"""
# Cargar el dataset
df = pd.read_csv("covtype.csv")

## Análisis Exploratorio de Datos (EDA)

wilderness_area_columns = ["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"]

soil_type_columns = [
    "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8",
    "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15",
    "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22",
    "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
    "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36",
    "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"
]

# Seleccionar las columnas de interés
numeric_columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"
]

# Información básica del dataset
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Comprobar si tiene valores nulos
print("\nA continuación se muestran las columnas que tienen valores nulos:")
if df.isnull().sum().sum() == 0:
    print("No hay valores nulos en el dataset.")
else:
    texto_columnas = ""
    for columna in df.columns:
        n_valores_nulos = df[columna].isnull().sum()
        if n_valores_nulos > 0:
            texto_columnas += f"{columna}: {n_valores_nulos} ({n_valores_nulos / df.shape[0] * 100:.2f}%)\n"
    print(texto_columnas)

# Comprobar si tiene valores negativos
print("\nA continuación se muestran las columnas que tienen valores negativos:")
if (df < 0).sum().sum() == 0:
    print("No hay valores negativos en el dataset.")
else:
    texto_columnas = ""
    for columna in df.columns:
        n_valores_negativos = df[df[columna] < 0].shape[0]
        if n_valores_negativos > 0:
            texto_columnas += f"{columna}: {n_valores_negativos} ({n_valores_negativos / df.shape[0] * 100:.2f}%)\n"
    print(texto_columnas)

columnasConMasOMenosDeUnSoilType = df[soil_type_columns].sum(axis=1) != 1
samples_with_multiple_soil_type = df[columnasConMasOMenosDeUnSoilType]
print("Número de muestras con múltiples tipos de suelo:", len(samples_with_multiple_soil_type))

columnasConMasOMenosDeUnWildernessArea = df[wilderness_area_columns].sum(axis=1) != 1
samples_with_multiple_wilderness_area = df[columnasConMasOMenosDeUnWildernessArea]
print("Número de muestras con múltiples áreas naturales:", len(samples_with_multiple_wilderness_area))

# 1. Distribuciones
def plot_distribtions(dataset):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=dataset, x='Cover_Type')
    plt.title('Distribución de tipos de cobertura forestal')
    plt.xlabel('Tipo de cobertura')
    plt.ylabel('Cantidad')
    plt.show()

    n_rows = 2
    n_cols = 2
    plots_por_figura = n_rows * n_cols

    # Iterar sobre las columnas en grupos de 4 (2x2)
    for i in range(0, len(numeric_columns), plots_por_figura):
        # Crear una figura con 2x2 subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()  # Convertir la matriz 2x2 en un array 1D para facilitar la iteración

        # Iterar sobre cada columna en el grupo actual
        for j in range(plots_por_figura):
            if i + j < len(numeric_columns):
                columna = numeric_columns[i + j]
                sns.histplot(data=dataset, x=columna, ax=axes[j])
                axes[j].set_title('Distribución de ' + columna)
                axes[j].set_xlabel(columna)
                axes[j].set_ylabel('Cantidad')
            else:
                # Si no hay más columnas, ocultar el subplot vacío
                axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(15, 4))

    # Wilderness areas plot - 1/4 width
    plt.subplot(1, 4, 1)  # 1 row, 4 columns, position 1
    wilderness_counts = dataset[wilderness_area_columns].sum()
    sns.barplot(x=[i.split('Wilderness_Area')[-1] for i in wilderness_counts.index], y=wilderness_counts.values)
    plt.title('Wilderness Areas')
    plt.xlabel('Area')

    # Soil types plot - 3/4 width
    plt.subplot(1, 4, (2, 4))  # 1 row, 4 columns, spans positions 2-4
    soil_counts = dataset[soil_type_columns].sum()
    sns.barplot(x=[i.split('Soil_Type')[-1] for i in soil_counts.index], y=soil_counts.values)
    plt.title('Soil Types')
    plt.xlabel('Type')

    plt.tight_layout()
    plt.show()


plot_distribtions(df)


def create_balanced_subset(dataset, target_samples_per_class):
    # Get the minimum number of samples per class (excluding very small classes)
    class_counts = dataset['Cover_Type'].value_counts()

    # Create undersampler
    rus = RandomUnderSampler(sampling_strategy={
        i: min(count, target_samples_per_class)
        for i, count in class_counts.items()
    }, random_state=RANDOM_STATE)

    # Undersample
    X = dataset.drop('Cover_Type', axis=1)
    y = dataset['Cover_Type']
    X_resampled, y_resampled = rus.fit_resample(X, y)

    return pd.concat([X_resampled, y_resampled], axis=1)


min_class_count = df['Cover_Type'].value_counts().min()
print(f"Smallest class has {min_class_count} samples")

# Create balanced dataset with that size
balanced_df = create_balanced_subset(df, min_class_count)
print(f"Balanced dataset has {balanced_df.shape[0]} samples")
plot_distribtions(balanced_df)

print("Todas las distribuciones menos Elevation, Wilderness Area y Soil Type siguen la misma distribución en el dataset balanceado.")

# 2. Matriz de correlación
plt.figure(figsize=(12, 8))
corr_matrix = balanced_df.corr()
sns.heatmap(corr_matrix, cmap='Purples', annot=False)
plt.title('Matriz de correlaciones')
plt.show()

## Preprocesamiento de datos y reducción de dimensionalidad
# Separar features y target
y = balanced_df['Cover_Type']
X = balanced_df.drop('Cover_Type', axis=1)

# Estandarización
# Separar variables numéricas y categóricas
numeric_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                   'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                   'Horizontal_Distance_To_Fire_Points', 'Hillshade_9am',
                   'Hillshade_Noon', 'Hillshade_3pm']

categorical_features = [col for col in X.columns if col not in numeric_features]

# Escalar solo las variables numéricas
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

pca = PCA(n_components=0.9, random_state=RANDOM_STATE) # Seleccionar componentes que expliquen el 90% de la varianza
X = pca.fit_transform(X)

print(f"PCA ha seleccionado {pca.n_components_} componentes principales.")

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)



## Entrenamiento y optimización de modelos
# Definición de modelos y parámetros para optimización
models = {
    'Perceptron': {
        'model': Perceptron(),
        'params': {
            'max_iter': [1000],
            'alpha': [0.0001, 0.001, 0.01],
            'eta0': [10.0, 1.0, 0.1]
        }
    },
    'Regresión logística': {
        'model': LogisticRegression(),
        'params': {
            'max_iter': [1000],
            'C': [0.1, 1, 10]
        }
    },
    'Máquina de vector de soporte (SVM) lineal': {
        'model': SVC(),
        'params': {
            'kernel': ['linear'],
            'C': [0.1, 1, 10]
        }
    },
    'Máquina de vector de soporte (SVM) gaussiano': {
        'model': SVC(),
        'params': {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1, 10]
        }
    },
#    'Máquina de vector de soporte (SVM) polinómico': {
#        'model': SVC(),
#        'params': {
#            'kernel': ['poly'],
#            'C': [0.1, 1, 10],
#            'gamma': ['scale', 'auto', 1], # reducir el número de combinaciones para ahorrar tiempo
#            'degree': [2, 3, 4]
#        }
#    },
    'Árbol de decisión': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    },
    'Bosque aleatorio': {
        'model': RandomForestClassifier(),
        'params': {
            'max_depth': [None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
            'n_estimators': [50, 100, 200]
        }
    }
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_results = {}
for name, model_info in models.items():
    print(f"\nOptimizando {name}...")
    grid = GridSearchCV(model_info['model'],
                        model_info['params'],
                        cv=cv,
                        scoring='balanced_accuracy', # se usa accuracy y no F1 para priorizar velocidad
                        n_jobs=-1)
    grid.fit(X_train, y_train)

    # Evaluación en conjunto de prueba
    y_pred = grid.predict(X_test)

    # Guardar resultados
    model_results[name] = {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'best_model': grid.best_estimator_,
        'test_score': accuracy_score(y_test, y_pred)
    }

    print(f"Mejores parámetros: {grid.best_params_}")
    print(f"Precisión en validación: {grid.best_score_:.3f}")
    print(f"Precisión en test: {model_results[name]['test_score']:.3f}")


def plot_performance(n_rows = 2, n_cols = 2):
    plots_por_figura = n_rows * n_cols

    for i in range(0, len(model_results), plots_por_figura):
        # Crear una figura con 2x2 subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()  # Convertir la matriz 2x2 en un array 1D para facilitar la iteración

        # Iterar sobre cada columna en el grupo actual
        for j in range(plots_por_figura):
            if i + j < len(model_results):
                model_name = list(model_results.keys())[i + j]
                model = model_results[model_name]['best_model']
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)  # calcula l'accuracy
                # calcula la taula de mètriques
                class_report = classification_report(y_test, y_pred, target_names=model.classes_, output_dict=True)
                # Crea un DataFrame de pandas per a una visualització més neta
                df_class_report = pd.DataFrame(class_report).transpose()
                # Visualitza la taula de mètriques amb seaborn
                sn.heatmap(df_class_report.iloc[:-3, :-1], annot=True, fmt=".2%", cmap="Purples", cbar=False,
                           linewidths=0.5, linecolor='black', annot_kws={"size": 10}, ax=axes[j])
                # Aplica etiquetes i títol
                axes[j].set_xlabel('\nMétricas')
                axes[j].set_ylabel('Clases')
                axes[j].set_title(f'{model_name}\n\nAccuracy: {accuracy:.2%}')
            else:
                # Si no hay más columnas, ocultar el subplot vacío
                axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    for i in range(0, len(model_results), plots_por_figura):
        # Crear una figura con 2x2 subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()  # Convertir la matriz 2x2 en un array 1D para facilitar la iteración

        # Iterar sobre cada columna en el grupo actual
        for j in range(plots_por_figura):
            if i + j < len(model_results):
                model_name = list(model_results.keys())[i + j]
                model = model_results[model_name]['best_model']
                y_pred = model.predict(X_test)  # calcula les prediccions
                conf_mat = confusion_matrix(y_test, y_pred)  # Calcula la matriu de confusió
                # Visualitza la matriu de confusió amb seaborn
                sn.heatmap(conf_mat, annot=True, fmt="d", cmap="Purples", cbar=False, xticklabels=model.classes_,
                           yticklabels=model.classes_, ax=axes[j])
                # Aplica etiquetes i títol
                axes[j].set_xlabel('Classes Predicció')
                axes[j].set_ylabel('Classes Reals')
                axes[j].set_title(f'Matriu de Confusió - {model_name}')

                # Resalta en vermell els errors
                for k in range(conf_mat.shape[0]):
                    for l in range(conf_mat.shape[1]):
                        if k != l and conf_mat[k, l] != 0:  # Resalta només les posicions fora de la diagonal
                            axes[j].text(l + 0.5, k + 0.5, str(conf_mat[k, l]), color='red', ha='center', va='center')
            else:
                # Si no hay más columnas, ocultar el subplot vacío
                axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()


plot_performance()
