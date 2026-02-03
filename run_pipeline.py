import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.feature_extraction import extract_features
from src.preprocessing import preprocess_features
from src.pca_reduction import apply_pca
from src.ga_optimizer import GeneticAlgorithm
from src.lstm_model import LSTMClassifier
from src.inference import run_inference


# =====================
# CONFIG
# =====================
DATA_PATH = "data/raw/dataset.csv"

SMILES_COL = "SMILES"
TARGET_COL = "KELAS"

FEATURES = [
    'C','O','B','N','P','S','F','Cl','Br','Z','@','I',
    'COC','=','[O-]','C=C','N+','C=O',':','.', '-', '+','[]','()','#'
]

N_PCA_COMPONENTS = 20
RANDOM_STATE = 42


# =====================
# LOAD DATA
# =====================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Available columns:", df.columns.tolist())


# =====================
# FEATURE EXTRACTION
# =====================
print("Extracting features...")
X_raw = df[SMILES_COL].apply(
    lambda x: extract_features(x, FEATURES)
)

X = pd.DataFrame(X_raw.tolist())


# =====================
# LABEL ENCODING (MULTI-CLASS)
# =====================
print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])

print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
num_classes = len(le.classes_)


# =====================
# PREPROCESSING
# =====================
print("Preprocessing...")
X_scaled = preprocess_features(X)


# =====================
# PCA
# =====================
print("Applying PCA...")
X_pca, pca_model = apply_pca(
    X_scaled,
    n_components=N_PCA_COMPONENTS
)


# =====================
# GENETIC ALGORITHM
# =====================
print("Running Genetic Algorithm...")
ga = GeneticAlgorithm(
    population_size=20,
    generations=15,
    random_state=RANDOM_STATE
)

best_features, best_score = ga.fit(X_pca, y)
print("Best GA score:", best_score)

X_selected = X_pca[:, best_features]


# =====================
# TRAIN / TEST SPLIT
# =====================
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


# =====================
# RESHAPE FOR LSTM
# =====================
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# =====================
# TRAIN LSTM
# =====================
print("Training LSTM (multiclass)...")
model = LSTMClassifier(
    input_shape=(X_train.shape[1], 1),
    num_classes=num_classes
)

model.train(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=50,
    batch_size=32
)


# =====================
# EVALUATION
# =====================
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)


# =====================
# INFERENCE
# =====================
print("Inference on test samples...")
preds = run_inference(model, X_test[:5])

decoded_preds = le.inverse_transform(preds)

print("Predicted classes (encoded):", preds)
print("Predicted classes (original):", decoded_preds)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.feature_extraction import extract_features
from src.preprocessing import preprocess_features
from src.pca_reduction import apply_pca
from src.ga_optimizer import GeneticAlgorithm
from src.lstm_model import LSTMClassifier
from src.inference import run_inference


# =====================
# CONFIG
# =====================
DATA_PATH = "data/raw/dataset.csv"

SMILES_COL = "SMILES"
TARGET_COL = "KELAS"

FEATURES = [
    'C','O','B','N','P','S','F','Cl','Br','Z','@','I',
    'COC','=','[O-]','C=C','N+','C=O',':','.', '-', '+','[]','()','#'
]

N_PCA_COMPONENTS = 20
RANDOM_STATE = 42


# =====================
# LOAD DATA
# =====================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Available columns:", df.columns.tolist())


# =====================
# FEATURE EXTRACTION
# =====================
print("Extracting features...")
X_raw = df[SMILES_COL].apply(
    lambda x: extract_features(x, FEATURES)
)

X = pd.DataFrame(X_raw.tolist())


# =====================
# LABEL ENCODING (MULTI-CLASS)
# =====================
print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])

print("Class mapping:", dict(zip(le.classes_, range(len(le.classes_)))))
num_classes = len(le.classes_)


# =====================
# PREPROCESSING
# =====================
print("Preprocessing...")
X_scaled = preprocess_features(X)


# =====================
# PCA
# =====================
print("Applying PCA...")
X_pca, pca_model = apply_pca(
    X_scaled,
    n_components=N_PCA_COMPONENTS
)


# =====================
# GENETIC ALGORITHM
# =====================
print("Running Genetic Algorithm...")
ga = GeneticAlgorithm(
    population_size=20,
    generations=15,
    random_state=RANDOM_STATE
)

best_features, best_score = ga.fit(X_pca, y)
print("Best GA score:", best_score)

X_selected = X_pca[:, best_features]


# =====================
# TRAIN / TEST SPLIT
# =====================
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


# =====================
# RESHAPE FOR LSTM
# =====================
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# =====================
# TRAIN LSTM
# =====================
print("Training LSTM (multiclass)...")
model = LSTMClassifier(
    input_shape=(X_train.shape[1], 1),
    num_classes=num_classes
)

model.train(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=50,
    batch_size=32
)


# =====================
# EVALUATION
# =====================
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)


# =====================
# INFERENCE
# =====================
print("Inference on test samples...")
preds = run_inference(model, X_test[:5])

decoded_preds = le.inverse_transform(preds)

print("Predicted classes (encoded):", preds)
print("Predicted classes (original):", decoded_preds)

