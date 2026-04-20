# =========================================================
# 0. IMPORTS + SETUP
# =========================================================
import numpy as np
import pandas as pd
import os
import cv2
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Dense, Flatten

tf.random.set_seed(42)
np.random.seed(42)

# =========================================================
# 1. TABULAR DATA
# =========================================================
df = pd.read_csv(r"D:\multimodel\heart.csv")
df = pd.get_dummies(df, drop_first=True).astype(np.float32)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 2. TABULAR MODEL
# =========================================================
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)
tabular_prob = xgb.predict_proba(X)[:, 1]

# =========================================================
# 3. X-RAY MODEL (STABLE)
# =========================================================
IMG_SIZE = 128

def load_xray(folder):
    X_img, y_img = [], []
    for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
        path = os.path.join(folder, cls)
        if not os.path.exists(path):
            continue

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
            X_img.append(image)
            y_img.append(label)

    return np.array(X_img), np.array(y_img)

X_xray, y_xray = load_xray(r"D:\multimodel\chest_xray\train")

inputs = Input(shape=(128,128,3))
x = Conv2D(32, 3, activation="relu")(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, 3, activation="relu")(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)

xray_model = Model(inputs, outputs)
xray_model.compile(optimizer="adam", loss="binary_crossentropy")

xray_model.fit(X_xray, y_xray, epochs=12, batch_size=32, validation_split=0.2)

xray_model(np.zeros((1,128,128,3)))

xray_prob = xray_model.predict(X_xray).flatten()

# prevent collapse
xray_prob = np.clip(xray_prob, 1e-5, 1-1e-5)

# =========================================================
# 4. ECG MODEL (CRITICAL FIX FOR LABELS)
# =========================================================
ecg_train = pd.read_csv(r"D:\multimodel\ecg\mitbih_train.csv")

X_ecg = ecg_train.iloc[:, :-1].values
y_ecg = ecg_train.iloc[:, -1].values

# IMPORTANT: ensure binary classification for stability
y_ecg = (y_ecg > 0).astype(int)

X_ecg = X_ecg.reshape(-1, X_ecg.shape[1], 1)

ecg_model = tf.keras.Sequential([
    Conv1D(32, 3, activation="relu", input_shape=(X_ecg.shape[1],1)),
    MaxPooling1D(),
    Conv1D(64, 3, activation="relu"),
    MaxPooling1D(),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

ecg_model.compile(optimizer="adam", loss="binary_crossentropy")

ecg_model.fit(X_ecg, y_ecg, epochs=12, batch_size=32, validation_split=0.2)

ecg_prob = ecg_model.predict(X_ecg).flatten()

ecg_prob = np.clip(ecg_prob, 1e-5, 1-1e-5)

# =========================================================
# 5. ALIGNMENT + NORMALIZATION
# =========================================================
min_len = min(len(tabular_prob), len(xray_prob), len(ecg_prob))

fusion_X = np.column_stack([
    tabular_prob[:min_len],
    xray_prob[:min_len],
    ecg_prob[:min_len]
])

fusion_y = y[:min_len]

fusion_X = np.log1p(fusion_X)   # stability boost

scaler = MinMaxScaler()
fusion_X = scaler.fit_transform(fusion_X)

# =========================================================
# 6. FUSION MODEL
# =========================================================
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    fusion_X,
    fusion_y,
    test_size=0.2,
    random_state=42,
    stratify=fusion_y
)

fusion_model = LogisticRegression(class_weight="balanced")
fusion_model.fit(Xf_train, yf_train)

pred = fusion_model.predict(Xf_test)
prob = fusion_model.predict_proba(Xf_test)[:, 1]

print("\n🔥 FINAL FUSION RESULT")
print(classification_report(yf_test, pred))
print("ROC-AUC:", roc_auc_score(yf_test, prob))

joblib.dump(fusion_model, "fusion_model.pkl")

# =========================================================
# 7. SHAP (SAFE)
# =========================================================
explainer = shap.Explainer(xgb, X)
shap_values = explainer(X.astype(np.float32))
shap.summary_plot(shap_values, X)

# =========================================================
# 8. LIME
# =========================================================
lime = LimeTabularExplainer(
    X.values,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    mode="classification"
)

exp = lime.explain_instance(X.iloc[0].values, xgb.predict_proba, num_features=5)
exp.as_pyplot_figure()
plt.show()

# =========================================================
# 9. GRAD-CAM SAFE
# =========================================================
def grad_cam(model, img):
    last_conv = None

    for layer in model.layers[::-1]:
        if isinstance(layer, Conv2D):
            last_conv = layer
            break

    grad_model = Model(
        inputs=model.input,
        outputs=[last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img)
        loss = pred[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    cam = tf.reduce_sum(pooled * conv_out[0], axis=-1)
    cam = np.maximum(cam, 0)
    cam /= (np.max(cam) + 1e-8)

    return cam

sample = X_xray[0].reshape(1,128,128,3)
heatmap = grad_cam(xray_model, sample)

heatmap = cv2.resize(heatmap, (128,128))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

orig = (sample[0]*255).astype(np.uint8)
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

plt.imshow(overlay)
plt.title("Grad-CAM")
plt.axis("off")
plt.show()

# =========================================================
# 10. FINAL OUTPUT
# =========================================================
for i in range(5):
    print("\nPatient", i)
    print("Tabular:", tabular_prob[i])
    print("X-ray:", xray_prob[i])
    print("ECG:", ecg_prob[i])

    final = fusion_model.predict([[fusion_X[i][0], fusion_X[i][1], fusion_X[i][2]]])[0]
    print("🔥 FINAL RISK:", final)

import joblib

joblib.dump(xgb,"xgb.pkl")
joblib.dump(fusion_model,"fusion_model.pkl")

xray_model.save("xray_model.h5")
ecg_model.save("ecg_model.h5")
