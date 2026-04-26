import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf, iris.target_names

model, target_names = load_model()

st.title("🌸 Классификация ирисов")
st.markdown("Введите параметры цветка, чтобы определить вид:")

sepal_length = st.slider("Длина чашелистика (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.slider("Ширина чашелистика (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Длина лепестка (cm)", 1.0, 7.0, 4.3, 0.1)
petal_width  = st.slider("Ширина лепестка (cm)", 0.1, 2.5, 1.3, 0.1)

if st.button("Предсказать вид"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    st.success(f"**Результат:** {target_names[prediction]} ({prediction})")
    proba = model.predict_proba(input_data)[0]
    proba_df = pd.DataFrame({
        "Вид": target_names,
        "Вероятность": proba
    })
    st.bar_chart(proba_df.set_index("Вид"))