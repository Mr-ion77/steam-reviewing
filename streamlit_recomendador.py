import streamlit as st
import pandas as pd
import pickle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Autenticación en Hugging Face
login("hf")  # Reemplazar con el mio pero no se puede subir a github 

st.set_page_config(page_title="Recomendador de Juegos", layout="centered")

# ---------- CARGA DE DATOS ----------
@st.cache_resource
def cargar_datos():
    df = pd.read_csv("juegos_con_vectores.csv")
    with open("combined_features.pkl", "rb") as f:
        combined = pickle.load(f)
    with open("similitud_combinada.pkl", "rb") as f:
        similitud = pickle.load(f)
    return df, combined, similitud

juegos_df, combined_features, similitud_coseno = cargar_datos()

# ---------- CARGA DEL MODELO MISTRAL ----------
@st.cache_resource
def cargar_mistral():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, use_auth_token=True)
    return tokenizer, model

tokenizer, model = cargar_mistral()

# ---------- FUNCIONES ----------
def recomendar_juegos(nombre_juego, top_n=5):
    indices_juegos = pd.Series(juegos_df.index, index=juegos_df['game_title'].str.lower().fillna("desconocido")).drop_duplicates()
    nombre_juego = nombre_juego.lower()
    if nombre_juego not in indices_juegos:
        return []
    
    idx = indices_juegos[nombre_juego]
    similitudes = list(enumerate(similitud_coseno[idx]))
    similitudes = sorted(similitudes, key=lambda x: x[1], reverse=True)[1:top_n+1]
    juegos_recomendados = [juegos_df.iloc[i[0]]['game_title'] for i in similitudes]
    return juegos_recomendados

def generar_explicacion(juego_base, recomendados):
    prompt = (
        f"He jugado a '{juego_base}' y me ha gustado. ¿Podrías explicarme por qué me podrían gustar también estos juegos: "
        f"{', '.join(recomendados)}?"
    )
    full_prompt = f"[INST] User: {prompt} [/INST]"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    respuesta = decoded.split("[/INST]")[-1].strip()
    return respuesta

# ---------- INTERFAZ STREAMLIT ----------
st.title("🎮 Recomendador de Juegos de Steam")
st.markdown("Selecciona un juego y descubre otros similares. Además, el chatbot te explicará por qué podrían gustarte.")

juego_seleccionado = st.selectbox("Selecciona un juego que te haya gustado:", sorted(juegos_df['game_title'].dropna().unique()))

if st.button("🔍 Recomendar"):
    recomendaciones = recomendar_juegos(juego_seleccionado)
    
    if not recomendaciones:
        st.error("No se encontraron recomendaciones.")
    else:
        st.subheader("🔝 Juegos recomendados:")
        for j in recomendaciones:
            st.markdown(f"- {j}")
        
        st.subheader("🤖 Explicación del Chatbot")
        with st.spinner("Generando explicación..."):
            explicacion = generar_explicacion(juego_seleccionado, recomendaciones)
        st.success("Explicación generada:")
        st.write(explicacion)
