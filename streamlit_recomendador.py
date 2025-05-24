import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# Token de Hugging Face directamente (¡no subir a GitHub!)
HF_TOKEN = "hf"

st.set_page_config(page_title="🎮 Recomendador de Juegos de Steam", layout="centered")


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

# ---------- CLIENTE DE HUGGING FACE (MODELO MEJORADO) ----------
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_TOKEN)  # Modelo más potente

# ---------- FUNCIONES MEJORADAS ----------
def generar_explicacion(juego_base, recomendados):
    prompt = f"""<s>
[INST] Eres un analista experto en videojuegos deportivos. El usuario amó '{juego_base}'.

Explica por qué estos 5 juegos son buenas recomendaciones:
{', '.join(recomendados)}

**Estructura obligatoria:**
1. Para CADA juego de la lista:
   - Género y características clave
   - Una similitud concreta con el juego base

Ejemplo de similitudes válidas: mecánicas de juego, modo carrera, personalización, competición online, licencias oficiales. [/INST]
"""

    respuesta = client.text_generation(
    prompt,
    max_new_tokens=800,  # Aumentado para cubrir 5 juegos
    temperature=0.65,
    repetition_penalty=1.1,
    stop_sequences=["</s>"]  # Evita truncamiento prematuro
    )
    return respuesta.strip()

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
# ---------- INTERFAZ STREAMLIT ----------
st.title("🎮 Recomendador de Juegos de Steam")
st.markdown("Selecciona un juego que te guste y descubre otros similares. Además, recibe una explicación detallada de por qué estos juegos son perfectos para ti.")

juego_seleccionado = st.selectbox("Selecciona un juego que te haya gustado,puedes escribir la inicial para facilitar la búsqueda🫢:", sorted(juegos_df['game_title'].dropna().unique()))

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
