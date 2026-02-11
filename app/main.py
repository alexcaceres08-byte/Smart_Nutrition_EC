import streamlit as st
import pathlib
import time
import os


os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import pandas as pd
from fastai.vision.all import *
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Smart Nutrition AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix para Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --- 2. ESTILOS CSS ---
st.markdown("""
    <style>
    /* 1. APP PRINCIPAL: FONDO BLANCO, TEXTO NEGRO */
    .stApp { 
        background-color: #FFFFFF !important; 
        color: #000000 !important; 
    }
    
    p, h1, h2, h3, li, .stMarkdown, label {
        color: #000000 !important;
    }

    /* 2. CAJONES OSCUROS (TEXTO BLANCO) */
    div[data-baseweb="select"] div { color: white !important; }
    ul[data-baseweb="menu"] li span { color: white !important; }
    div[data-baseweb="popover"] div { color: white !important; }

    [data-testid="stFileUploader"] { color: white !important; }
    [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
        color: #ddd !important;
    }
    button[data-testid="baseButton-secondary"] {
        color: white !important;
        border-color: white !important;
    }
    
    /* 3. TARJETAS DE RESULTADO */
    .result-card {
        background-color: #F8F9FA;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #1A2980;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .result-card h3, .result-card p, .result-card div, .result-card b, .result-card span {
        color: #000000 !important;
    }

    /* 4. BARRA DE DETECCIÓN */
    .detection-bar {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 25px;
    }

    /* 5. TÍTULOS */
    h1 {
        background: -webkit-linear-gradient(45deg, #1A2980, #26D0CE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 6. BOTONES */
    .stButton>button {
        background: linear-gradient(45deg, #FF512F, #DD2476);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        height: 50px;
    }

    /* 7. SIDEBAR */
    section[data-testid="stSidebar"] { 
        background-color: #F0F2F6 !important; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. RUTAS ---
try:
    APP_PATH = pathlib.Path(__file__).resolve().parent
    MODELS_DIR = APP_PATH / 'models' 
    MLRUNS_PATH = APP_PATH / 'mlruns'
    DATA_PATH = APP_PATH.parent / 'data' / 'images'
    CSV_PATH = APP_PATH.parent / 'data' / 'labels.csv'

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_PATH.mkdir(parents=True, exist_ok=True)
except:
    st.error("Error crítico en rutas.")
    st.stop()

# --- 4. CARGA ---
def get_x(r): return DATA_PATH / r['fname']
def get_y(r): return r['labels'].split(' ')

@st.cache_resource
def load_learner_cached(model_path):
    return load_learner(model_path)

def rebuild_database():
    if not CSV_PATH.exists():
        data_list = []
        for folder in DATA_PATH.iterdir():
            if folder.is_dir():
                label = folder.name
                for img_file in folder.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        fname = f"{label}/{img_file.name}"
                        data_list.append({'fname': fname, 'labels': label})
        df = pd.DataFrame(data_list)
        # SINTAXIS EXPANDIDA PARA EVITAR ERRORES
        if not df.empty:
            df.to_csv(CSV_PATH, index=False)
        else:
            with open(CSV_PATH, 'w') as f:
                f.write("fname,labels\n")

rebuild_database()

# --- 5. CHEF INTELIGENTE ---
def buscar_receta_saludable(productos_detectados):
    items = set(productos_detectados)
    
    # COMBOS
    if "FIDEOS_SUMESA" in items and "ATUN_REAL" in items:
        return {
            "titulo": "🍝 Pasta Proteica del Mar",
            "ingredientes": "Fideos Sumesa + Atún Real",
            "preparacion": "Cocina los fideos al dente. Mezcla con atún, limón y aceite de oliva.",
            "beneficios": "<br><b>🧪 Ciencia:</b> Carbohidratos complejos + Proteína magra.<br><b>✅ Beneficios:</b><br>• Recuperación muscular.<br>• Energía sostenida (4 horas)."
        }

    if "LECHE_NUTRI" in items and "AVENA_QUAKER" in items and "BANANA" in items:
        return {
            "titulo": "🦍 Batido 'Gorilla Power'",
            "ingredientes": "1 Banana + 3 cdas Avena + Leche",
            "preparacion": "Licúa 45 seg. con hielo.",
            "beneficios": "<br><b>🧪 Ciencia:</b> Absorción rápida (Banana) + Lenta (Avena).<br><b>✅ Beneficios:</b><br>• Masa Muscular.<br>• Saciedad total."
        }

    if "LECHE_NUTRI" in items and "AVENA_QUAKER" in items:
        return {
            "titulo": "🥣 Porridge Cremoso",
            "ingredientes": "Avena Quaker + Leche Nutri",
            "preparacion": "Hierve la avena en leche por 5 min.",
            "beneficios": "<br><b>🧪 Ciencia:</b> El calor rompe almidones para mejor digestión.<br><b>✅ Beneficios:</b><br>• Huesos Fuertes (Calcio).<br>• Baja el Colesterol."
        }

    if "GUITIG" in items and "MANZANA" in items:
        return {
            "titulo": "🥂 Soda Natural",
            "ingredientes": "Güitig + Manzana picada",
            "preparacion": "Vaso con hielo, Güitig y trocitos de manzana.",
            "beneficios": "<br><b>🧪 Ciencia:</b> Sensación de gaseosa sin insulina.<br><b>✅ Beneficios:</b><br>• Hidratación Celular.<br>• Cero Calorías."
        }
    
    if "MANZANA" in items and "BANANA" in items:
        return {
            "titulo": "🥗 Dúo Frutal",
            "ingredientes": "Manzana + Banana",
            "preparacion": "Pica en cubos con gotas de limón.",
            "beneficios": "<br><b>🧪 Ciencia:</b> Potasio + Pectina.<br><b>✅ Beneficios:</b><br>• Energía Mental.<br>• Salud Dental."
        }

    if "LECHE_NUTRI" in items and "SNACKS" in items:
        return {
            "titulo": "⚖️ Balance Glucémico",
            "ingredientes": "Snack + Vaso de Leche",
            "preparacion": "Sirve el snack en plato. Acompaña con leche.",
            "beneficios": "<br><b>🧪 Ciencia:</b> La proteína frena el pico de azúcar.<br><b>✅ Beneficios:</b><br>• Menos Ansiedad.<br>• Aporte de Calcio."
        }
    
    if "GASEOSAS" in items and "ATUN_REAL" in items:
        return {
            "titulo": "🐟 Almuerzo Express",
            "ingredientes": "Atún Real + Gaseosa (con hielo)",
            "preparacion": "Atún con limón. Gaseosa diluida.",
            "beneficios": "<br><b>🧪 Ciencia:</b> Proteína pura + Energía rápida.<br><b>✅ Beneficios:</b><br>• Alerta Mental.<br>• Mantenimiento Muscular."
        }

    # INDIVIDUALES
    if "SNACKS" in items:
        return {"titulo": "🍿 Análisis de Snack", "ingredientes": "Producto Procesado", "preparacion": "⚠️ Sirve en plato pequeño.", "beneficios": "<br><b>🧪 Dato:</b> Energía densa y sodio.<br><b>✅ Tip:</b> Bebe agua después."}
    
    if "GASEOSAS" in items:
        return {"titulo": "🍹 Bebida Refrescante", "ingredientes": "Bebida Carbonatada", "preparacion": "Sírvela HELADA.", "beneficios": "<br><b>🧪 Dato:</b> Energía líquida.<br><b>✅ Tip:</b> Ideal 15 min antes de entrenar."}
    
    if "BANANA" in items:
        return {"titulo": "🍌 Barra Natural", "ingredientes": "Banana", "preparacion": "Comer directo.", "beneficios": "<br><b>🧪 Ciencia:</b> Potasio y B6.<br><b>✅ Beneficios:</b> Cero calambres."}
    
    if "MANZANA" in items:
        return {"titulo": "🍎 Joya Nutricional", "ingredientes": "Manzana", "preparacion": "Lavar y comer con cáscara.", "beneficios": "<br><b>🧪 Ciencia:</b> Ácido Ursólico.<br><b>✅ Beneficios:</b> Quema grasa y limpia dientes."}
    
    if "ATUN_REAL" in items:
        return {"titulo": "🥗 Proteína Líquida", "ingredientes": "Atún", "preparacion": "Con limón y sal.", "beneficios": "<br><b>🧪 Ciencia:</b> 95% Proteína limpia.<br><b>✅ Beneficios:</b> Músculo puro sin grasa."}
    
    if "LECHE_NUTRI" in items:
        return {"titulo": "🥛 Oro Blanco", "ingredientes": "Leche", "preparacion": "Tibia o fría.", "beneficios": "<br><b>🧪 Ciencia:</b> Hidratación con electrolitos.<br><b>✅ Beneficios:</b> Recuperación nocturna."}
    
    if "GUITIG" in items:
        return {"titulo": "💧 Agua Mineral", "ingredientes": "Güitig", "preparacion": "Con limón.", "beneficios": "<br><b>🧪 Ciencia:</b> Gas natural digestivo.<br><b>✅ Beneficios:</b> Digestión rápida."}
    
    if "FIDEOS_SUMESA" in items:
         return {"titulo": "🍝 Carga de Glucógeno", "ingredientes": "Pasta", "preparacion": "Al Dente.", "beneficios": "<br><b>🧪 Ciencia:</b> Índice glucémico medio.<br><b>✅ Beneficios:</b> Energía sin sueño."}
    
    if "AVENA_QUAKER" in items:
         return {"titulo": "🥞 Supergrano", "ingredientes": "Avena", "preparacion": "Remojada.", "beneficios": "<br><b>🧪 Ciencia:</b> Beta-glucanos.<br><b>✅ Beneficios:</b> Elimina colesterol."}
    
    return None

# --- 6. ENTRENAMIENTO ---
def train_new_version():
    mlflow.set_tracking_uri(MLRUNS_PATH.as_uri())
    client = MlflowClient()
    
    try:
        latest = client.get_latest_versions("Inventario_Productos_EC")
        ver = max([int(v.version) for v in latest]) + 1 if latest else 1
    except: ver = 1
    
    print(f"ℹ️ Creando Versión {ver}...")
    
    p_acc = 0.0
    try:
        exp = client.get_experiment_by_name("Smart_Inventory_Ecuador")
        if exp:
            runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
            if runs: p_acc = runs[0].data.metrics.get("accuracy", 0.0)
    except: pass

    try:
        df = pd.read_csv(CSV_PATH)
        dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                           get_x=get_x, get_y=get_y,
                           item_tfms=Resize(224),
                           batch_tfms=aug_transforms())
        
        # FIX CRÍTICO: num_workers=0
        dls = dblock.dataloaders(df, bs=8, num_workers=0)
        
        learn = vision_learner(dls, resnet18, metrics=accuracy_multi)
        
        mlflow.set_experiment("Smart_Inventory_Ecuador")
        EPOCHS = 4
        
        with mlflow.start_run(run_name=f"Train_v{ver}"):
            learn.fine_tune(EPOCHS)
            model_name = f"model_v{ver}.pkl"
            learn.export(MODELS_DIR / model_name)
            
            new_acc = float(learn.recorder.values[-1][2])
            mlflow.log_metric("accuracy", new_acc)
            try: mlflow.pytorch.log_model(learn.model, "model", registered_model_name="Inventario_Productos_EC")
            except: pass

            diff = new_acc - p_acc
            icon = "📈" if diff >= 0 else "🔻"
            bar = "█" * 40 
            
            print(f"\n{bar}")
            print(f"📊 REPORTE DE EVOLUCIÓN (Versión {ver})")
            print(f"{bar}")
            print(f"   ANT: {p_acc:.2%} (Epochs: 4)")
            print(f"   NEW: {new_acc:.2%} (Epochs: {EPOCHS})")
            print(f"{bar}")
            print(f"   RESULTADO: {icon} Mejora de {diff:+.2%}")
            print(f"{bar}\n")
            return model_name
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- 7. INTERFAZ ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2921/2921822.png", width=80)
st.sidebar.title("🎛️ Centro de Control")

m_files = sorted(list(MODELS_DIR.glob("model_v*.pkl")), key=lambda f: f.stat().st_mtime) if MODELS_DIR.exists() else []
learn = None

if m_files:
    sel_m = st.sidebar.selectbox("Versión Activa:", [f.name for f in m_files], index=len(m_files)-1)
    try:
        learn = load_learner_cached(MODELS_DIR / sel_m)
        st.sidebar.success(f"✅ Cerebro: **{sel_m}**")
    except: st.sidebar.error("Error cargando modelo.")
else: st.sidebar.warning("⚠️ Sin modelos.")

st.sidebar.markdown("---")

st.title("🥗 Smart Nutrition AI")
st.markdown("Tu asistente personal de inventario y cocina saludable.")

col1, col2 = st.columns([1, 1.5])

uploaded_file = st.file_uploader("Sube tu foto aquí", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file and learn:
    st.sidebar.markdown("### 🚀 Corrección")
    st.sidebar.info("¿Me equivoqué? Selecciona lo correcto abajo:")
    
    opciones = list(learn.dls.vocab)
    correctos = st.sidebar.multiselect("Etiqueta Real:", opciones)
    
    if st.sidebar.button("💾 Guardar y APRENDER"):
        if not correctos:
            st.sidebar.error("¡Selecciona la etiqueta!")
        else:
            with st.status("⚙️ Re-entrenando...", expanded=True) as status:
                primaria = correctos[0]
                dest = DATA_PATH / primaria
                dest.mkdir(parents=True, exist_ok=True)
                fname = f"auto_{time.strftime('%H%M%S')}.jpg"
                
                uploaded_file.seek(0)
                with open(dest / fname, "wb") as f: f.write(uploaded_file.getbuffer())
                with open(CSV_PATH, 'a') as f: f.write(f"{primaria}/{fname},{' '.join(correctos)}\n")
                
                new_m = train_new_version()
                if new_m:
                    status.update(label="¡Listo!", state="complete")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    status.update(label="Error en entrenamiento.", state="error")
else:
    st.sidebar.info("👆 Sube una foto para ver opciones de corrección.")

if uploaded_file:
    uploaded_file.seek(0)
    img = PILImage.create(uploaded_file)
    with col1:
        st.image(img, caption="Producto Analizado", use_container_width=True)

    if learn:
        pred, idx, probs = learn.predict(img)
        labels = learn.dls.vocab
        detected = [l.upper() for l, p in zip(labels, probs) if p > 0.8]
        
        with col2:
            if detected:
                st.markdown(f"""<div class="detection-bar">🔍 DETECTADO: {' + '.join(detected)}</div>""", unsafe_allow_html=True)
                receta = buscar_receta_saludable(detected)
                if receta:
                    st.markdown(f"""
                    <div class="result-card">
                        <h3 style="color:#1A2980; margin-top:0;">💡 {receta['titulo']}</h3>
                        <p><b>🥘 Ingredientes:</b> {receta['ingredientes']}</p>
                        <p><b>👨‍🍳 Preparación:</b> {receta['preparacion']}</p>
                        <hr style="border-color:#ddd;">
                        <div>{receta['beneficios']}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No estoy seguro. Usa la barra lateral para corregirme.")
            
            with st.expander("📊 Ver Probabilidades Técnicas", expanded=True):
                probs_map = {l: float(p) for l, p in zip(labels, probs)}
                sorted_probs = sorted(probs_map.items(), key=lambda x: x[1], reverse=True)
                for label, prob in sorted_probs:
                    if prob > 0.01:
                        c1, c2 = st.columns([3, 4])
                        c1.markdown(f"<span style='color:black; font-weight:bold;'>{label.upper()}: {prob:.1%}</span>", unsafe_allow_html=True)
                        c2.progress(prob)
else:

    st.info("👆 Sube una foto de tus ingredientes para comenzar.")
