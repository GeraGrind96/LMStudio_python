import subprocess
import requests
import socket
import os
import time
import signal
import sys
import psutil
from openai import OpenAI
import json
from datetime import datetime
from collections import deque

LMSTUDIO_API_URL = "http://localhost:3000/v1/models"
MODEL_NAME = ""
LMS_PATH = "/home/gerardo/.lmstudio/bin/lms"
LM_STUDIO_APPIMAGE = "/home/robolab/robocomp/components/robocomp-shadow/agents/LLM_agent/LM-Studio-0.3.20-4-x64.AppImage"

server_proc = None
modelo_proc = None
appimage_proc = None

# Variable global para guardar el historial de mensajes
messages = []
expended_times = []

memory_limit = 6
messages_memory = deque(maxlen=memory_limit)

model_name = ""

# ------------------- NUEVO -------------------
def listar_json_dataset():
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.exists(dataset_dir):
        print(f"❌ No existe la carpeta: {dataset_dir}")
        return []
    archivos_json = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
    return [os.path.join(dataset_dir, f) for f in archivos_json]

def seleccionar_json():
    archivos = listar_json_dataset()
    if not archivos:
        print("⚠️ No se encontraron archivos .json en la carpeta 'dataset'.")
        return None
    print("\n📂 Archivos JSON disponibles en 'dataset':")
    for i, archivo in enumerate(archivos):
        print(f"{i+1}. {os.path.basename(archivo)}")
    while True:
        try:
            eleccion = int(input("\nElige un archivo (número, 0 para cancelar): "))
            if eleccion == 0:
                return None
            if 1 <= eleccion <= len(archivos):
                return archivos[eleccion - 1]
            else:
                print("❗ Opción fuera de rango.")
        except ValueError:
            print("❗ Introduce un número válido.")

def cargar_json(ruta_json):
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "messages" not in data or not isinstance(data["messages"], list):
            raise ValueError("El JSON no contiene la clave 'messages' con una lista.")
        return data["messages"]
    except Exception as e:
        print(f"❌ Error al leer el JSON: {e}")
        return []

# Guardar historial en un JSON
def guardar_historial():
    if not messages:
        return
    carpeta_historial = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historial")
    os.makedirs(carpeta_historial, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta_historial, f"historial_{timestamp}_memory_size_{memory_limit}_{model_name}.json")
    try:
        for i, expended_time in enumerate(expended_times):
            messages[i*2+1]["LLM_expended_time"] = expended_time
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"messages": messages}, f, ensure_ascii=False, indent=2)
        print(f"💾 Historial guardado en: {ruta}")
    except Exception as e:
        print(f"❌ Error al guardar historial: {e}")

# Handler para Ctrl+C
def manejar_interrupcion(sig, frame):
    print("\n⚡ Interrumpido por usuario. Guardando historial...")
    guardar_historial()
    sys.exit(0)

signal.signal(signal.SIGINT, manejar_interrupcion)

def esperar_api(timeout=60):
    print("⏳ Esperando a que la API de LM Studio esté disponible...")
    inicio = time.time()
    while time.time() - inicio < timeout:
        try:
            r = requests.get(LMSTUDIO_API_URL)
            if r.status_code == 200:
                print("✅ LM Studio API está activa.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("❌ Timeout: la API de LM Studio no respondió.")
    return False

def obtener_modelo_lanzado_lmstudio():
    try:
        resultado = subprocess.run(
            [LMS_PATH, "ps", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        modelos = json.loads(resultado.stdout)
        if len(modelos) > 0:
            return modelos[0]["identifier"]
    except subprocess.CalledProcessError as e:
        print("❌ Error al obtener la lista de modelos:", e.stderr)
        return None


def chat_local(modelo_seleccionado, mensajes_json=None):
    global messages, expended_times
    client = OpenAI(base_url="http://localhost:3000/v1", api_key="lm-studio")

    main_prompt = """
Eres un robot social diseñado para seguir personas en un ámbito sociosanitario. Tus misiones son las siguientes:
- Analizar datos que vas a recibir desde el punto de vista de un robot social, y asociarlos con datos anteriores con la finalidad de encontrar comportamientos que puedan perturbar la misión. Por ejemplo, si la persona se está alejando del robot.
- Tras el análisis, responder únicamente "0" si del análisis infieres que no hay posibilidad de perturbar el transcurso de la misión, y "1" en el caso de que exista la posibilidad."
"""

    messages = [{
        "role": "system",
        "content": main_prompt
    }]

    expended_times = []
    if mensajes_json:
        print("📄 Ejecutando conversación desde JSON...\n")
        for idx, msg in enumerate(mensajes_json):
            processed_dict = {}
            processed_dict["lateral_distance"] = round(msg["distance"][0], 2)
            processed_dict["front_distance"] = round(msg["distance"][1], 2)
            developer_prompt = """
            - Distancia lateral de la persona respecto al robot: {lateral_distance}.
            - Distancia frontal de la persona respecto al robot: {frontal_distance}.
            """
            print(processed_dict)

            messages.append({"role": "user", "content": developer_prompt.format(
                lateral_distance=processed_dict["lateral_distance"],
                frontal_distance=processed_dict["front_distance"]
            )})
            messages_memory.append({"role": "user", "content": developer_prompt.format(
                lateral_distance=processed_dict["lateral_distance"],
                frontal_distance=processed_dict["front_distance"]
            )})

            if len(messages_memory) == memory_limit:
                messages_memory.popleft()
            try:
                start = time.time()
                if memory_limit > 0:
                    print("Memory size", len(messages_memory))
                    completion = client.chat.completions.create(
                        model=modelo_seleccionado,
                        messages=[{
                            "role": "system",
                            "content": main_prompt
                        }] + list(messages_memory),
                        temperature=0.7,
                    )
                else:
                    completion = client.chat.completions.create(
                        model=modelo_seleccionado,
                        messages=messages,
                        temperature=0.7,
                    )
                expended_milliseconds = time.time() - start
                expended_times.append(expended_milliseconds)
                reply = completion.choices[0].message.content
                print(f"[{idx+1}/{len(mensajes_json)}] 🤖 Respuesta: {reply}\n")
                messages.append({"role": "assistant", "content": reply})
                messages_memory.append({"role": "assistant", "content": reply})
            except Exception as e:
                print("❌ Error al generar respuesta:", e)
                break
        guardar_historial()
    else:
        print("🤖 LLM Chat (escribe 'salir' para terminar)\n")
        while True:
            user_input = input("🧑 Tú: ")
            if user_input.lower() in {"salir", "exit", "quit"}:
                print("👋 Hasta luego.")
                guardar_historial()
                break
            messages.append({"role": "user", "content": user_input})
            try:
                completion = client.chat.completions.create(
                    model=modelo_seleccionado,
                    messages=messages,
                    temperature=0.7,
                )
                reply = completion.choices[0].message.content
                print(f"🤖 Asistente: {reply}\n")
                messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                print("❌ Error al generar respuesta:", e)
                break

if __name__ == "__main__":
    if esperar_api():
        ruta_json = seleccionar_json()
        mensajes_json = cargar_json(ruta_json) if ruta_json else None
        modelo_cargado = obtener_modelo_lanzado_lmstudio()
        model_name = modelo_cargado
        if modelo_cargado is not None:
            try:
                chat_local(modelo_cargado, mensajes_json)
            except KeyboardInterrupt:
                manejar_interrupcion(None, None)
    else:
        print("⚠️ No se pudo verificar la disponibilidad del servidor.")
