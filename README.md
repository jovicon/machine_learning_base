
# 🐍 Python + Jupyter Notebooks en VS Code

Setup reproducible para análisis de datos con entorno virtual aislado.

---

## 📋 Requisitos previos

- [Python 3.11+](https://www.python.org/downloads/)
- [Visual Studio Code](https://code.visualstudio.com/)
- Extensiones de VS Code:
  - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (`ms-python.python`)
  - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) (`ms-toolsai.jupyter`)

---

## 🚀 Setup inicial (solo primera vez)

### 1. Clonar el repositorio

```bash
git clonehttps://github.com/jovicon/machine_learning_base
cd machine_learning_base
```

### 2. Ejecutar setup automático

```bash
make setup
```

Este comando:

- Crea el entorno virtual `.venv`
- Instala las dependencias desde `requirements.txt`
- Registra el kernel de Jupyter en VS Code

### 3. Activar el entorno virtual

```bash
source .venv/bin/activate
```

> **Windows:**
>
> ```bash
> .venv\Scripts\activate
> ```

---

## 📓 Abrir un notebook en VS Code

1. Abre VS Code en la carpeta del proyecto
2. Abre o crea un archivo `.ipynb` desde `notebooks/`
3. En la esquina superior derecha, clic en **"Select Kernel"**
4. Selecciona **"Python (nombre-del-proyecto)"**

> Si el kernel no aparece, ejecuta `make setup` nuevamente y recarga VS Code (`Ctrl+Shift+P` → `Developer: Reload Window`)

---

## 🛠️ Comandos disponibles

```bash
make setup      # Crea venv, instala dependencias y registra kernel
make clean      # Elimina el entorno virtual
```

---

## 📁 Estructura del proyecto

```
.
├── .venv/                  # Entorno virtual (ignorado en git)
├── .env                    # Variables de entorno - credenciales (ignorado en git)
├── .gitignore
├── Makefile
├── requirements.txt        # Dependencias del proyecto
├── notebooks/
│   └── exploracion.ipynb   # Notebooks de análisis
└── README.md
```

---

## ➕ Agregar nuevas dependencias

```bash
# Activar el venv primero
source .venv/bin/activate

# Instalar la nueva librería
pip install nombre-libreria

# Actualizar requirements.txt
pip freeze > requirements.txt
```

Commitea el `requirements.txt` actualizado para que el equipo tenga las mismas versiones.

---

## 🔑 Variables de entorno

Crea un archivo `.env` en la raíz del proyecto (nunca lo commitees):

```env
MONGO_URI=mongodb+srv://...
DB_NAME=nombre_db
```

Carga las variables en tu notebook:

```python
from dotenv import load_dotenv
import os

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
```

---

## ❓ Troubleshooting

**`jupyter: command not found`**

```bash
source .venv/bin/activate
jupyter --version
```

**El kernel no aparece en VS Code**

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=nombre-proyecto --display-name "Python (nombre-proyecto)"
```

Luego recarga VS Code: `Ctrl+Shift+P` → `Developer: Reload Window`

**Conflicto con Python del sistema**

Siempre trabaja con el venv activado. Verifica con:

```bash
which python   # debe apuntar a .venv/bin/python
```
