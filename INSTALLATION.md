# 📥 Guide d'Installation - TradingSystemStack

Guide complet pour installer et démarrer TradingSystemStack **depuis GitHub**.

---

## 📋 Table des Matières

1. [Prérequis](#prérequis)
2. [Installation depuis GitHub](#installation-depuis-github)
3. [Configuration](#configuration)
4. [Démarrage](#démarrage)
5. [Vérification](#vérification)
6. [Troubleshooting](#troubleshooting)

---

## 1. Prérequis

### Logiciels Requis

- **Python 3.10 ou supérieur**
  ```bash
  python --version
  # Doit afficher: Python 3.10.x ou plus
  ```

- **Git**
  ```bash
  git --version
  ```

- **pip** (gestionnaire de packages Python)
  ```bash
  pip --version
  ```

### Optionnel mais Recommandé

- **TA-Lib** (pour certains indicateurs techniques)

  **Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install ta-lib
  ```

  **macOS:**
  ```bash
  brew install ta-lib
  ```

  **Windows:**
  - Télécharger depuis: http://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
  - Installer le fichier .whl correspondant à votre version Python

---

## 2. Installation depuis GitHub

### Étape 1: Cloner le Dépôt

```bash
# Cloner le dépôt
git clone https://github.com/SMY335/TradingSystemStack.git

# Entrer dans le dossier
cd TradingSystemStack
```

### Étape 2: Créer un Environnement Virtuel (Recommandé)

**Linux/macOS:**
```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate
```

**Windows:**
```cmd
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
venv\Scripts\activate
```

Vous devriez voir `(venv)` au début de votre ligne de commande.

### Étape 3: Installer les Dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer toutes les dépendances
pip install -r requirements.txt
```

**Installation Minimale (si requirements.txt pose problème):**
```bash
pip install fastapi uvicorn pandas numpy scipy pydantic typer plotly yfinance
```

### Étape 4: Installer TA-Lib Python (Optionnel)

**Si vous avez installé TA-Lib système à l'étape des prérequis:**
```bash
pip install TA-Lib
```

**Sinon, l'application fonctionnera sans (avec fonctionnalités limitées)**

---

## 3. Configuration

### Créer le Fichier de Configuration

```bash
# Copier l'exemple de configuration
cp .env.example .env

# Éditer selon vos besoins (optionnel)
nano .env  # ou vim, code, etc.
```

### Configuration par Défaut

Le fichier `.env.example` contient des valeurs par défaut fonctionnelles. Vous n'avez **pas besoin** de le modifier pour démarrer.

**Optionnel** - Si vous voulez utiliser des APIs externes:
- **Alpha Vantage API Key**: Pour données financières avancées
- **FRED API Key**: Pour données économiques réelles

---

## 4. Démarrage

### Méthode 1: Script de Démarrage (Le Plus Simple)

**Linux/macOS:**
```bash
./start_api.sh
```

**Windows:**
```cmd
start_api.bat
```

### Méthode 2: Commande Directe

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Méthode 3: Python Direct

```bash
python -m src.api.main
```

### Ce que vous devriez voir:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using StatReload
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 5. Vérification

### Test 1: Health Check (Terminal)

Ouvrez un **nouveau terminal** et testez:

```bash
curl http://localhost:8000/health
```

**Résultat attendu:**
```json
{"status":"healthy","version":"2.0.0"}
```

### Test 2: Swagger UI (Navigateur)

Ouvrez votre navigateur et allez sur:
```
http://localhost:8000/docs
```

**Vous devriez voir** l'interface Swagger UI avec tous les endpoints API! 🎉

### Test 3: CLI

Dans un terminal:
```bash
python -m src.cli --help
```

**Vous devriez voir** la liste des commandes disponibles.

---

## 6. Troubleshooting

### Problème: "Module not found"

**Solution 1 - Vérifier l'environnement virtuel:**
```bash
# Assurez-vous que (venv) est actif
which python  # Linux/macOS
where python  # Windows

# Devrait pointer vers venv/bin/python ou venv\Scripts\python
```

**Solution 2 - Réinstaller les dépendances:**
```bash
pip install -r requirements.txt --force-reinstall
```

**Solution 3 - Ajouter au PYTHONPATH:**
```bash
# Linux/macOS
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
```

---

### Problème: "Port 8000 already in use"

**Solution 1 - Utiliser un autre port:**
```bash
uvicorn src.api.main:app --reload --port 8001
```

**Solution 2 - Trouver et tuer le processus:**
```bash
# Linux/macOS
lsof -i :8000
kill -9 <PID>

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

### Problème: "TA-Lib not found"

**C'est normal!** TA-Lib est optionnel.

**Option 1 - Installer TA-Lib:**
Voir la section [Prérequis](#optionnel-mais-recommandé)

**Option 2 - Continuer sans:**
L'application fonctionne sans TA-Lib, certains indicateurs seront simplement indisponibles.

---

### Problème: Erreurs lors de `pip install -r requirements.txt`

**Solution - Installation manuelle minimale:**
```bash
pip install fastapi uvicorn pandas numpy scipy pydantic typer
```

**Puis démarrer et installer les packages manquants au besoin.**

---

### Problème: "Import Error" au démarrage

**Vérifier la structure du projet:**
```bash
ls src/
# Devrait afficher: api, data, utils, indicators, patterns, etc.
```

**Vérifier que vous êtes à la racine:**
```bash
pwd  # Linux/macOS
cd   # Windows

# Devrait afficher: .../TradingSystemStack
```

---

## ✅ Installation Réussie!

Si vous voyez:
- ✅ Swagger UI sur http://localhost:8000/docs
- ✅ `/health` retourne `{"status":"healthy"}`
- ✅ CLI fonctionne avec `python -m src.cli --help`

**Félicitations! L'installation est complète!** 🎉

---

## 🚀 Étapes Suivantes

Maintenant que tout fonctionne:

1. **Explorer Swagger UI**: http://localhost:8000/docs
2. **Lire le Quick Start**: `QUICK_START.md`
3. **Tester un endpoint**: Calculer un RSI dans Swagger
4. **Lire la documentation**: `DEPLOYMENT_GUIDE.md`

---

## 📚 Ressources

- **Quick Start**: `QUICK_START.md` - Démarrage rapide
- **Deployment**: `DEPLOYMENT_GUIDE.md` - Guide de déploiement complet
- **Scanner DSL**: `docs/SCANNER_DSL.md` - Documentation du scanner
- **API Docs**: http://localhost:8000/docs (après démarrage)

---

## 🆘 Besoin d'Aide?

Si vous rencontrez des problèmes:

1. Vérifiez les logs du terminal où uvicorn tourne
2. Consultez la section [Troubleshooting](#troubleshooting)
3. Ouvrez une issue sur GitHub avec:
   - Votre OS et version Python
   - La commande exacte utilisée
   - Le message d'erreur complet

---

**Bonne utilisation de TradingSystemStack!** 📊🚀
