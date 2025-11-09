# 🪟 Guide de Démarrage Windows - TradingSystemStack

Guide spécifique pour les utilisateurs Windows.

---

## ⚡ Démarrage Rapide (Sans Scripts)

**La méthode la plus simple sur Windows:**

```cmd
# 1. Ouvrir Command Prompt ou PowerShell

# 2. Naviguer vers le dossier
cd C:\Users\VotreNom\TradingSystemStack

# 3. Démarrer l'API
uvicorn src.api.main:app --reload

# 4. Ouvrir le navigateur
start http://localhost:8000/docs
```

**C'est tout! Pas de configuration de scripts nécessaire!** ✅

---

## 🔧 Problème: "Scripts sont désactivés sur ce système"

Si vous essayez d'utiliser `start_api.bat` ou `.\start_api.ps1` et obtenez cette erreur, voici les solutions:

### **Solution 1: NE PAS Utiliser de Scripts (Recommandé)**

Utilisez directement la commande:

```cmd
uvicorn src.api.main:app --reload
```

### **Solution 2: Autoriser PowerShell (Cette Session Seulement)**

1. **Ouvrir PowerShell en tant qu'Administrateur**:
   - Rechercher "PowerShell" dans le menu Démarrer
   - Clic droit → "Exécuter en tant qu'administrateur"

2. **Autoriser les scripts** (temporaire):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

3. **Dans le même PowerShell**, naviguer et démarrer:
   ```powershell
   cd C:\Users\VotreNom\TradingSystemStack
   .\start_api.ps1
   ```

### **Solution 3: Autoriser PowerShell (Permanent - Votre Compte)**

⚠️ **Change la sécurité de votre compte utilisateur**

1. **PowerShell en Administrateur**

2. **Commande**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Taper `Y`** pour confirmer

4. **Maintenant** vous pouvez toujours utiliser `.\start_api.ps1`

---

## 📋 Installation Complète sur Windows

### **Étape 1: Prérequis**

```cmd
# Vérifier Python
python --version
# Doit afficher: Python 3.10.x ou plus

# Vérifier pip
pip --version

# Vérifier Git
git --version
```

### **Étape 2: Cloner le Dépôt**

```cmd
# Dans Command Prompt ou PowerShell
cd C:\Users\VotreNom\Documents

# Cloner
git clone https://github.com/SMY335/TradingSystemStack.git

# Entrer dans le dossier
cd TradingSystemStack
```

### **Étape 3: Créer Environnement Virtuel (Recommandé)**

```cmd
# Créer venv
python -m venv venv

# Activer venv
venv\Scripts\activate

# Vous devriez voir (venv) avant votre ligne de commande
```

### **Étape 4: Installer les Dépendances**

```cmd
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer toutes les dépendances
pip install -r requirements.txt
```

**Si vous avez des erreurs**, installez manuellement le minimum:

```cmd
pip install fastapi uvicorn pandas numpy scipy pydantic typer
```

### **Étape 5: Démarrer l'API**

```cmd
# Méthode 1: uvicorn direct
uvicorn src.api.main:app --reload

# Méthode 2: via Python
python -m uvicorn src.api.main:app --reload

# Méthode 3: Script Python
python -m src.api.main
```

### **Étape 6: Tester**

**Dans un navigateur**:
```
http://localhost:8000/docs
```

**Dans un autre Command Prompt**:
```cmd
curl http://localhost:8000/health
```

**OU via PowerShell**:
```powershell
Invoke-WebRequest -Uri http://localhost:8000/health
```

---

## 🎯 Script de Démarrage PowerShell

J'ai créé un script PowerShell qui fonctionne mieux sur Windows:

### **Utilisation**:

1. **Ouvrir PowerShell** (pas forcément en admin)

2. **Naviguer vers le dossier**:
   ```powershell
   cd C:\Users\VotreNom\TradingSystemStack
   ```

3. **Si première fois** (autoriser ce script uniquement):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start_api.ps1
   ```

4. **Si autorisé de façon permanente**:
   ```powershell
   .\start_api.ps1
   ```

---

## 🐛 Problèmes Courants Windows

### **Problème 1: "python n'est pas reconnu"**

**Solution**:
- Réinstaller Python depuis python.org
- ✅ Cocher "Add Python to PATH" pendant l'installation

### **Problème 2: "pip n'est pas reconnu"**

**Solution**:
```cmd
python -m pip --version
# Utiliser "python -m pip" au lieu de "pip"
```

### **Problème 3: "uvicorn n'est pas reconnu"**

**Solution**:
```cmd
pip install uvicorn
# Puis utiliser:
python -m uvicorn src.api.main:app --reload
```

### **Problème 4: Port 8000 déjà utilisé**

**Solution 1 - Trouver et tuer le processus**:
```cmd
# Trouver le processus
netstat -ano | findstr :8000

# Tuer le processus (remplacer <PID> par le numéro)
taskkill /PID <PID> /F
```

**Solution 2 - Utiliser un autre port**:
```cmd
uvicorn src.api.main:app --reload --port 8001
```

### **Problème 5: "Module not found" erreurs**

**Solution**:
```cmd
# S'assurer d'être dans le bon dossier
cd C:\Users\VotreNom\TradingSystemStack

# Vérifier
dir src\api\main.py
# Doit afficher le fichier

# Réinstaller dépendances
pip install -r requirements.txt --force-reinstall
```

---

## 🎨 Accès à l'Interface Web

Une fois l'API démarrée (quelle que soit la méthode):

### **URLs Disponibles**:

| Interface | URL | Raccourci |
|-----------|-----|-----------|
| Swagger UI | http://localhost:8000/docs | `start http://localhost:8000/docs` |
| ReDoc | http://localhost:8000/redoc | `start http://localhost:8000/redoc` |
| API Base | http://localhost:8000 | - |
| Health | http://localhost:8000/health | - |

### **Ouvrir depuis Command Prompt**:
```cmd
start http://localhost:8000/docs
```

### **Ouvrir depuis PowerShell**:
```powershell
Start-Process "http://localhost:8000/docs"
```

---

## 📝 Créer un Raccourci de Bureau (Optionnel)

1. **Créer un fichier** `Start_API.bat` sur votre bureau avec:
   ```batch
   @echo off
   cd C:\Users\VotreNom\TradingSystemStack
   call venv\Scripts\activate
   uvicorn src.api.main:app --reload
   pause
   ```

2. **Double-cliquer** pour démarrer!

---

## 🆘 Aide Supplémentaire

Si vous avez encore des problèmes:

1. **Vérifiez les logs** dans le terminal
2. **Consultez** `INSTALLATION.md` pour plus de détails
3. **Assurez-vous**:
   - ✅ Python 3.10+ installé
   - ✅ Dans le bon dossier (`cd TradingSystemStack`)
   - ✅ Dépendances installées (`pip install -r requirements.txt`)
   - ✅ Fichier `src\api\main.py` existe

---

## ✅ Vérification Finale

**Test que tout fonctionne**:

```cmd
# 1. Vérifier Python
python --version

# 2. Vérifier que vous êtes dans le bon dossier
dir src\api\main.py

# 3. Démarrer l'API
uvicorn src.api.main:app --reload

# 4. Dans un navigateur
start http://localhost:8000/docs

# 5. Vous devriez voir Swagger UI! 🎉
```

---

**Bonne utilisation sur Windows!** 🪟🚀
