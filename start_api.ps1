# TradingSystemStack API Startup Script (PowerShell)
# Usage: .\start_api.ps1

Write-Host "🚀 Démarrage de TradingSystemStack API..." -ForegroundColor Green
Write-Host ""
Write-Host "📍 L'API sera accessible sur:" -ForegroundColor Cyan
Write-Host "   - API Base:     http://localhost:8000" -ForegroundColor White
Write-Host "   - Swagger UI:   http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "   - ReDoc:        http://localhost:8000/redoc" -ForegroundColor White
Write-Host "   - Health Check: http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "🔄 Démarrage en mode développement (auto-reload)..." -ForegroundColor Green
Write-Host ""
Write-Host "💡 Pour arrêter: Appuyez sur Ctrl+C" -ForegroundColor Gray
Write-Host ""

# Vérifier si on est dans le bon dossier
if (-Not (Test-Path "src\api\main.py")) {
    Write-Host "❌ Erreur: Fichier src\api\main.py non trouvé!" -ForegroundColor Red
    Write-Host "   Assurez-vous d'être dans le dossier TradingSystemStack" -ForegroundColor Yellow
    pause
    exit 1
}

# Démarrer l'API
try {
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
}
catch {
    Write-Host ""
    Write-Host "❌ Erreur lors du démarrage:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Vérifiez que uvicorn est installé:" -ForegroundColor Yellow
    Write-Host "   pip install uvicorn fastapi" -ForegroundColor White
    pause
}
