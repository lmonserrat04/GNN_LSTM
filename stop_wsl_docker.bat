@echo off
echo [1/3] Deteniendo contenedor gnn_search_task...
wsl -u ia_admin docker stop gnn

echo [2/3] Cerrando Docker Desktop...
taskkill /IM "com.docker.service.exe" /F >nul 2>&1
timeout /t 3 /nobreak >nul

echo [3/3] Apagando WSL...
wsl --shutdown

echo Listo. Docker y WSL detenidos.
