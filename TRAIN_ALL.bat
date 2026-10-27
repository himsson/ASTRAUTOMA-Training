@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=py -3"
%PY% -c "" >nul 2>&1 || set "PY=python"
%PY% -u train_all.py %*
if errorlevel 1 pause
