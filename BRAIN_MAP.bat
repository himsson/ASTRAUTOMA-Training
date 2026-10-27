@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ====================================================================
rem  Kerbal Intelligence Agency - live brain map
rem  Opens a local page that shows what the network is learning.
rem  ASCII only + CRLF line endings: cmd.exe breaks on anything else.
rem ====================================================================

chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

title Kerbal Intelligence Agency - brain map
mode con: cols=90 lines=20 >nul 2>&1
color 0B

cd /d "%~dp0"

echo.
echo   KERBAL INTELLIGENCE AGENCY - BRAIN MAP
echo   The page refreshes itself while training runs.
echo   Closing this window does NOT stop training.
echo.

set "KIA_PY="

rem --- 1. Python launcher (most reliable on Windows) ---
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "KIA_PY=py -3"
    goto :found
)

rem --- 2. python from PATH (skip the Microsoft Store stub) ---
python -c "import sys" >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
        echo %%P | find /i "WindowsApps" >nul
        if errorlevel 1 (
            set "KIA_PY=python"
            goto :found
        )
    )
)

rem --- 3. common install locations ---
if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    set "KIA_PY="%LOCALAPPDATA%\Python\bin\python.exe""
    goto :found
)
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" (
        set "KIA_PY="%%D\python.exe""
        goto :found
    )
)

echo   [ERROR] Python 3.10+ not found.
echo.
pause
exit /b 1

:found
echo   Python: !KIA_PY!
echo.

!KIA_PY! "%~dp0kia_brain_map.py"

if not "%errorlevel%"=="0" (
    echo.
    echo   Brain map exited with code %errorlevel%.
    pause
)
endlocal
