@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ====================================================================
rem  Kerbal Intelligence Agency - 3D brain map
rem  Opens a local 3D page: skills outside, real neurons inside.
rem  ASCII only + CRLF line endings: cmd.exe breaks on anything else.
rem ====================================================================

chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

title Kerbal Intelligence Agency - brain map 3D
mode con: cols=90 lines=20 >nul 2>&1
color 0D

cd /d "%~dp0"

echo.
echo   KERBAL INTELLIGENCE AGENCY - BRAIN MAP 3D
echo   Mouse wheel flies you into the core. Hover a node to read it.
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

!KIA_PY! "%~dp0kia_brain3d.py"

if not "%errorlevel%"=="0" (
    echo.
    echo   Brain map 3D exited with code %errorlevel%.
    pause
)
endlocal
