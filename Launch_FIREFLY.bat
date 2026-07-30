@echo off
:: FIREFLY - Windows launcher
::
:: On first run (or after a dependency change) this script:
::   1. detects a missing or stale virtual environment,
::   2. installs dependencies with pip output visible so the user can
::      see progress -- installs take 3-8 minutes on first run because
::      PySide6 + PyTorch are large wheels,
::   3. launches FIREFLY when the install finishes.
::
:: A small fingerprint of pyproject.toml + the Python ABI detects dependency
:: changes after `git pull`; `pip check` catches a damaged environment.

setlocal EnableExtensions EnableDelayedExpansion

set "FOLDER=%~dp0"
set "APP=%FOLDER%run_firefly.py"
set "VENV=%FOLDER%sptpalm-env"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "FINGERPRINT_SCRIPT=%FOLDER%scripts\dependency_fingerprint.py"
set "IMPORT_CHECK=%FOLDER%scripts\runtime_import_check.py"
set "STAMP=%VENV%\.firefly-deps.sha256"

:: -- Sanity check: app file present ------------------------------------
if not exist "%APP%" (
    echo.
    echo  FIREFLY files not found at %FOLDER%.
    echo  Re-extract the FIREFLY folder and try again.
    echo.
    pause
    exit /b 1
)

:: -- Locate a system Python (used only to create the venv) -------------
set "PYTHON="
for /f "delims=" %%i in ('where python3 2^>nul') do (
    set "PYTHON=%%i"
    goto :found_python
)
for /f "delims=" %%i in ('where python 2^>nul') do (
    set "PYTHON=%%i"
    goto :found_python
)
echo.
echo  Python 3 is not installed.
echo  Install it from https://www.python.org/downloads/
echo  (tick "Add Python to PATH" during installation)
echo.
pause
exit /b 1
:found_python

:: -- Check whether the venv matches this checkout ----------------------
:: A PySide6 import only proves that one package exists.  It cannot detect a
:: venv left behind by `git pull` without a newly-added runtime dependency.
set "NEEDS_SETUP=0"
if not exist "%VENV_PY%" (
    set "NEEDS_SETUP=1"
) else (
    set "CURRENT_FINGERPRINT="
    for /f "delims=" %%i in ('"%VENV_PY%" "%FINGERPRINT_SCRIPT%"') do set "CURRENT_FINGERPRINT=%%i"
    if not defined CURRENT_FINGERPRINT set "NEEDS_SETUP=1"
    if "!NEEDS_SETUP!"=="0" if not exist "%STAMP%" set "NEEDS_SETUP=1"
    if "!NEEDS_SETUP!"=="0" (
        set "SAVED_FINGERPRINT="
        set /p "SAVED_FINGERPRINT=" < "%STAMP%"
        if not "!CURRENT_FINGERPRINT!"=="!SAVED_FINGERPRINT!" set "NEEDS_SETUP=1"
    )
    if "!NEEDS_SETUP!"=="0" (
        "%VENV_PY%" "%IMPORT_CHECK%" >nul 2>&1
        if errorlevel 1 set "NEEDS_SETUP=1"
    )
    if "!NEEDS_SETUP!"=="0" (
        "%VENV_PY%" -m pip check >nul 2>&1
        if errorlevel 1 set "NEEDS_SETUP=1"
    )
)

if "!NEEDS_SETUP!"=="1" goto :setup
goto :launch

:setup
echo.
echo  ============================================================
echo    FIREFLY setup / dependency update
echo    Installs PySide6, PyTorch, scipy, and the current FIREFLY package.
echo    Expect 3-8 minutes depending on network speed.
echo  ============================================================
echo.

cd /d "%FOLDER%"

if not exist "%VENV_PY%" (
    echo ^> python -m venv sptpalm-env
    "%PYTHON%" -m venv sptpalm-env
    if errorlevel 1 (
        echo.
        echo  Failed to create the virtual environment.
        echo  Check that Python 3 has the venv module (it does by default).
        pause
        exit /b 1
    )
)

echo.
echo ^> upgrading pip
"%VENV_PY%" -m pip install --upgrade pip

echo.
echo ^> installing the current FIREFLY package and dependencies
:: Do not pass --upgrade: a compatible CUDA Torch manually installed by a
:: source user must remain in place.  Pip still fills any missing dependency.
"%VENV_PY%" -m pip install -e .
if errorlevel 1 (
    echo.
    echo  Dependency installation failed.  Re-run this script to retry,
    echo  or run manually:
    echo      %VENV_PY% -m pip install -e .
    pause
    exit /b 1
)

"%VENV_PY%" "%IMPORT_CHECK%"
if errorlevel 1 (
    echo.
    echo  One or more FIREFLY runtime modules still cannot be imported.
    echo  Re-run this script after resolving the errors above.
    pause
    exit /b 1
)

"%VENV_PY%" -m pip check
if errorlevel 1 (
    echo.
    echo  The virtual environment still has incompatible dependencies.
    echo  Re-run this script after resolving the pip errors above.
    pause
    exit /b 1
)

for /f "delims=" %%i in ('"%VENV_PY%" "%FINGERPRINT_SCRIPT%"') do set "CURRENT_FINGERPRINT=%%i"
if not defined CURRENT_FINGERPRINT (
    echo.
    echo  Could not record the dependency fingerprint.
    pause
    exit /b 1
)
> "%STAMP%" echo !CURRENT_FINGERPRINT!

echo.
echo  ============================================================
echo    Setup complete - launching FIREFLY...
echo  ============================================================
echo.

:launch
cd /d "%FOLDER%"
"%VENV_PY%" "%APP%"
