@echo off

if not exist "%~dp0..\..\_venv" goto :SETUP
if not exist "%~dp0..\..\_venv\Scripts\activate.bat" goto :SETUP
if not exist "%~dp0..\..\_venv\Scripts\deactivate.bat" goto :SETUP
goto :SETUP_COMPLETE
:SETUP
@echo:
@echo: o Setting up Python Virtual Environment...
@echo:
if exist "%~dp0..\..\_venv" rmdir /q/s "%~dp0..\..\_venv"
mkdir "%~dp0..\..\_venv"
call python -m venv "%~dp0..\..\_venv"
:SETUP_COMPLETE
call "%~dp0..\..\_venv\Scripts\activate.bat"
if "%PYTHONPATH_BACKUP%" == "" set PYTHONPATH_BACKUP=%PYTHONPATH%
set ERRORLEVEL=
set PYTHONPATH=%PM_MODULE_DIR%;%~dp0..\..\python\lib;
exit /b
