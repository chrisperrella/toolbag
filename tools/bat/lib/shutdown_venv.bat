@echo off

if not "%TOOLBAG_PYTHONPATH_BACKUP%" == "" set PYTHONPATH=%TOOLBAG_PYTHONPATH_BACKUP%
set TOOLBAG_PYTHONPATH_BACKUP=
call "%~dp0..\..\_venv\Scripts\deactivate.bat"
