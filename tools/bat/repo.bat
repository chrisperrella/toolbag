@echo off

if "%1"=="wipe" if "%2"=="" goto WIPE

:RUN_REPO_COMMAND
if "%PM_PACKAGES_ROOT%" == "" call "%~dp0..\packman\packman.cmd" version -q
call "%~dp0.\lib\init_venv.bat"
"%~dp0..\_venv\Scripts\python.exe" "%~dp0..\python\repo.py" %*
set TOOLBAG_ERRORLEVEL=%ERRORLEVEL%
call "%~dp0.\lib\shutdown_venv.bat"
if not "%TOOLBAG_ERRORLEVEL%" == "0" exit /b %TOOLBAG_ERRORLEVEL%
set TOOLBAG_ERRORLEVEL=
exit /b 0

:SUB_REMOVE_FOLDERS
setlocal enabledelayedexpansion
pushd %~dp0..
for /d %%D in ("%CD%\*") do (
    set MY_DIRNAME=%%~nxD
    if "!MY_DIRNAME:~0,1!" == "_" (
        @echo: o Removing !MY_DIRNAME! folder...
        @echo:   - %%D
        rmdir /q/s "%%D"
    )
)
popd
endlocal
exit /b

:WIPE
call :SUB_REMOVE_FOLDERS
set TOOLBAG_ERRORLEVEL=
exit /b 0
