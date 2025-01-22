@echo off
cd /D %~dp0.\..\..\..
setlocal
set FULLPATH=%CD%
for %%a in ("%FULLPATH%") do set "NAME=%%~nxa"
title %NAME% prompt
endlocal
set PATH=%PATH%;%CD%\Tools\bat
:SETUP
if %ERRORLEVEL% equ 0 call "%CD%\Tools\bat\repo.bat" setup
if not %ERRORLEVEL% equ 0 goto ERROR
goto DONE
:ERROR
timeout /t 10
exit /b 1
:DONE
exit /b 0