@echo off
setlocal enabledelayedexpansion

cd %~dp0..
if errorlevel 1 exit /B 1

call poe check
if errorlevel 1 exit /B 1

exit /B 0
