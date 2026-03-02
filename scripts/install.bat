@echo off
setlocal enabledelayedexpansion

cd %~dp0..
if errorlevel 1 exit /B 1

call pip install -e ".[dev]"
if errorlevel 1 exit /B 1

call poe format
if errorlevel 1 exit /B 1

call poe lint
if errorlevel 1 exit /B 1

call poe hook
if errorlevel 1 exit /B 1

exit /B 0
