@echo off
setlocal enabledelayedexpansion

cd %~dp0..

call pip install -e ".[dev]"
call poe hook

exit /B 0
