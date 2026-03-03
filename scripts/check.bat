@echo off
setlocal enabledelayedexpansion

cd %~dp0..
call poe check

pause
exit /B 0
