@echo off
rem Studiosus - double-click to stand the local stack up and open Hermes.
rem The window stays open so the flame report and any errors are readable.
title Studiosus
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\studiosus_launcher.ps1" %*
if errorlevel 1 pause
