@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i

SET OUTPUT_FOLDER=./training/%DTime%_moco
REM SET MODEL=%1
REM SET FOLDER=%1
REM SET "OUTPUT_FOLDER=E:\OneDrive - Sectra\Research\2019\representation_shift\results\november2019\%FOLDER%"

ECHO %OUTPUT_FOLDER%
python src/main.py ^
--my-config "config_train.conf" ^
--save_dir %OUTPUT_FOLDER%
