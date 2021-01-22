@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i

REM SET OUTPUT_FOLDER=./training/%DTime%_moco
REM SET MODEL=%1
SET FOLDER=%1
SET TARGET_EPOCH=%2
SET "OUTPUT_FOLDER=%FOLDER%"

ECHO %OUTPUT_FOLDER%
python src/main.py ^
--my-config "config_linear.conf" ^
--save_dir %OUTPUT_FOLDER%/linear ^
--load_checkpoint_dir %OUTPUT_FOLDER%/moco_model_%TARGET_EPOCH%.pt
REM --epochs 2 ^
REM --data_input_dir "F:/data/camelyon17/slide_data202003" ^
REM --save_dir %OUTPUT_FOLDER% ^
REM --save_after 1 ^
REM --validate ^
REM --training_data_csv "F:/data/camelyon17/slide_data202003/camelyon17_patches_unbiased.csv" ^
REM --validation_data_csv "F:/data/camelyon17/slide_data202003/camelyon17_patches_unbiased.csv" ^
REM --test_data_csv "F:/data/camelyon17/slide_data202003/camelyon17_patches_unbiased.csv"
