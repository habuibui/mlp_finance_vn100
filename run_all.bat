@echo off
cd /d "%~dp0"
REM Dung python trong PATH. Neu khong co, dat PYTHON=duong_dan_day_du\to\python.exe
if not defined PYTHON set PYTHON=python

echo ========================================================
echo [1/2] Running Sentiment Analysis (PhoBERT)...
echo ========================================================
"%PYTHON%" "%~dp0sentiment_analysis.py"
if errorlevel 1 echo Failed: sentiment_analysis.py

echo.
echo ========================================================
echo [2/2] Running MLP Training and Evaluation...
echo ========================================================
"%PYTHON%" "%~dp0mlp_complete.py"
if errorlevel 1 echo Failed: mlp_complete.py

echo.
echo ========================================================
echo ALL DONE! Check model_results.csv and visualizations folder.
echo ========================================================
pause
