@echo off

echo ========================================================
echo [1/2] Running Sentiment Analysis (PhoBERT)...
echo ========================================================
"C:\Users\HA BUI\AppData\Local\Programs\Python\Python310\python.exe" "D:\Document\STUDY\HK11\K224141657\sentiment_analysis.py"

echo.
echo ========================================================
echo [2/2] Running MLP Training & Evaluation...
echo ========================================================
"C:\Users\HA BUI\AppData\Local\Programs\Python\Python310\python.exe" "D:\Document\STUDY\HK11\K224141657\mlp_complete.py"

echo.
echo ========================================================
echo ALL DONE! Check 'model_results.csv' and 'visualizations' folder.
echo ========================================================
pause
