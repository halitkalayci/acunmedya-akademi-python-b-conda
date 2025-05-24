@echo off
echo ========================================
echo    Turkiye Ev Fiyat Tahmin API
echo ========================================
echo.
echo API Baslatiliyor...
echo Base URL: http://127.0.0.1:8000
echo Dokümantasyon: http://127.0.0.1:8000/docs
echo.
echo API'yi durdurmak icin Ctrl+C basin
echo.

uvicorn api:app --host 127.0.0.1 --port 8000 --reload

pause 