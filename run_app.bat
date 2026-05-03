@echo off
chcp 65001 >nul
echo =====================================================
echo   Установка зависимостей
echo =====================================================

set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HUGGINGFACE_HUB_DISABLE_SYMLINKS=1

echo [1] Устанавливаем soundfile и scipy (для чтения аудио без torchcodec)...
pip install soundfile scipy pydub -q

echo [2] Запуск приложения...
streamlit run app.py
pause
