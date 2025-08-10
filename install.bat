@echo off
echo 🚀 Market & Social Insights Dashboard - Installation
echo ===================================================
echo.

echo 📦 Installing dependencies...
python install_dependencies.py

echo.
echo 🎉 Installation complete!
echo.
echo 📋 Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: streamlit run app.py
echo 3. Or run demo: streamlit run demo.py
echo 4. Open http://localhost:8501 in your browser
echo.
pause 