@echo off
echo Building ML Pipeline GUI...
pip install pyinstaller

pyinstaller --onefile --windowed --name "MLPipelineGUI" --icon=icon.ico gui.py

echo.
if exist "dist\MLPipelineGUI.exe" (
    echo SUCCESS: MLPipelineGUI.exe built successfully!
) else (
    echo ERROR: Build failed!
)
pause