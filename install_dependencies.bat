@echo off
echo ============================================================
echo Installing Dependencies for Prescription Recognition
echo ============================================================
echo.

echo Installing OpenCV...
pip install opencv-python
echo.

echo Installing Pandas...
pip install pandas
echo.

echo Installing tqdm (progress bars)...
pip install tqdm
echo.

echo Installing editdistance (for CER calculation)...
pip install editdistance
echo.

echo Installing matplotlib (for visualization)...
pip install matplotlib
echo.

echo Installing scikit-learn (for metrics)...
pip install scikit-learn
echo.

echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Now run: python quickstart.py
echo.
pause