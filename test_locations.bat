@echo off
echo Testing multiple locations...
echo.

echo [1/3] Testing India Gate...
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"location\":\"India Gate\",\"standard\":\"IN\"}"
echo.
echo.

echo [2/3] Testing Red Fort...
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"location\":\"Red Fort (Lal Qila)\",\"standard\":\"IN\"}"
echo.
echo.

echo [3/3] Testing Lotus Temple...
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"location\":\"Lotus Temple\",\"standard\":\"IN\"}"
echo.
echo.

echo Done!
pause