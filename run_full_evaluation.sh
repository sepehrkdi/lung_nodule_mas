#!/bin/bash
mkdir -p results

# Batches 1 to 29 (covering 0 to ~2900)
echo "Starting batch 1 (0-100)..."
python main_extended.py --evaluate --data nlmcxr --start-index 0 --max-cases 100 --export results/results_0-100.json

echo "Starting batch 2 (100-200)..."
python main_extended.py --evaluate --data nlmcxr --start-index 100 --max-cases 100 --export results/results_100-200.json

echo "Starting batch 3 (200-300)..."
python main_extended.py --evaluate --data nlmcxr --start-index 200 --max-cases 100 --export results/results_200-300.json

echo "Starting batch 4 (300-400)..."
python main_extended.py --evaluate --data nlmcxr --start-index 300 --max-cases 100 --export results/results_300-400.json

echo "Starting batch 5 (400-500)..."
python main_extended.py --evaluate --data nlmcxr --start-index 400 --max-cases 100 --export results/results_400-500.json

echo "Starting batch 6 (500-600)..."
python main_extended.py --evaluate --data nlmcxr --start-index 500 --max-cases 100 --export results/results_500-600.json

echo "Starting batch 7 (600-700)..."
python main_extended.py --evaluate --data nlmcxr --start-index 600 --max-cases 100 --export results/results_600-700.json

echo "Starting batch 8 (700-800)..."
python main_extended.py --evaluate --data nlmcxr --start-index 700 --max-cases 100 --export results/results_700-800.json

echo "Starting batch 9 (800-900)..."
python main_extended.py --evaluate --data nlmcxr --start-index 800 --max-cases 100 --export results/results_800-900.json

echo "Starting batch 10 (900-1000)..."
python main_extended.py --evaluate --data nlmcxr --start-index 900 --max-cases 100 --export results/results_900-1000.json

echo "Starting batch 11 (1000-1100)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1000 --max-cases 100 --export results/results_1000-1100.json

echo "Starting batch 12 (1100-1200)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1100 --max-cases 100 --export results/results_1100-1200.json

echo "Starting batch 13 (1200-1300)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1200 --max-cases 100 --export results/results_1200-1300.json

echo "Starting batch 14 (1300-1400)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1300 --max-cases 100 --export results/results_1300-1400.json

echo "Starting batch 15 (1400-1500)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1400 --max-cases 100 --export results/results_1400-1500.json

echo "Starting batch 16 (1500-1600)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1500 --max-cases 100 --export results/results_1500-1600.json

echo "Starting batch 17 (1600-1700)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1600 --max-cases 100 --export results/results_1600-1700.json

echo "Starting batch 18 (1700-1800)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1700 --max-cases 100 --export results/results_1700-1800.json

echo "Starting batch 19 (1800-1900)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1800 --max-cases 100 --export results/results_1800-1900.json

echo "Starting batch 20 (1900-2000)..."
python main_extended.py --evaluate --data nlmcxr --start-index 1900 --max-cases 100 --export results/results_1900-2000.json

echo "Starting batch 21 (2000-2100)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2000 --max-cases 100 --export results/results_2000-2100.json

echo "Starting batch 22 (2100-2200)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2100 --max-cases 100 --export results/results_2100-2200.json

echo "Starting batch 23 (2200-2300)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2200 --max-cases 100 --export results/results_2200-2300.json

echo "Starting batch 24 (2300-2400)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2300 --max-cases 100 --export results/results_2300-2400.json

echo "Starting batch 25 (2400-2500)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2400 --max-cases 100 --export results/results_2400-2500.json

echo "Starting batch 26 (2500-2600)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2500 --max-cases 100 --export results/results_2500-2600.json

echo "Starting batch 27 (2600-2700)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2600 --max-cases 100 --export results/results_2600-2700.json

echo "Starting batch 28 (2700-2800)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2700 --max-cases 100 --export results/results_2700-2800.json

echo "Starting batch 29 (2800-2900)..."
python main_extended.py --evaluate --data nlmcxr --start-index 2800 --max-cases 100 --export results/results_2800-2900.json

echo "All batches complete!"
