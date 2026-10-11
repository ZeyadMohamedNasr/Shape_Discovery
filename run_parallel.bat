@echo off
echo Launching simulations in parallel...

start cmd /k "python DiscoverJuncBW.py"
start cmd /k "python DiscoverJunction.py"

echo Both scripts are now running in their own command windows!