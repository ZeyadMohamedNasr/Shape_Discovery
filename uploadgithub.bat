@echo off
echo Staging files...
git add -A

set /p msg="Enter your commit message: "
if "%msg%"=="" set msg="Update scripts and project files"

echo Committing and Pushing to GitHub...
git commit -m "%msg%"
git push origin main

echo Done!