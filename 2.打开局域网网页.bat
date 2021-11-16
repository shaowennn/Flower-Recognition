@echo off
for /f "tokens=4" %%a in ('route print^|findstr 0.0.0.0.*0.0.0.0') do (
 set IP=%%a 
goto aa
)
:aa
set IP=%IP: =%
echo "http://%IP%:8088/"
@start "" "http://%IP%:8088/"
pause>nul
