@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Force UTF-8 throughout: console, echo >>, and PowerShell file writes
chcp 65001 >nul

set "WORKDIR=D:\engamd89-dev\js-ts\semantic-tool-router"
set "LOG_FILE=%WORKDIR%\test-routes.log"
set "STOP_FILE=%WORKDIR%\test-routes.stop"

cd /d "%WORKDIR%" || exit /b 1

rem Start a fresh log each run
> "%LOG_FILE%" echo test-routes started: %DATE% %TIME%

rem Remove any old shutdown request
if exist "%STOP_FILE%" del "%STOP_FILE%" >nul 2>&1

call :log "Logging to: %LOG_FILE%"
call :log "Graceful shutdown: create this file to stop after the current query:"
call :log "%STOP_FILE%"
call :log ""

for %%q in (
    "List all the files with *.ts"
    "Find every *.json file under src/"
    "Which files match tests/**/*.test.ts"
    "Locate all markdown files in the repo"
    "read package.json"
    "show me the contents of README.md"
    "print the file src/cli.ts"
    "open and display tsconfig.json"
    "write a new hello.txt file"
    "create a fresh .env from the example"
    "overwrite output.log with an empty file"
    "make a new component called Button.tsx"
    "edit the README to change the title"
    "replace all occurrences of foo with bar in src/index.ts"
    "update the version field in package.json"
    "modify the header in the landing page"
    "move src/old.ts to src/new.ts"
    "relocate the assets folder into public/"
    "transfer logs/ to /tmp/archive"
    "rename src/old.ts to src/new.ts"
    "migrate the component to a new name"
    "change the file extension from .js to .ts"
    "list files in src/"
    "show me what is inside the tools directory"
    "enumerate the children of dist/"
    "ls the components folder"
    "scan the src directory"
    "walk the project tree"
    "show the directory structure recursively"
    "traverse the folder and print every path"
    "search for TODO comments in all source files"
    "grep for process.env across the repo"
    "find every place where cosineSimilarity is called"
    "look for the string TODO in markdown files"
    "run npm test"
    "execute the build script"
    "run a shell command to check git status"
    "install dependencies with yarn"
    "fetch the prosodica.ai homepage"
    "download the JSON from https://api.example.com/data"
    "get the latest release notes from GitHub"
    "invoke the code-reviewer skill"
    "use the test-generator skill on src/math"
    "run the documentation skill"
    "tell me a joke"
    "what is the weather like"
    "explain quantum entanglement"
    "the meaning of life"
) do (
    if exist "%STOP_FILE%" goto :graceful_shutdown

    call :log ""
    call :log "..............................................................................."
    call :log "QUERY: %%~q"
    call :log "..............................................................................."

    rem Tee: print each line to console AND append to log as UTF-8 (no BOM)
    call yarn start route "%%~q" 2>&1 | powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$input | ForEach-Object { Write-Host $_; [System.IO.File]::AppendAllText($env:LOG_FILE, $_ + [Environment]::NewLine, [System.Text.Encoding]::UTF8) }"

    call :log ""
    call :log "EXIT CODE: !ERRORLEVEL!"

    if exist "%STOP_FILE%" goto :graceful_shutdown
)

call :log ""
call :log "Completed all routes: %DATE% %TIME%"
exit /b 0

:graceful_shutdown
call :log ""
call :log "Graceful shutdown requested."
call :log "Stopped after current query: %DATE% %TIME%"
if exist "%STOP_FILE%" del "%STOP_FILE%" >nul 2>&1
exit /b 0

:log
echo(%~1
>> "%LOG_FILE%" echo(%~1
exit /b 0
