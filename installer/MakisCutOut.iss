; Inno Setup 7 script. Build with scripts\build_installer.ps1: it fetches uv and renders the icon into build\ first.
; Setup copies the source and uv. setup_env.cmd then downloads Python and PyTorch (~4 GB) into the install folder.

#ifndef AppVersion
  #error AppVersion is passed by scripts\build_installer.ps1 from app\__init__.py
#endif
#define AppName "Maki's CutOut"
#define AppURL "https://github.com/MakiNoAtorie/Maki-s-CutOut"
#define Launcher "{app}\.venv\gui\pythonw.exe"

[Setup]
; Never change AppId: upgrades and uninstall find the installed copy by it.
AppId={{BC9A560A-F372-4471-AC44-4C7D0ED64111}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} v{#AppVersion}
AppPublisher=MakiNoAtorie
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
AppMutex=MakisCutOut
VersionInfoVersion={#AppNumVersion}
; models\ is written next to the code, so the install folder must be writable without admin rights.
PrivilegesRequired=lowest
; uv links Python through a junction, which RedirectionGuard blocks (error 448) in setup_env.cmd. Setup never elevates.
RedirectionGuard=no
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; Python environment (~4 GB) and AI models (~1.7 GB) are downloaded after the file copy.
ExtraDiskSpaceRequired=6000000000
LicenseFile=..\LICENSE
SetupIconFile=build\CutOut.ico
UninstallDisplayIcon={app}\CutOut.ico
UninstallDisplayName={#AppName}
WizardStyle=modern dynamic
Compression=lzma2/max
SolidCompression=yes
OutputDir=..\dist
OutputBaseFilename=MakisCutOut-v{#AppVersion}-Setup

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Remove source files that a newer version no longer ships.
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\engine"

[Files]
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\pyproject.toml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\uv.lock"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\.python-version"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\app\*.py"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "..\app\icon.svg"; DestDir: "{app}\app"; Flags: ignoreversion
Source: "..\engine\*.py"; DestDir: "{app}\engine"; Flags: ignoreversion
Source: "setup_env.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "build\uv.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "build\CutOut.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{#Launcher}"; Parameters: """{app}\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\CutOut.ico"; Comment: "Cut subjects out of images"; AppUserModelID: "MakiNoAtorie.MakisCutOut"
Name: "{autodesktop}\{#AppName}"; Filename: "{#Launcher}"; Parameters: """{app}\main.py"""; WorkingDir: "{app}"; IconFilename: "{app}\CutOut.ico"; Comment: "Cut subjects out of images"; AppUserModelID: "MakiNoAtorie.MakisCutOut"; Tasks: desktopicon

[Run]
Filename: "{#Launcher}"; Parameters: """{app}\main.py"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent; Check: EnvReady

[UninstallDelete]
; Created after install by setup_env.cmd and the app. Named one by one so a custom install folder is never wiped whole.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\app\__pycache__"
Type: filesandordirs; Name: "{app}\engine\__pycache__"
Type: files; Name: "{app}\setup_env.log"
Type: filesandordirs; Name: "{localappdata}\CutOut"

[Code]
var
  EnvOk: Boolean;

function EnvReady: Boolean;
begin
  Result := EnvOk;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code, Show: Integer;
  Args: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  WizardForm.StatusLabel.Caption := 'Downloading Python and PyTorch (about 3 GB). Progress shows in the console window...';
  Show := SW_SHOWNORMAL;
  Args := '';
  if WizardSilent then
  begin
    Show := SW_HIDE;
    Args := ' nopause > "' + ExpandConstant('{app}\setup_env.log') + '" 2>&1';
  end;
  EnvOk := Exec(ExpandConstant('{cmd}'), '/c ""' + ExpandConstant('{app}\setup_env.cmd') + '"' + Args + '"',
    ExpandConstant('{app}'), Show, ewWaitUntilTerminated, Code) and (Code = 0);
  if not EnvOk then
    SuppressibleMsgBox('The Python environment could not be downloaded. Check your internet connection and run Setup again.',
      mbError, MB_OK, IDOK);
end;
