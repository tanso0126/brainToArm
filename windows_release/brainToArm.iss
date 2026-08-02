#ifndef AppVersion
  #define AppVersion "2.0.0"
#endif

[Setup]
AppId={{6F196D1A-8B20-43EF-B36A-76F2C3D79AE8}
AppName=brainToArm 통합 운영실
AppVersion={#AppVersion}
AppPublisher=brainToArm Team
DefaultDirName={localappdata}\Programs\brainToArm
DefaultGroupName=brainToArm
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=brainToArm-Windows-Setup-v{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\brainToArm.exe
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Files]
Source: "..\dist\brainToArm\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README_WINDOWS.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "START_HERE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\brainToArm 통합 운영실"; Filename: "{app}\brainToArm.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\brainToArm 통합 운영실"; Filename: "{app}\brainToArm.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 brainToArm 통합 운영실 바로가기 만들기"; GroupDescription: "바로가기:"

[Run]
Filename: "{app}\brainToArm.exe"; Description: "brainToArm 통합 운영실 실행"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
