; ArSL Translation Overlay — NSIS Installer
; Requires: NSIS 3.0+, pyinstaller-built dist/

!define PRODUCT_NAME "ArSL Translation Overlay"
!define PRODUCT_VERSION "1.0.0"
!define PRODUCT_PUBLISHER "ArSL Project"
!define PRODUCT_WEB_SITE "https://github.com/arsl-project/arsl-overlay"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\ArSL Overlay.exe"

SetCompressor lzma
RequestExecutionLevel admin

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "ArSL-Overlay-Setup-${PRODUCT_VERSION}.exe"
InstallDir "$PROGRAMFILES64\ArSL Overlay"
InstallDirRegKey HKLM "${PRODUCT_DIR_REGKEY}" ""

Section "Install"
    SetOutPath "$INSTDIR"

    File /r "dist\ArSL Overlay\*.*"

    CreateDirectory "$APPDATA\ArSL Overlay\data\models"
    CreateDirectory "$APPDATA\ArSL Overlay\data\datasets"
    CreateDirectory "$APPDATA\ArSL Overlay\data\logs"
    CreateDirectory "$APPDATA\ArSL Overlay\config"

    CreateShortCut "$DESKTOP\ArSL Overlay.lnk" "$INSTDIR\ArSL Overlay.exe"
    CreateDirectory "$SMPROGRAMS\ArSL Overlay"
    CreateShortCut "$SMPROGRAMS\ArSL Overlay\ArSL Overlay.lnk" "$INSTDIR\ArSL Overlay.exe"
    CreateShortCut "$SMPROGRAMS\ArSL Overlay\Uninstall.lnk" "$INSTDIR\uninst.exe"

    WriteUninstaller "$INSTDIR\uninst.exe"
    WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\ArSL Overlay.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "UninstallString" "$INSTDIR\uninst.exe"
SectionEnd

Section "Uninstall"
    Delete "$DESKTOP\ArSL Overlay.lnk"
    RMDir /r "$SMPROGRAMS\ArSL Overlay"
    RMDir /r "$INSTDIR"
    DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
SectionEnd
