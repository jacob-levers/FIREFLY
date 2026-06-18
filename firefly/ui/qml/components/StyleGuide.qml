import QtQuick
import QtQuick.Layouts

// Phase-1 component preview — shows the QML library that the tabs will build on.
// Temporary: replaced by the real tab content in Phase 2+.
Flickable {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    contentWidth: width
    contentHeight: col.implicitHeight + 56
    clip: true

    ColumnLayout {
        id: col
        x: 28; y: 18
        width: Math.min(760, root.width - 56)
        spacing: sc.sp10

        Text { text: "Component library"; color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true }
        Text {
            text: "Phase 1 preview · these primitives compose the upcoming tabs."
            color: pal.TXT_MUTED; font.pixelSize: sc.textSm
        }

        // Buttons
        ColumnLayout {
            spacing: sc.sp4
            Text { text: "BUTTONS"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
            RowLayout {
                spacing: sc.sp4
                Button { variant: "primary"; text: "Start"; icon: "play" }
                Button { variant: "secondary"; text: "Browse"; icon: "folder-open" }
                Button { variant: "danger"; text: "Stop" }
                Button { variant: "secondary"; text: "Disabled"; enabled: false }
                IconButton { icon: "settings"; tip: "Settings" }
                IconButton { icon: "x"; danger: true; tip: "Close" }
            }
        }

        // Badges
        ColumnLayout {
            spacing: sc.sp4
            Text { text: "BADGES & STATUS"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
            RowLayout {
                spacing: sc.sp4
                Badge { text: "ready"; tone: pal.SUCCESS; dot: true }
                Badge { text: "blocked"; tone: pal.DANGER; dot: true }
                Badge { text: "auto-threshold"; tone: pal.WARN }
                Badge { text: ".czi · 16,384 frames"; tone: pal.ACC }
            }
        }

        // Inputs
        ColumnLayout {
            spacing: sc.sp4
            Text { text: "INPUTS"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
            RowLayout {
                spacing: sc.sp6
                FieldInput { Layout.preferredWidth: 220; placeholderText: "minmass…" }
                RowLayout {
                    spacing: sc.sp3
                    Switch { checked: true }
                    Text { text: "GPU localiser"; color: pal.TXT; font.pixelSize: sc.textSm }
                }
            }
        }

        // Alerts
        ColumnLayout {
            spacing: sc.sp4
            Layout.fillWidth: true
            Text { text: "FEEDBACK"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
            Alert { Layout.fillWidth: true; severity: "success"; text: "Run finished — 18,743 tracks, median D = 0.21 µm²/s." }
            Alert { Layout.fillWidth: true; severity: "warn"; text: "Couldn't read the metadata. Tick Override and enter the value from your acquisition." }
            Alert { Layout.fillWidth: true; severity: "danger"; text: "No localisations were produced. Lower minmass and try again." }
        }

        // Accordion (sidebar section)
        ColumnLayout {
            spacing: sc.sp4
            Layout.fillWidth: true
            Text { text: "SIDEBAR ACCORDION"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: section.implicitHeight + sc.sp6 * 2
                CollapsibleSection {
                    id: section
                    x: sc.sp6; y: sc.sp6
                    width: parent.width - sc.sp6 * 2
                    title: "Detection"; icon: "scan-search"; expanded: true
                    RowLayout {
                        width: parent.width
                        Text { text: "Diameter"; color: pal.TXT_MUTED; font.pixelSize: sc.textSm }
                        Item { Layout.fillWidth: true }
                        FieldInput { Layout.preferredWidth: 90; text: "7 px" }
                    }
                    RowLayout {
                        width: parent.width
                        Text { text: "Use MPS"; color: pal.TXT_MUTED; font.pixelSize: sc.textSm }
                        Item { Layout.fillWidth: true }
                        Switch { checked: true }
                    }
                }
            }
        }
    }
}
