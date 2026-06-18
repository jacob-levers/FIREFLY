import QtQuick
import QtQuick.Layouts
import "../components"

// Import tab: pick an input file (typed chip + format/frame badges), choose an
// output folder, and set calibration (pixel size / frame interval, each with a
// metadata-override). Bound to ImportController; running lands in Phase 3.
Flickable {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    contentWidth: width
    contentHeight: col.implicitHeight + 56
    clip: true

    ColumnLayout {
        id: col
        x: 28; y: 20
        width: Math.min(820, root.width - 56)
        spacing: sc.sp6

        Text { text: "Input"; color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true }

        // ── file chip ────────────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 68
            RowLayout {
                anchors.fill: parent
                anchors.margins: sc.sp5
                spacing: sc.sp5
                Rectangle {
                    width: 40; height: 40; radius: sc.radiusLg
                    color: Qt.rgba(0.345, 0.651, 1.0, Import.hasFile ? 0.12 : 0.05)
                    border.width: 1
                    border.color: Qt.rgba(0.345, 0.651, 1.0, 0.22)
                    Icon {
                        anchors.centerIn: parent
                        name: Import.isCsv ? "circle-dot" : "microscope"
                        color: Import.hasFile ? pal.ACC : pal.TXT_MUTED
                        size: 20
                    }
                }
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: sc.sp1
                    Text {
                        text: Import.hasFile ? Import.fileName : "No file selected"
                        color: Import.hasFile ? pal.TXT : pal.TXT_MUTED
                        font.pixelSize: sc.textSm
                        font.family: Import.hasFile ? "Menlo" : Qt.application.font.family
                        elide: Text.ElideMiddle
                        Layout.fillWidth: true
                    }
                    RowLayout {
                        visible: Import.hasFile
                        spacing: sc.sp3
                        Badge { text: Import.fileFormat; tone: pal.ACC }
                        Badge {
                            visible: Import.frameCount > 0
                            text: Import.frameCount.toLocaleString(Qt.locale(), "f", 0) + " frames"
                            tone: pal.TXT_MUTED
                        }
                    }
                }
                Button {
                    variant: "secondary"; text: "Browse…"; icon: "folder-open"
                    onClicked: Import.browseFile()
                }
            }
        }

        // ── output folder ────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            spacing: sc.sp4
            Text { text: "Output"; color: pal.TXT_MUTED; font.pixelSize: sc.textSm; Layout.preferredWidth: 60 }
            Text {
                text: Import.outDir || "(beside the input file)"
                color: Import.outDir ? pal.TXT : pal.TXT_MUTED
                font.pixelSize: sc.textSm; font.family: "Menlo"
                elide: Text.ElideMiddle; Layout.fillWidth: true
            }
            Button { variant: "secondary"; text: "Choose…"; icon: "folder-open"; onClicked: Import.browseOutDir() }
        }

        // ── calibration ──────────────────────────────────────────────
        Text {
            text: "Calibration"; color: pal.TXT
            font.pixelSize: sc.textLg; font.bold: true; topPadding: sc.sp4
        }
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: cal.implicitHeight + sc.sp6 * 2
            ColumnLayout {
                id: cal
                x: sc.sp6; y: sc.sp6
                width: parent.width - sc.sp6 * 2
                spacing: sc.sp6

                component CalRow: RowLayout {
                    property string label
                    property bool overridden
                    property real value
                    property string unit
                    signal toggled(bool on)
                    signal committed(real v)
                    width: cal.width
                    spacing: sc.sp4
                    Text { text: label; color: pal.TXT; font.pixelSize: sc.textSm; Layout.preferredWidth: 150 }
                    Switch { checked: overridden; onToggled: (c) => parent.toggled(c) }
                    Text { text: "Override"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                    Item { Layout.fillWidth: true }
                    Text {
                        visible: !overridden
                        text: "read from file metadata"
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                    }
                    FieldInput {
                        Layout.preferredWidth: 120
                        enabled: overridden
                        opacity: enabled ? 1.0 : 0.45
                        horizontalAlignment: TextInput.AlignRight
                        text: value.toFixed(3)
                        onEditingFinished: {
                            var v = parseFloat(text)
                            if (!isNaN(v)) parent.committed(v)
                        }
                    }
                }
                CalRow {
                    label: "Pixel size (µm)"; overridden: Import.overridePx; value: Import.pixelSize
                    onToggled: (on) => Import.overridePx = on
                    onCommitted: (v) => Import.pixelSize = v
                }
                CalRow {
                    label: "Frame interval (s)"; overridden: Import.overrideFi; value: Import.frameInterval
                    onToggled: (on) => Import.overrideFi = on
                    onCommitted: (v) => Import.frameInterval = v
                }
            }
        }

        // ── start (wired in Phase 3) ─────────────────────────────────
        RowLayout {
            Layout.topMargin: sc.sp4
            spacing: sc.sp4
            Button { variant: "primary"; text: "Start analysis"; icon: "play"; enabled: Import.hasFile }
            Text {
                text: "Running is wired in Phase 3."
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                Layout.alignment: Qt.AlignVCenter
            }
        }
    }
}
