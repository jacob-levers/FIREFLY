import QtQuick
import QtQuick.Layouts
import "../components"

// Import tab: pick an input file (typed chip + format/frame badges), choose an
// output folder, and set calibration (pixel size / frame interval, each with a
// metadata-override). Bound to ImportController; running lands in Phase 3.
Flickable {
    id: root
    property bool batchMode: false
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

        // ── single / batch mode toggle ────────────────────────────────
        RowLayout {
            spacing: sc.sp2
            Repeater {
                model: [{ t: "Single file", b: false, ic: "scan-search" },
                        { t: "Batch folder", b: true, ic: "layers" }]
                delegate: Rectangle {
                    required property var modelData
                    readonly property bool active: root.batchMode === modelData.b
                    implicitWidth: pillRow.implicitWidth + sc.sp6 * 2
                    implicitHeight: 32; radius: sc.radiusLg
                    color: active ? Qt.rgba(0.345, 0.651, 1.0, 0.14) : pal.PANEL_ALT
                    border.width: 1; border.color: active ? pal.ACC : pal.BORDER
                    RowLayout {
                        id: pillRow; anchors.centerIn: parent; spacing: sc.sp2
                        Icon { name: modelData.ic; size: 14; color: active ? pal.ACC : pal.TXT_MUTED }
                        Text { text: modelData.t; color: active ? pal.ACC : pal.TXT_MUTED
                               font.pixelSize: sc.textSm }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: root.batchMode = modelData.b }
                }
            }
        }

        Text { visible: !root.batchMode; text: "Input"; color: pal.TXT
               font.pixelSize: sc.textXl; font.bold: true }

        // ── file chip ────────────────────────────────────────────────
        Card {
            visible: !root.batchMode
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
            visible: !root.batchMode
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

        // ── batch folder + series list ────────────────────────────────
        ColumnLayout {
            visible: root.batchMode
            Layout.fillWidth: true
            spacing: sc.sp4
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 64
                RowLayout {
                    anchors.fill: parent; anchors.margins: sc.sp5; spacing: sc.sp4
                    Icon { name: "layers"; size: 20; color: Batch.folder ? pal.ACC : pal.TXT_MUTED }
                    ColumnLayout {
                        Layout.fillWidth: true; spacing: sc.sp1
                        Text { text: Batch.folder || "No folder selected"
                               color: Batch.folder ? pal.TXT : pal.TXT_MUTED
                               font.pixelSize: sc.textSm; font.family: Batch.folder ? "Menlo" : Qt.application.font.family
                               elide: Text.ElideMiddle; Layout.fillWidth: true; Layout.preferredWidth: 0 }
                        Text { text: Batch.summary; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                    }
                    Switch { checked: Batch.recursive; onToggled: (c) => Batch.recursive = c }
                    Text { text: "Subfolders"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                    Button { variant: "secondary"; text: "Browse…"; icon: "folder-open"
                             onClicked: Batch.browseFolder() }
                }
            }
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp4
                Text { text: "Output"; color: pal.TXT_MUTED; font.pixelSize: sc.textSm; Layout.preferredWidth: 60 }
                Text { text: Batch.outputDir || (Batch.folder ? "(folder)/batch_results" : "—")
                       color: Batch.outputDir ? pal.TXT : pal.TXT_MUTED
                       font.pixelSize: sc.textXs; font.family: "Menlo"
                       elide: Text.ElideMiddle; Layout.fillWidth: true; Layout.preferredWidth: 0 }
                Button { variant: "secondary"; text: "Choose…"; icon: "folder-open"
                         onClicked: Batch.browseOutputDir() }
            }
            RowLayout {
                visible: Batch.series.length > 0
                Layout.fillWidth: true; spacing: sc.sp3
                Text { text: "Select"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.fillWidth: true }
                Text { text: "All"; color: pal.ACC; font.pixelSize: sc.textXs
                       MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                   onClicked: Batch.selectAll(true) } }
                Text { text: "None"; color: pal.ACC; font.pixelSize: sc.textXs
                       MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                   onClicked: Batch.selectAll(false) } }
            }
            Card {
                visible: Batch.series.length > 0
                Layout.fillWidth: true
                Layout.preferredHeight: Math.min(280, seriesCol.implicitHeight + sc.sp4 * 2)
                Flickable {
                    anchors.fill: parent; anchors.margins: sc.sp4
                    contentWidth: width; contentHeight: seriesCol.implicitHeight; clip: true
                    ColumnLayout {
                        id: seriesCol
                        width: parent.width; spacing: 1
                        Repeater {
                            model: Batch.series
                            delegate: Rectangle {
                                required property var modelData
                                Layout.fillWidth: true; implicitHeight: 26; radius: sc.radiusXs
                                color: srowHov.hovered ? pal.PANEL_ALT : "transparent"
                                RowLayout {
                                    anchors.fill: parent; anchors.leftMargin: sc.sp2
                                    anchors.rightMargin: sc.sp2; spacing: sc.sp3
                                    Icon { name: modelData.checked ? "circle-check" : "circle-dot"
                                           size: 14; color: modelData.checked ? pal.ACC : pal.TXT_MUTED }
                                    Text { text: modelData.name; color: pal.TXT; font.pixelSize: sc.textXs
                                           font.family: "Menlo"; elide: Text.ElideMiddle
                                           Layout.fillWidth: true; Layout.preferredWidth: 0 }
                                    Text { visible: modelData.fileCount > 1
                                           text: "×" + modelData.fileCount; color: pal.TXT_MUTED
                                           font.pixelSize: sc.textXs }
                                }
                                HoverHandler { id: srowHov }
                                TapHandler { onTapped: Batch.setChecked(modelData.key, !modelData.checked) }
                            }
                        }
                    }
                }
            }
            Alert { visible: Batch.generateError !== ""; Layout.fillWidth: true
                    severity: "warn"; text: Batch.generateError }

            // batch progress
            ColumnLayout {
                visible: Batch.running || Batch.status !== ""
                Layout.fillWidth: true; spacing: sc.sp2
                Rectangle {
                    Layout.fillWidth: true; implicitHeight: 8; radius: 4
                    color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                    Rectangle {
                        height: parent.height
                        width: Math.max(0, Math.min(1, Batch.progress / 100)) * parent.width
                        radius: 4
                        gradient: Gradient {
                            orientation: Gradient.Horizontal
                            GradientStop { position: 0.0; color: pal.SUCCESS }
                            GradientStop { position: 1.0; color: pal.ACC }
                        }
                        Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                    }
                }
                Text { text: Batch.status; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
            }
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

        // ── start ────────────────────────────────────────────────────
        RowLayout {
            Layout.topMargin: sc.sp4
            spacing: sc.sp4
            Button {
                visible: !root.batchMode
                variant: "primary"; text: "Start analysis"; icon: "play"
                enabled: Import.hasFile && !Analysis.running
                onClicked: { App.setTab(1); Analysis.start(); }
            }
            Button {
                visible: root.batchMode
                variant: Batch.running ? "danger" : "primary"
                text: Batch.running ? "Stop batch" : "Start batch"
                icon: Batch.running ? "x" : "play"
                enabled: Batch.running || Batch.canRun
                onClicked: Batch.running ? Batch.stop() : Batch.generate()
            }
            Text {
                text: root.batchMode
                      ? (Batch.canRun || Batch.running ? "Processes the selected series."
                                                       : "Pick a folder and select series.")
                      : (Import.hasFile ? "Runs on the Analysis tab."
                                        : "Pick an input file to begin.")
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                Layout.alignment: Qt.AlignVCenter
            }
        }
    }
}
