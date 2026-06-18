import QtQuick
import QtQuick.Layouts
import "../components"

// Compare tab: configure 2–6 condition groups (label · colour · folders), set an
// output folder, and Generate → run_comparison; the overlaid figure appears here
// and the full report opens on the Results tab. Bound to the `Compare` controller.
Flickable {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    contentWidth: width
    contentHeight: grid.implicitHeight + 96
    clip: true

    GridLayout {
        id: grid
        x: 28; y: 20
        width: root.width - 56
        columns: width < 880 ? 1 : 2
        columnSpacing: sc.sp6
        rowSpacing: sc.sp6

        // ── left: overlaid figure ────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 460
            Layout.minimumWidth: 360
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: sc.sp4
                spacing: sc.sp3
                RowLayout {
                    Layout.fillWidth: true
                    Icon { name: "git-compare"; size: 14; color: pal.TXT_MUTED }
                    Text { text: "OVERLAID COMPARISON"; color: pal.TXT_MUTED
                           font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                    Item { Layout.fillWidth: true }
                    MotionLegend { model: Compare.motionClasses }
                }
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    radius: sc.radiusMd; color: "#000000"
                    border.width: 1; border.color: pal.BORDER; clip: true
                    Image {
                        anchors.fill: parent; anchors.margins: 1
                        visible: Compare.hasResult
                        fillMode: Image.PreserveAspectFit
                        smooth: true; cache: false; asynchronous: true
                        source: Compare.hasResult ? ("image://comparefig/" + Compare.overlayFigureToken) : ""
                    }
                    ColumnLayout {
                        anchors.centerIn: parent
                        visible: !Compare.hasResult
                        spacing: sc.sp2
                        Icon { name: "git-compare"; size: 36; color: pal.TXT_MUTED
                               Layout.alignment: Qt.AlignHCenter }
                        Text { text: Compare.running ? "Comparing…" : "Add 2+ folders per group, then Generate"
                               color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                               Layout.alignment: Qt.AlignHCenter }
                    }
                }
                // significance summary under the figure
                RowLayout {
                    Layout.fillWidth: true
                    visible: Compare.hasResult && Compare.pValueLabel !== ""
                    spacing: sc.sp3
                    Badge { text: Compare.pValueLabel
                            tone: Compare.significant ? pal.SUCCESS : pal.TXT_MUTED; dot: true }
                    Text { text: Compare.testLabel; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                           elide: Text.ElideRight; Layout.fillWidth: true }
                }
            }
        }

        // ── right: conditions + output + generate ────────────────────────
        ColumnLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 300
            Layout.alignment: Qt.AlignTop
            spacing: sc.sp6

            // conditions
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: condCol.implicitHeight + sc.sp5 * 2
                ColumnLayout {
                    id: condCol
                    x: sc.sp5; y: sc.sp5
                    width: parent.width - sc.sp5 * 2
                    spacing: sc.sp3
                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "CONDITIONS"; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                        Item { Layout.fillWidth: true }
                        Text { text: Compare.conditions.length + " / 6"; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs }
                    }
                    Repeater {
                        model: Compare.conditions
                        delegate: ColumnLayout {
                            required property var modelData
                            Layout.fillWidth: true
                            spacing: sc.sp2
                            RowLayout {
                                Layout.fillWidth: true; spacing: sc.sp3
                                Rectangle { width: 11; height: 11; radius: 3; color: modelData.colorHex
                                            Layout.alignment: Qt.AlignVCenter }
                                FieldInput {
                                    Layout.fillWidth: true
                                    text: modelData.name
                                    onEditingFinished: Compare.setLabel(modelData.id, text)
                                }
                                Badge { text: modelData.folderCount + (modelData.folderCount === 1 ? " folder" : " folders")
                                        tone: modelData.folderCount > 0 ? pal.ACC : pal.TXT_MUTED
                                        Layout.alignment: Qt.AlignVCenter }
                                IconButton { icon: "x"; tip: "Remove condition"; danger: true
                                             size: 26; onClicked: Compare.removeCondition(modelData.id) }
                            }
                            // folder drop + add row
                            DropArea {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 30
                                onDropped: (drop) => { if (drop.hasUrls) Compare.addFolders(modelData.id, drop.urls) }
                                Rectangle {
                                    anchors.fill: parent
                                    radius: sc.radiusSm
                                    color: parent.containsDrag ? Qt.rgba(0.345, 0.651, 1.0, 0.12) : pal.PANEL_ALT
                                    border.width: 1
                                    border.color: parent.containsDrag ? pal.ACC : pal.BORDER
                                    RowLayout {
                                        anchors.fill: parent; anchors.leftMargin: sc.sp3
                                        anchors.rightMargin: sc.sp2; spacing: sc.sp2
                                        Icon { name: "folder-plus"; size: 13; color: pal.TXT_MUTED }
                                        Text { text: "Drop run folders, or"; color: pal.TXT_MUTED
                                               font.pixelSize: sc.textXs; Layout.fillWidth: true }
                                        Text { text: "Browse…"; color: pal.ACC; font.pixelSize: sc.textXs
                                               MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                                           onClicked: Compare.browseAddFolder(modelData.id) } }
                                    }
                                }
                            }
                        }
                    }
                    Button {
                        Layout.fillWidth: true
                        variant: "secondary"; text: "Add condition"; icon: "plus"
                        enabled: Compare.conditions.length < 6
                        onClicked: Compare.addCondition()
                    }
                }
            }

            // group statistics (after a run)
            Card {
                Layout.fillWidth: true
                visible: Compare.statsRows.length > 0
                Layout.preferredHeight: statCol.implicitHeight + sc.sp5 * 2
                ColumnLayout {
                    id: statCol
                    x: sc.sp5; y: sc.sp5
                    width: parent.width - sc.sp5 * 2
                    spacing: sc.sp2
                    Text { text: "GROUP STATISTICS"; color: pal.TXT_MUTED
                           font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "Group"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.fillWidth: true }
                        Text { text: "D"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.preferredWidth: 56 }
                        Text { text: "α"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.preferredWidth: 44 }
                    }
                    Repeater {
                        model: Compare.statsRows
                        delegate: RowLayout {
                            required property var modelData
                            Layout.fillWidth: true
                            Text { text: modelData.groupLabel; color: pal.TXT; font.pixelSize: sc.textXs
                                   elide: Text.ElideRight; Layout.fillWidth: true }
                            Text { text: modelData.d; color: pal.TXT; font.pixelSize: sc.textXs
                                   font.family: "Menlo"; Layout.preferredWidth: 56 }
                            Text { text: modelData.a; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                   font.family: "Menlo"; Layout.preferredWidth: 44 }
                        }
                    }
                }
            }

            // output + generate
            ColumnLayout {
                Layout.fillWidth: true
                spacing: sc.sp3
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp3
                    Text { text: "Output"; color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                           Layout.preferredWidth: 52 }
                    Text { text: Compare.outputDir || "(choose a folder)"
                           color: Compare.outputDir ? pal.TXT : pal.TXT_MUTED
                           font.pixelSize: sc.textXs; font.family: "Menlo"
                           elide: Text.ElideMiddle; Layout.fillWidth: true }
                    Button { variant: "secondary"; text: "Choose…"; icon: "folder-open"
                             onClicked: Compare.browseOutputDir() }
                }
                Alert { visible: Compare.generateError !== ""; Layout.fillWidth: true
                        severity: "warn"; text: Compare.generateError }
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp4
                    Button {
                        variant: Compare.running ? "danger" : "primary"
                        text: Compare.running ? "Stop" : "Generate comparison"
                        icon: Compare.running ? "x" : "git-compare"
                        enabled: Compare.running || Compare.canGenerate
                        onClicked: Compare.running ? Compare.stop() : Compare.generate()
                    }
                    Text { text: Compare.status; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                           Layout.alignment: Qt.AlignVCenter; elide: Text.ElideRight; Layout.fillWidth: true }
                }
            }
        }
    }
}
