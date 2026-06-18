import QtQuick
import QtQuick.Layouts
import "../components"

// Results tab: a comparison's {stem}_results.json rendered as a figure card +
// headline metrics + group chips + per-metric verdict cards + output files.
// Bound to the `Results` controller (read-only; populated after a Compare run or
// via "Open a previous comparison…").
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
        width: root.width - 56
        spacing: sc.sp6

        RowLayout {
            Layout.fillWidth: true
            Text { text: Results.hasResults ? "Comparison results" : "Results"
                   color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true }
            Badge { visible: Results.hasResults; text: Results.headerTitle; tone: pal.ACC
                    Layout.alignment: Qt.AlignVCenter }
            Item { Layout.fillWidth: true }
            Button { variant: "secondary"; text: "Open previous…"; icon: "folder-open"
                     onClicked: Results.openPrevious() }
        }

        Alert { visible: Results.pairWarn !== ""; Layout.fillWidth: true
                severity: "warn"; text: Results.pairWarn }

        // group chips
        Flow {
            Layout.fillWidth: true
            visible: Results.groupChips.length > 0
            spacing: sc.sp3
            Repeater {
                model: Results.groupChips
                delegate: Rectangle {
                    required property var modelData
                    implicitWidth: chipRow.implicitWidth + sc.sp4 * 2
                    implicitHeight: 26; radius: sc.radiusPill
                    color: pal.PANEL; border.width: 1; border.color: pal.BORDER
                    RowLayout {
                        id: chipRow
                        anchors.centerIn: parent; spacing: sc.sp2
                        Rectangle { width: 9; height: 9; radius: 4.5; color: modelData.color }
                        Text { text: modelData.label; color: pal.TXT; font.pixelSize: sc.textXs }
                        Text { text: "· " + modelData.count; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                    }
                }
            }
        }

        // figure + right column
        GridLayout {
            Layout.fillWidth: true
            columns: width < 880 ? 1 : 2
            columnSpacing: sc.sp6
            rowSpacing: sc.sp6

            // figure card
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 420
                Layout.minimumWidth: 360
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: sc.sp4
                    spacing: sc.sp3
                    RowLayout {
                        Layout.fillWidth: true
                        Icon { name: "waypoints"; size: 14; color: pal.TXT_MUTED }
                        Text { text: "COMPARISON FIGURE"; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                        Item { Layout.fillWidth: true }
                        MotionLegend { model: Results.motionClasses }
                    }
                    Rectangle {
                        Layout.fillWidth: true; Layout.fillHeight: true
                        radius: sc.radiusMd; color: "#000000"
                        border.width: 1; border.color: pal.BORDER; clip: true
                        Image {
                            anchors.fill: parent; anchors.margins: 1
                            visible: Results.hasFigure
                            fillMode: Image.PreserveAspectFit
                            smooth: true; cache: false; asynchronous: true
                            source: Results.hasFigure ? ("image://resultfig/" + Results.figureToken) : ""
                        }
                        Text {
                            anchors.centerIn: parent; visible: !Results.hasFigure
                            text: "Run a comparison to see its figure"
                            color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                        }
                        MouseArea {
                            anchors.fill: parent; enabled: Results.hasFigure
                            cursorShape: Results.hasFigure ? Qt.PointingHandCursor : Qt.ArrowCursor
                            onClicked: Results.openUrl(Results.figureUrl)
                        }
                    }
                }
            }

            // right column: headline metrics + output files
            ColumnLayout {
                Layout.fillWidth: true
                Layout.minimumWidth: 320
                Layout.alignment: Qt.AlignTop
                spacing: sc.sp6

                Card {
                    Layout.fillWidth: true
                    Layout.preferredHeight: hm.implicitHeight + sc.sp5 * 2
                    ColumnLayout {
                        id: hm
                        x: sc.sp5; y: sc.sp5
                        width: parent.width - sc.sp5 * 2
                        spacing: sc.sp4
                        Text { text: "HEADLINE METRICS"; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                        GridLayout {
                            Layout.fillWidth: true
                            columns: 2; columnSpacing: sc.sp10; rowSpacing: sc.sp5
                            MetricStat { label: "Tracks"; value: Results.tracksLabel; tone: pal.ACC }
                            MetricStat { label: "Median D"; value: Results.medianD; unit: "µm²/s" }
                            MetricStat { label: "Median α"; value: Results.medianAlpha }
                            MetricStat { label: "α₂ (non-Gauss)"; value: Results.alpha2; tone: pal.WARN }
                        }
                    }
                }

                Card {
                    Layout.fillWidth: true
                    visible: Results.outputFiles.length > 0
                    Layout.preferredHeight: ofc.implicitHeight + sc.sp5 * 2
                    ColumnLayout {
                        id: ofc
                        x: sc.sp5; y: sc.sp5
                        width: parent.width - sc.sp5 * 2
                        spacing: sc.sp3
                        RowLayout {
                            Layout.fillWidth: true
                            Text { text: "OUTPUT FILES"; color: pal.TXT_MUTED
                                   font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                            Item { Layout.fillWidth: true }
                            Button { variant: "secondary"; text: "Open folder"; icon: "folder-open"
                                     enabled: Results.hasOutputFolder; onClicked: Results.openFolder() }
                        }
                        Repeater {
                            model: Results.outputFiles
                            delegate: Rectangle {
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: 28; radius: sc.radiusSm
                                color: rowHov.hovered ? pal.PANEL_ALT : "transparent"
                                RowLayout {
                                    anchors.fill: parent; anchors.leftMargin: sc.sp2
                                    anchors.rightMargin: sc.sp2; spacing: sc.sp2
                                    Icon {
                                        name: modelData.kind === "figure" ? "image"
                                            : modelData.kind === "pdf" ? "copy" : "database"
                                        size: 13; color: pal.TXT_MUTED
                                    }
                                    Text { text: modelData.relPath; color: pal.TXT
                                           font.pixelSize: sc.textXs; font.family: "Menlo"
                                           elide: Text.ElideMiddle; Layout.fillWidth: true }
                                    Icon { name: "arrow-up-right"; size: 12; color: pal.TXT_MUTED }
                                }
                                HoverHandler { id: rowHov }
                                TapHandler { onTapped: Results.openFile(modelData.path) }
                            }
                        }
                    }
                }
            }
        }

        // per-metric verdict cards
        ColumnLayout {
            Layout.fillWidth: true
            visible: Results.metricCards.length > 0
            spacing: sc.sp4
            Text { text: "Per-metric results"; color: pal.TXT; font.pixelSize: sc.textLg; font.bold: true }
            Repeater {
                model: Results.metricCards
                delegate: Card {
                    required property var modelData
                    Layout.fillWidth: true
                    Layout.preferredHeight: mc.implicitHeight + sc.sp5 * 2
                    ColumnLayout {
                        id: mc
                        x: sc.sp5; y: sc.sp5
                        width: parent.width - sc.sp5 * 2
                        spacing: sc.sp2
                        RowLayout {
                            Layout.fillWidth: true; spacing: sc.sp2
                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: modelData.severity === "success" ? pal.SUCCESS
                                     : modelData.severity === "warn" ? pal.WARN : pal.ACC
                            }
                            Text { text: modelData.title; color: pal.TXT
                                   font.pixelSize: sc.textMd; font.bold: true }
                        }
                        Text {
                            Layout.fillWidth: true
                            text: modelData.verdictHtml
                            textFormat: Text.RichText
                            color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }
    }
}
