import QtQuick
import QtQuick.Layouts
import "components"
import "tabs"

// FIREFLY QML shell (Phase 1 foundation): themed header + landing/main pages.
// Hosted by a QQuickWidget (root is an Item, the QMainWindow owns the window).
Item {
    id: root
    implicitWidth: 1100
    implicitHeight: 760

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    Rectangle {            // app canvas
        anchors.fill: parent
        color: pal.BG

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            // ── header ───────────────────────────────────────────────
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 52
                color: pal.PANEL
                Rectangle {            // hairline base
                    anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
                    height: 1; color: pal.BORDER
                }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: sc.sp8
                    anchors.rightMargin: sc.sp8
                    spacing: sc.sp4
                    Text {
                        text: "FIRE"; color: pal.TXT
                        font.pixelSize: 17; font.bold: true
                        font.letterSpacing: 0.5
                    }
                    Text {
                        text: "FLY"; color: pal.ACC
                        font.pixelSize: 17; font.bold: true
                        font.letterSpacing: 0.5
                        Layout.leftMargin: -sc.sp4
                    }
                    Text {
                        text: "v" + appVersion; color: pal.TXT_MUTED
                        font.pixelSize: sc.textXs
                        Layout.leftMargin: sc.sp2
                    }
                    Item { Layout.fillWidth: true }
                    // Home affordance (only in the main UI)
                    Text {
                        visible: App.page === "main"
                        text: "‹ Home"; color: pal.TXT_MUTED
                        font.pixelSize: sc.textSm
                        MouseArea {
                            anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                            onClicked: App.goLanding()
                        }
                    }
                }
            }

            // ── body: landing or main ────────────────────────────────
            Loader {
                Layout.fillWidth: true
                Layout.fillHeight: true
                sourceComponent: App.page === "landing" ? landingPage : mainPage
            }
        }
    }

    // Manual-polygon ROI editor — a full-window modal shown while drawing.
    Loader {
        anchors.fill: parent
        active: Roi.editing
        sourceComponent: Component { RoiOverlay {} }
    }

    // ── landing page ─────────────────────────────────────────────────
    Component {
        id: landingPage
        Item {
            LandingBackdrop { anchors.fill: parent }      // glow + drifting dots

            ColumnLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 40
                anchors.rightMargin: Math.max(40, parent.width * 0.28)
                spacing: sc.sp3

                RowLayout {                                // eyebrow + microscope
                    spacing: sc.sp2
                    Icon { name: "microscope"; color: pal.WARN; size: 13 }
                    Text {
                        text: "FLUORESCENCE INFERENCE & RECONSTRUCTION ENGINE"
                        color: pal.WARN
                        font.pixelSize: sc.textXs; font.bold: true
                        font.letterSpacing: 2.0
                    }
                }
                Text {
                    text: "What would you like to do?"
                    color: pal.TXT
                    font.pixelSize: sc.displayMd; font.bold: true
                }
                Text {
                    text: "Single-particle tracking PALM · localise, link, and analyse single molecules."
                    color: pal.TXT_MUTED
                    font.pixelSize: sc.textLg
                    bottomPadding: sc.sp4
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: sc.sp6
                    rowSpacing: sc.sp6
                    Repeater {
                        model: [
                            { icon: "scan-search", t: "Analyse a sample", d: "Run the full pipeline on one .czi / .tif file.", tab: 0 },
                            { icon: "layers",      t: "Batch a folder",   d: "Process every file in a folder — in parallel on capable machines.", tab: 0 },
                            { icon: "git-compare", t: "Compare groups",   d: "Overlay 2–6 analysis-output folders into one figure.", tab: 2 },
                            { icon: "waypoints",   t: "Visualise tracks", d: "Open a previous run in the interactive viewer.", tab: 4 }
                        ]
                        delegate: Tile {
                            required property var modelData
                            Layout.fillWidth: true
                            icon: modelData.icon
                            title: modelData.t
                            desc: modelData.d
                            onClicked: App.enterMain(modelData.tab)
                        }
                    }
                }
            }
        }
    }

    // ── main page (left parameter dock + tab bar + content) ──────────
    Component {
        id: mainPage
        RowLayout {
            spacing: 0

            // ── persistent analysis-parameter dock (Import + Analysis) ───
            Rectangle {
                id: dock
                readonly property bool shown: App.currentTab === 0 || App.currentTab === 1
                Layout.preferredWidth: shown ? 312 : 0
                Layout.fillHeight: true
                visible: Layout.preferredWidth > 0
                clip: true
                color: pal.PANEL
                Behavior on Layout.preferredWidth {
                    NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic }
                }
                Rectangle {     // right hairline
                    anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                    width: 1; color: pal.BORDER
                }
                Flickable {
                    anchors.fill: parent
                    anchors.rightMargin: 1
                    contentWidth: width
                    contentHeight: paramCol.implicitHeight + sc.sp8
                    clip: true
                    ParameterSidebar {
                        id: paramCol
                        x: sc.sp4; y: sc.sp5
                        width: parent.width - sc.sp4 * 2
                    }
                }
            }

            // ── tab bar + per-tab content ────────────────────────────────
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0
                RowLayout {                 // tab pills
                    Layout.fillWidth: true
                    Layout.margins: sc.sp4
                    spacing: sc.sp2
                    Repeater {
                        model: App.tabs
                        delegate: Rectangle {
                            required property int index
                            required property string modelData
                            readonly property bool active: App.currentTab === index
                            implicitWidth: lbl.implicitWidth + sc.sp8 * 2
                            implicitHeight: 30
                            radius: sc.radiusLg
                            color: active ? Qt.rgba(0.345, 0.651, 1.0, 0.14) : "transparent"
                            border.width: 1
                            border.color: active ? pal.ACC : pal.BORDER
                            Text {
                                id: lbl; anchors.centerIn: parent; text: modelData
                                color: active ? pal.ACC : pal.TXT_MUTED
                                font.pixelSize: sc.textSm
                            }
                            TapHandler { onTapped: App.setTab(index) }
                        }
                    }
                    Item { Layout.fillWidth: true }
                }
                Loader {                     // per-tab content
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    sourceComponent: App.currentTab === 0 ? importTab
                                   : App.currentTab === 1 ? analysisTab
                                   : App.currentTab === 2 ? compareTab
                                   : App.currentTab === 3 ? resultsTab
                                   : App.currentTab === 4 ? visualiseTab
                                   : comingSoon
                }
            }
        }
    }

    // ── tab content components ───────────────────────────────────────
    Component { id: importTab; ImportTab {} }
    Component { id: analysisTab; AnalysisTab {} }
    Component { id: compareTab; CompareTab {} }
    Component { id: resultsTab; ResultsTab {} }
    Component { id: visualiseTab; VisualiseTab {} }
    Component {
        id: comingSoon
        Item {
            Text {
                anchors.centerIn: parent
                text: App.tabs[App.currentTab] + " — coming soon"
                color: pal.TXT_MUTED; font.pixelSize: sc.textLg
            }
        }
    }
}
