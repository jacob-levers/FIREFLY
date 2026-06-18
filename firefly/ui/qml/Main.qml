import QtQuick
import QtQuick.Layouts

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

    // ── landing page ─────────────────────────────────────────────────
    Component {
        id: landingPage
        Item {
            ColumnLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 40
                anchors.rightMargin: Math.max(40, parent.width * 0.28)
                spacing: sc.sp3
                Text {
                    text: "FLUORESCENCE INFERENCE & RECONSTRUCTION ENGINE"
                    color: pal.WARN
                    font.pixelSize: sc.textXs; font.bold: true
                    font.letterSpacing: 2.0
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
                            { t: "Analyse a sample", d: "Run the full pipeline on one .czi / .tif file.", tab: 0 },
                            { t: "Batch a folder",   d: "Process every file in a folder — in parallel on capable machines.", tab: 0 },
                            { t: "Compare groups",   d: "Overlay 2–6 analysis-output folders into one figure.", tab: 2 },
                            { t: "Visualise tracks", d: "Open a previous run in the interactive viewer.", tab: 4 }
                        ]
                        delegate: Rectangle {
                            required property var modelData
                            Layout.fillWidth: true
                            Layout.preferredHeight: 96
                            radius: sc.radius2xl
                            color: hov.hovered ? pal.PANEL_ALT : pal.PANEL
                            border.width: 1
                            border.color: hov.hovered ? pal.ACC : pal.BORDER
                            Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 140 } }
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: sc.sp6
                                spacing: sc.sp2
                                Rectangle {            // icon chip placeholder
                                    width: 34; height: 34; radius: sc.radius2xl
                                    color: Qt.rgba(0.345, 0.651, 1.0, 0.10)
                                    border.color: Qt.rgba(0.345, 0.651, 1.0, 0.22)
                                    border.width: 1
                                    Rectangle {
                                        anchors.centerIn: parent
                                        width: 12; height: 12; radius: 3
                                        color: pal.ACC
                                    }
                                }
                                Item { Layout.fillHeight: true }
                                Text {
                                    text: modelData.t; color: pal.TXT
                                    font.pixelSize: sc.textLg; font.bold: true
                                }
                                Text {
                                    text: modelData.d; color: pal.TXT_MUTED
                                    font.pixelSize: sc.textSm
                                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                                }
                            }
                            HoverHandler { id: hov }
                            TapHandler { onTapped: App.enterMain(modelData.tab) }
                        }
                    }
                }
            }
        }
    }

    // ── main page (tab bar + placeholder content) ────────────────────
    Component {
        id: mainPage
        ColumnLayout {
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
            Rectangle {
                Layout.fillWidth: true; Layout.fillHeight: true
                color: pal.BG
                Text {
                    anchors.centerIn: parent
                    text: App.tabs[App.currentTab] + " — coming soon"
                    color: pal.TXT_MUTED; font.pixelSize: sc.textLg
                }
            }
        }
    }
}
