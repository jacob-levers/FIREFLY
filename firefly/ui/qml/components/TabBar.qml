import QtQuick
import QtQuick.Layouts

// TabBar — a complete sliding-highlight tab bar. ONE accent pill slides between
// tabs instead of each tab popping its own colour; the active label crossfades
// and its (optional) icon does a tiny scale-pop. Drop-in for the tab row in
// Main.qml.
//
//   TabBar {
//       model: [{ t: "Import", icon: "scan-search" }, { t: "Analysis", icon: "cpu" }, …]
//       current: App.currentTab
//       onPicked: (i) => App.setTab(i)
//   }
Item {
    id: root
    property var model: []
    property int current: 0
    signal picked(int index)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    implicitHeight: 38
    implicitWidth: row.implicitWidth

    // sliding highlight (behind the labels)
    Rectangle {
        id: pill
        property Item t: row.children[root.current] || null
        x:      t ? t.x : 0
        width:  t ? t.width : 0
        height: 30
        anchors.verticalCenter: parent.verticalCenter
        radius: sc.radiusLg
        color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14)      // ACC @ 14%
        border.width: 1
        border.color: pal.ACC
        Behavior on x     { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic } }
        Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic } }
    }

    Row {
        id: row
        spacing: 4
        Repeater {
            model: root.model
            delegate: Item {
                required property int index
                required property var modelData
                readonly property bool on: root.current === index
                width: cell.implicitWidth + sc.sp8 * 2
                height: 30
                anchors.verticalCenter: parent.verticalCenter

                RowLayout {
                    id: cell
                    anchors.centerIn: parent
                    spacing: sc.sp2
                    Icon {
                        visible: modelData.icon !== undefined
                        name: modelData.icon || ""
                        size: 13
                        color: parent.parent.on ? pal.ACC : pal.TXT_MUTED
                        scale: parent.parent.on ? 1 : 0.92
                        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                        Behavior on scale { NumberAnimation { duration: Theme.reducedMotion ? 0 : 120; easing.type: Easing.OutCubic } }
                    }
                    Text {
                        text: modelData.t
                        font.pixelSize: sc.textSm
                        font.bold: parent.parent.on
                        color: parent.parent.on ? pal.ACC : pal.TXT_MUTED
                        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                    }
                }
                TapHandler { onTapped: root.picked(index) }
                HoverHandler { cursorShape: Qt.PointingHandCursor }
            }
        }
    }
}
