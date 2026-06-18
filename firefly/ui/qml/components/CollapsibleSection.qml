import QtQuick
import QtQuick.Layouts

// Accordion section: header (chevron + per-concern icon + title) with an accent
// left-bar when open, and animated-height content. Children declared inside are
// placed in the content column. Used by every parameter sidebar.
Column {
    id: root
    property string title: ""
    property string icon: ""
    property bool expanded: true
    default property alias content: body.data

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    spacing: 0
    width: parent ? parent.width : implicitWidth

    Rectangle {                                  // header
        width: parent.width
        height: 34
        radius: sc.radiusSm
        color: hh.hovered ? pal.PANEL_ALT : pal.PANEL

        Rectangle {
            visible: root.expanded
            anchors { left: parent.left; top: parent.top; bottom: parent.bottom }
            width: 3; radius: sc.radiusSm; color: pal.ACC
        }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: sc.sp6
            anchors.rightMargin: sc.sp4
            spacing: sc.sp3
            Icon {
                name: root.expanded ? "chevron-down" : "chevron-right"
                color: pal.TXT_MUTED; size: 14
            }
            Icon { visible: root.icon !== ""; name: root.icon; color: pal.ACC; size: 14 }
            Text {
                text: root.title; color: pal.TXT
                font.pixelSize: sc.textMd; font.bold: true
            }
            Item { Layout.fillWidth: true }
        }
        HoverHandler { id: hh; cursorShape: Qt.PointingHandCursor }
        TapHandler { onTapped: root.expanded = !root.expanded }
    }

    Item {                                       // animated content
        width: parent.width
        clip: true
        height: root.expanded ? body.implicitHeight + sc.sp2 : 0
        Behavior on height {
            NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic }
        }
        Column {
            id: body
            width: parent.width
            topPadding: sc.sp3
            spacing: sc.sp3
        }
    }
}
