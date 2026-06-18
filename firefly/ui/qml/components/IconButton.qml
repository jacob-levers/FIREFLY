import QtQuick
import QtQuick.Controls

// Icon-only button (header / transport / tool actions). Hover raises a panel
// fill + border; `danger` tints red on hover (close / destructive).
Rectangle {
    id: root
    property string icon: ""
    property int size: 30
    property bool danger: false
    property string tip: ""
    signal clicked()

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    implicitWidth: size
    implicitHeight: size
    radius: sc.radiusSm
    color: hov.hovered ? pal.PANEL_ALT : "transparent"
    border.width: 1
    border.color: hov.hovered ? (root.danger ? pal.DANGER : pal.BORDER_HI) : "transparent"
    Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 130 } }

    Icon {
        anchors.centerIn: parent
        name: root.icon
        size: Math.round(root.size * 0.5)
        color: (root.danger && hov.hovered) ? pal.DANGER
             : hov.hovered ? pal.TXT : pal.TXT_MUTED
    }

    ToolTip.visible: root.tip !== "" && hov.hovered
    ToolTip.text: root.tip

    HoverHandler { id: hov; cursorShape: Qt.PointingHandCursor }
    TapHandler { onTapped: root.clicked() }
}
