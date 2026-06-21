import QtQuick

// Themed checkbox (box + check glyph). Distinct from Switch — used for discrete
// options (export formats, notifications) per the design system. Emits
// toggled(checked).
Item {
    id: root
    property bool checked: false
    signal toggled(bool checked)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    implicitWidth: 18
    implicitHeight: 18

    Rectangle {
        anchors.fill: parent
        radius: sc.radiusXs
        color: root.checked ? pal.ACC : pal.PANEL_ALT
        border.width: 1
        border.color: root.checked ? pal.ACC : (hov.hovered ? pal.BORDER_HI : pal.BORDER)
        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
        Icon {
            anchors.centerIn: parent
            visible: root.checked
            name: "check"; size: 12; color: pal.ACC_FG
            scale: root.checked ? 1 : 0.6        // tiny pop on check
            Behavior on scale { NumberAnimation { duration: Theme.reducedMotion ? 0 : 90; easing.type: Easing.OutCubic } }
        }
    }

    HoverHandler { id: hov; cursorShape: Qt.PointingHandCursor }
    TapHandler {
        onTapped: { root.checked = !root.checked; root.toggled(root.checked) }
    }
}
