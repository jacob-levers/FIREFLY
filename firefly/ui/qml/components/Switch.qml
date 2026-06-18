import QtQuick

// Themed on/off toggle (track + sliding knob). Emits toggled(checked).
Item {
    id: root
    property bool checked: false
    signal toggled(bool checked)

    readonly property var pal: Theme.palette
    readonly property int dur: Theme.reducedMotion ? 0 : 130

    implicitWidth: 38
    implicitHeight: 20

    Rectangle {
        anchors.fill: parent
        radius: height / 2
        color: root.checked ? pal.ACC : pal.PANEL_ALT
        border.width: 1
        border.color: root.checked ? pal.ACC : pal.BORDER
        Behavior on color { ColorAnimation { duration: root.dur } }

        Rectangle {
            height: parent.height - 4
            width: height
            radius: height / 2
            y: 2
            x: root.checked ? parent.width - width - 2 : 2
            color: root.checked ? pal.ACC_FG : pal.TXT_MUTED
            Behavior on x { NumberAnimation { duration: root.dur; easing.type: Easing.OutCubic } }
        }
    }

    TapHandler {
        onTapped: {
            root.checked = !root.checked
            root.toggled(root.checked)
        }
    }
}
