import QtQuick
import QtQuick.Controls

// Themed single-line text field: raised input fill, hairline border that turns
// accent on focus (the design system's focus cue).
TextField {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    color: pal.TXT
    placeholderTextColor: pal.TXT_MUTED
    font.pixelSize: sc.textSm
    selectByMouse: true
    leftPadding: sc.sp4
    rightPadding: sc.sp4
    topPadding: sc.sp3
    bottomPadding: sc.sp3

    background: Rectangle {
        radius: sc.radiusSm
        color: pal.PANEL_ALT
        border.width: 1
        border.color: root.activeFocus ? pal.ACC : pal.BORDER
        Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
    }
}
