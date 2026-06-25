import QtQuick
import QtQuick.Controls

// Themed single-line text field: raised input fill, border that thickens to a
// 2px accent ring on focus (the design system's focus cue — visible for
// keyboard navigation, not just a hairline tint).
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
        border.width: root.activeFocus ? 2 : 1
        border.color: root.activeFocus ? pal.ACC : pal.BORDER
        Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
        Behavior on border.width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
    }
}
