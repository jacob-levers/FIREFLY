import QtQuick
import QtQuick.Layouts

// Themed button. variant: "primary" (accent fill) | "secondary" (panel + border)
// | "danger" (red on hover). Optional leading icon. Hover lightens, press
// darkens — no scale/bounce, per the brief.
Rectangle {
    id: root
    property string text: ""
    property string icon: ""
    property string variant: "secondary"
    signal clicked()
    // `enabled` is the built-in Item property (controls input + our opacity).

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property bool primary: variant === "primary"
    readonly property bool danger: variant === "danger"
    readonly property int dur: Theme.reducedMotion ? 0 : 130
    readonly property color fg: primary ? pal.ACC_FG
                                        : (danger && hov.hovered ? pal.DANGER : pal.TXT)

    implicitHeight: 30
    implicitWidth: row.implicitWidth + sc.sp8 * 2
    radius: sc.radiusMd
    opacity: enabled ? 1.0 : 0.45
    color: !enabled ? pal.PANEL_ALT
         : primary ? (press.pressed ? pal.ACC_PRESSED : hov.hovered ? pal.ACC_HOVER : pal.ACC)
         : (press.pressed ? pal.BG : hov.hovered ? pal.PANEL_ALT : pal.PANEL)
    border.width: primary ? 0 : 1
    border.color: (danger && hov.hovered) ? pal.DANGER
                : hov.hovered ? pal.BORDER_HI : pal.BORDER
    Behavior on color { ColorAnimation { duration: root.dur } }
    Behavior on border.color { ColorAnimation { duration: root.dur } }

    RowLayout {
        id: row
        anchors.centerIn: parent
        spacing: sc.sp2
        Icon {
            visible: root.icon !== ""
            name: root.icon; size: 15; color: root.fg
        }
        Text {
            visible: root.text !== ""
            text: root.text; color: root.fg
            font.pixelSize: sc.textSm; font.bold: root.primary
        }
    }

    HoverHandler { id: hov; enabled: root.enabled; cursorShape: Qt.PointingHandCursor }
    TapHandler { id: press; enabled: root.enabled; onTapped: root.clicked() }
}
