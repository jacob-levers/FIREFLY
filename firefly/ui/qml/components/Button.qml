import QtQuick
import QtQuick.Layouts
import QtQuick.Controls

// Themed button. variant: "primary" (accent fill) | "secondary" (panel + border)
// | "danger" (red on hover). Optional leading icon. Hover lightens, press
// darkens — no scale/bounce, per the brief.  Icon-only (text:"") renders as a
// compact square; set `tip` for a hover tooltip.
Rectangle {
    id: root
    property string text: ""
    property string icon: ""
    property string variant: "secondary"
    property string tip: ""                    // hover tooltip (great for icon-only)
    property bool spin: false                  // continuously rotate the icon
    signal clicked()
    // `enabled` is the built-in Item property (controls input + our opacity).

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property bool primary: variant === "primary"
    readonly property bool danger: variant === "danger"
    readonly property bool ghost: variant === "ghost"
    readonly property int dur: Theme.reducedMotion ? 0 : 130
    readonly property color fg: primary ? pal.ACC_FG
                                        : ghost ? (hov.hovered ? pal.TXT : pal.TXT_MUTED)
                                        : (danger && hov.hovered ? pal.DANGER : pal.TXT)

    implicitHeight: 30
    // icon-only → square; labelled → text width + padding
    implicitWidth: text !== "" ? row.implicitWidth + sc.sp8 * 2 : implicitHeight
    radius: sc.radiusMd
    opacity: enabled ? 1.0 : 0.45
    color: !enabled ? (ghost ? "transparent" : pal.PANEL_ALT)
         : primary ? (press.pressed ? pal.ACC_PRESSED : hov.hovered ? pal.ACC_HOVER : pal.ACC)
         : ghost ? (hov.hovered ? pal.PANEL_ALT : "transparent")
         : (press.pressed ? pal.BG : hov.hovered ? pal.PANEL_ALT : pal.PANEL)
    border.width: (primary || ghost) ? 0 : 1
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
            RotationAnimator on rotation {
                running: root.spin && !Theme.reducedMotion
                from: 0; to: 360; duration: 900; loops: Animation.Infinite
            }
        }
        Text {
            visible: root.text !== ""
            text: root.text; color: root.fg
            font.pixelSize: sc.textSm; font.bold: root.primary
        }
    }

    HoverHandler { id: hov; enabled: root.enabled; cursorShape: Qt.PointingHandCursor }
    TapHandler { id: press; enabled: root.enabled; onTapped: root.clicked() }

    ToolTip.text: root.tip
    ToolTip.delay: 500
    ToolTip.visible: root.tip !== "" && hov.hovered
}
