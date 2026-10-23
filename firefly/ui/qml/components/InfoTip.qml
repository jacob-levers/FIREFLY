import QtQuick
import QtQuick.Controls as QQC

// A small "info" glyph that shows `text` on hover — for help copy that would
// otherwise sit in the panel as a permanent paragraph.  The Basic style's own
// ToolTip never wraps (its implicit width is the one-line text width), so this
// one caps the width at `maxWidth` and uses the theme's panel colours.
Item {
    id: root
    property string text: ""
    property int size: 13
    property int maxWidth: 280

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    implicitWidth: size
    implicitHeight: size
    visible: text !== ""

    Icon {
        anchors.centerIn: parent
        name: "info"; size: root.size
        color: hov.hovered ? root.pal.TXT : root.pal.TXT_MUTED
    }
    // Pad the hit area so the glyph is easy to land on without crowding it.
    HoverHandler { id: hov; margin: 4; cursorShape: Qt.WhatsThisCursor }

    QQC.ToolTip {
        visible: hov.hovered && root.text !== ""
        delay: 250
        text: root.text
        width: Math.min(implicitWidth, root.maxWidth)
        padding: root.sc.sp4
        contentItem: Text {
            text: root.text
            wrapMode: Text.WordWrap
            color: root.pal.TXT
            font.pixelSize: root.sc.textXs
            lineHeight: 1.3
        }
        background: Rectangle {
            color: root.pal.PANEL_ALT
            border.color: root.pal.BORDER_HI
            radius: root.sc.radiusMd
        }
    }
}
