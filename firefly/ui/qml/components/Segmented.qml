import QtQuick
import QtQuick.Layouts

// Segmented pill control.  `options` is a list of strings or {v,t,icon} maps;
// `value` is the selected value; `picked(v)` fires on tap.  `solid` gives the
// active segment a filled-accent look (used by the view toggle); otherwise the
// active segment uses the accent-soft treatment.
Rectangle {
    id: root
    property var options: []
    property string value
    property bool solid: false
    signal picked(string v)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    implicitHeight: 30
    radius: solid ? height / 2 : 8
    color: pal.PANEL_ALT
    border.width: 1
    border.color: pal.BORDER

    RowLayout {
        anchors.fill: parent
        anchors.margins: 3
        spacing: 3
        Repeater {
            model: root.options
            delegate: Rectangle {
                required property var modelData
                readonly property bool isStr: (typeof modelData === "string")
                readonly property string v: isStr ? modelData : modelData.v
                readonly property string t: isStr ? modelData : modelData.t
                readonly property string ic: isStr ? "" : (modelData.icon || "")
                readonly property bool on: root.value === v
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: root.solid ? height / 2 : 6
                color: on ? (root.solid ? pal.ACC : Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14))
                          : "transparent"
                Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                RowLayout {
                    anchors.centerIn: parent
                    spacing: sc.sp1
                    Icon {
                        visible: ic !== ""
                        name: ic; size: 13
                        color: on ? (root.solid ? "#08111d" : pal.ACC) : pal.TXT_MUTED
                    }
                    Text {
                        text: t
                        font.pixelSize: sc.textSm; font.bold: true
                        color: on ? (root.solid ? "#08111d" : pal.ACC) : pal.TXT_MUTED
                    }
                }
                TapHandler { onTapped: root.picked(v) }
                HoverHandler { cursorShape: Qt.PointingHandCursor }
            }
        }
    }
}
