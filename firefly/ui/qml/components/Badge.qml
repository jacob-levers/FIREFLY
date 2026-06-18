import QtQuick
import QtQuick.Layouts

// A small pill: tinted fill + 1px tinted border, optional leading dot. Used for
// status (StatusPill = Badge with a status tone), counts, format chips, etc.
Rectangle {
    id: root
    property string text: ""
    property color tone: Theme.palette.TXT_MUTED
    property bool dot: false

    readonly property var sc: Theme.scale

    implicitHeight: 18
    implicitWidth: row.implicitWidth + sc.sp4 * 2
    radius: sc.radiusPill
    color: Qt.rgba(tone.r, tone.g, tone.b, 0.14)
    border.width: 1
    border.color: Qt.rgba(tone.r, tone.g, tone.b, 0.32)

    RowLayout {
        id: row
        anchors.centerIn: parent
        spacing: sc.sp2
        Rectangle {
            visible: root.dot
            width: 6; height: 6; radius: 3; color: root.tone
            Layout.alignment: Qt.AlignVCenter
        }
        Text {
            text: root.text; color: root.tone
            font.pixelSize: sc.textXs; font.bold: true
        }
    }
}
