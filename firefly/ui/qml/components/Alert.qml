import QtQuick
import QtQuick.Layouts

// Severity callout: tinted fill + accent left-bar + icon + message.
// severity: "info" | "success" | "warn" | "danger".
Rectangle {
    id: root
    property string text: ""
    property string severity: "info"

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property color tone: severity === "success" ? pal.SUCCESS
                                : severity === "warn"    ? pal.WARN
                                : severity === "danger"  ? pal.DANGER : pal.ACC
    readonly property string ic: severity === "success" ? "circle-check"
                               : severity === "warn"    ? "triangle-alert"
                               : severity === "danger"  ? "triangle-alert" : "info"

    implicitHeight: row.implicitHeight + sc.sp6 * 2
    radius: sc.radiusMd
    color: Qt.rgba(tone.r, tone.g, tone.b, 0.10)
    border.width: 1
    border.color: Qt.rgba(tone.r, tone.g, tone.b, 0.30)

    Rectangle {
        anchors { left: parent.left; top: parent.top; bottom: parent.bottom }
        width: 3; radius: root.radius; color: root.tone
    }

    RowLayout {
        id: row
        anchors.fill: parent
        anchors.margins: sc.sp6
        anchors.leftMargin: sc.sp8
        spacing: sc.sp4
        Icon { name: root.ic; color: root.tone; size: 16; Layout.alignment: Qt.AlignTop }
        Text {
            text: root.text; color: pal.TXT; font.pixelSize: sc.textSm
            Layout.fillWidth: true; wrapMode: Text.WordWrap
        }
    }
}
