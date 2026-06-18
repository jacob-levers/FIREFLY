import QtQuick
import QtQuick.Layouts

// One headline-metric cell: small label over a large value (+ optional unit).
// The collapse-proof building block of the Results/Compare metrics grid.
ColumnLayout {
    id: root
    property string label: ""
    property string value: "—"
    property string unit: ""
    property color tone: Theme.palette.TXT
    readonly property var sc: Theme.scale
    readonly property var pal: Theme.palette
    spacing: 2

    Text { text: root.label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
    RowLayout {
        spacing: sc.sp2
        Text { text: root.value; color: root.tone; font.pixelSize: sc.textXl; font.bold: true }
        Text {
            visible: root.unit !== ""
            text: root.unit; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
            Layout.alignment: Qt.AlignBottom
            bottomPadding: 3
        }
    }
}
