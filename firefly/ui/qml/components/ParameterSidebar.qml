import QtQuick
import QtQuick.Layouts
import "."

// The analysis-parameter sidebar: one collapsible section per concern (Imaging
// metadata … Figures), each a Repeater of FieldRow bound to the SidebarController.
// Every edit writes straight through to the QSettings keys params_builder reads,
// so the worker param dict stays byte-identical. Drop into any column layout.
ColumnLayout {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    spacing: sc.sp2

    RowLayout {
        Layout.fillWidth: true
        Icon { name: "sliders-horizontal"; size: 14; color: pal.ACC }
        Text { text: "Analysis parameters"; color: pal.TXT
               font.pixelSize: sc.textLg; font.bold: true }
        Item { Layout.fillWidth: true }
        Text {
            text: "Reset all"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                        onClicked: Sidebar.resetAll() }
        }
    }

    Repeater {
        model: Sidebar.sections
        delegate: CollapsibleSection {
            required property var modelData
            Layout.fillWidth: true
            title: modelData.title
            icon: modelData.icon
            expanded: false

            Repeater {
                model: Sidebar.fields(modelData.key)
                delegate: FieldRow {
                    required property var modelData
                    width: parent ? parent.width : implicitWidth
                    field: modelData
                }
            }
        }
    }
}
