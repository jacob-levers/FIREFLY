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
    spacing: sc.sp3

    RowLayout {
        Layout.fillWidth: true
        Layout.bottomMargin: sc.sp1
        spacing: sc.sp3
        Rectangle {                              // tinted icon-chip
            Layout.preferredWidth: 24; Layout.preferredHeight: 24
            radius: sc.radiusLg
            color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14)   // ACC @ 14%
            Icon { anchors.centerIn: parent; name: "sliders-horizontal"
                   size: 14; color: pal.ACC }
        }
        Text {
            text: "ANALYSIS PARAMETERS"; color: pal.TXT_MUTED
            font.pixelSize: sc.textXs; font.weight: Font.DemiBold
            font.letterSpacing: 0.8
        }
        Item { Layout.fillWidth: true }
        Text {
            text: "Reset all"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                        onClicked: Sidebar.resetAll() }
        }
    }

    // ── preset strip ──────────────────────────────────────────────────────
    RowLayout {
        Layout.fillWidth: true
        spacing: sc.sp2
        Select {
            Layout.fillWidth: true
            Layout.preferredWidth: 0
            model: Preset.names
            currentIndex: Math.max(0, Preset.names.indexOf(Preset.active))
            onPicked: (t) => Preset.load(t)
        }
        Badge { visible: Preset.modified; text: "• modified"; tone: pal.WARN
                Layout.alignment: Qt.AlignVCenter }
        IconButton { icon: "plus"; tip: "Save preset…"; onClicked: Preset.saveAs() }
        IconButton { icon: "x"; tip: "Delete preset"; danger: true
                     onClicked: Preset.confirmRemove(Preset.active) }
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

            // ROI section: manual-polygon editor entry point
            Button {
                visible: modelData.key === "roi"
                         && (Sidebar.revision, Sidebar.get("analysis/roi_mode") === "Manual polygon")
                width: parent ? parent.width : implicitWidth
                variant: "secondary"
                text: !Import.hasFile ? "Pick an image file to draw an ROI"
                    : Roi.fileHasRoi(Import.filePath) ? "Edit polygon ROI ✓" : "Draw polygon ROI…"
                icon: "move"
                enabled: Import.hasFile && !Import.isCsv
                onClicked: Roi.editFile(Import.filePath)
            }
        }
    }
}
