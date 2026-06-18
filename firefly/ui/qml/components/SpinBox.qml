import QtQuick
import QtQuick.Layouts
import QtQuick.Controls

// Numeric field with step buttons. Supports doubles (decimals), an int mode
// (decimals 0), a suffix (e.g. " %"), and a special-value label shown when the
// value sits at `from` (the "off"/"auto"/"all" cases). Emits committed(real).
Rectangle {
    id: root
    property real value: 0
    property real from: 0
    property real to: 100
    property real step: 1
    property int decimals: 0
    property string suffix: ""
    property string special: ""        // shown instead of the number when value==from
    signal committed(real v)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property string display:
        (special !== "" && value <= from) ? special
        : (decimals > 0 ? value.toFixed(decimals) : Math.round(value).toString()) + suffix

    implicitHeight: 28
    implicitWidth: 120
    radius: sc.radiusSm
    color: pal.PANEL_ALT
    border.width: 1
    border.color: input.activeFocus ? pal.ACC : pal.BORDER
    Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }

    function _clamp(v) { return Math.max(from, Math.min(to, v)) }
    function _commit(v) {
        var c = _clamp(v)
        root.value = c
        root.committed(c)
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0
        TextInput {
            id: input
            Layout.fillWidth: true
            Layout.leftMargin: sc.sp3
            verticalAlignment: TextInput.AlignVCenter
            color: pal.TXT
            font.pixelSize: sc.textSm
            selectByMouse: true
            clip: true
            // show the formatted display unless the user is actively editing
            text: activeFocus ? text : root.display
            onActiveFocusChanged: if (activeFocus) {
                text = (root.special !== "" && root.value <= root.from)
                       ? "" : (root.decimals > 0 ? root.value.toFixed(root.decimals)
                                                 : Math.round(root.value).toString())
                selectAll()
            }
            onEditingFinished: {
                var v = parseFloat(text)
                if (!isNaN(v)) root._commit(v)
                else root.committed(root.value)   // revert display
                focus = false
            }
        }
        // step buttons
        ColumnLayout {
            Layout.preferredWidth: 16
            Layout.fillHeight: true
            spacing: 0
            Repeater {
                model: [{ ic: "chevron-up", d: 1 }, { ic: "chevron-down", d: -1 }]
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: stepHov.hovered ? pal.PANEL : "transparent"
                    Icon {
                        anchors.centerIn: parent
                        name: modelData.ic === "chevron-up" ? "chevron-down" : "chevron-down"
                        rotation: modelData.ic === "chevron-up" ? 180 : 0
                        size: 11; color: pal.TXT_MUTED
                    }
                    HoverHandler { id: stepHov; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: root._commit(root.value + modelData.d * root.step) }
                }
            }
        }
    }
}
