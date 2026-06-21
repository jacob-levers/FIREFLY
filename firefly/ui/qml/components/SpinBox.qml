import QtQuick

// Numeric field (plain text entry — no stepper buttons). Supports doubles
// (decimals), an int mode (decimals 0), a suffix (e.g. " %"), and a
// special-value label shown when the value sits at `from` (the "off"/"auto"/
// "all" cases). Emits committed(real).
Rectangle {
    id: root
    property real value: 0
    property real from: 0
    property real to: 100
    property real step: 1               // retained for API compatibility
    property int decimals: 0
    property string suffix: ""
    property string special: ""         // shown instead of the number when value==from
    property int textAlign: TextInput.AlignLeft   // e.g. AlignHCenter for short values
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

    TextInput {
        id: input
        anchors.fill: parent
        anchors.leftMargin: sc.sp3
        anchors.rightMargin: sc.sp3
        horizontalAlignment: root.textAlign
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
}
