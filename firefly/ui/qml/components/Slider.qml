import QtQuick

// Horizontal value slider: gradient-filled track + glowing knob + a mono value
// readout. Drag or click the track to set; emits moved(real) live and
// committed(real) on release. Used by sidebar fields flagged slider:true.
Item {
    id: root
    property real value: 0
    property real from: 0
    property real to: 1
    property real step: 0
    property int decimals: 2
    property string suffix: ""
    property bool showValue: true            // hide the readout for scrubbers
    signal moved(real v)
    signal committed(real v)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property real frac: to > from ? Math.max(0, Math.min(1, (value - from) / (to - from))) : 0

    implicitHeight: 22
    implicitWidth: 160

    // Emit the new value but DON'T assign root.value — the owner binds value to
    // the model and writes it back via onMoved, so the binding stays live
    // (preset / reset still move the knob). Assigning here would sever it.
    function _apply(px, commit) {
        var w = track.width - knob.width
        var f = w > 0 ? Math.max(0, Math.min(1, (px - knob.width / 2) / w)) : 0
        var v = from + f * (to - from)
        if (step > 0) v = from + Math.round((v - from) / step) * step
        v = Math.max(from, Math.min(to, v))
        root.moved(v)
        if (commit) root.committed(v)
    }

    // Double-click the readout to type an exact value (clamped to [from,to]).
    function _startEdit() {
        if (!root.enabled) return
        valueEdit.text = root.decimals > 0 ? root.value.toFixed(root.decimals)
                                           : Math.round(root.value).toString()
        valueEdit.visible = true
        valueEdit.forceActiveFocus()
        valueEdit.selectAll()
    }
    function _commitEdit() {
        var v = parseFloat(valueEdit.text)
        valueEdit.visible = false
        if (isNaN(v)) return
        if (root.step > 0) v = from + Math.round((v - from) / root.step) * root.step
        v = Math.max(from, Math.min(to, v))
        root.moved(v); root.committed(v)
    }

    Rectangle {                                   // track
        id: track
        anchors.left: parent.left
        anchors.right: root.showValue ? valueLbl.left : parent.right
        anchors.rightMargin: root.showValue ? sc.sp3 : 0
        anchors.verticalCenter: parent.verticalCenter
        height: 6; radius: 3
        color: pal.PANEL_ALT
        border.width: 1; border.color: pal.BORDER

        Rectangle {                               // filled portion
            height: parent.height; radius: 3
            width: knob.x + knob.width / 2
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0.0; color: pal.ACC }
                GradientStop { position: 1.0; color: pal.ACC_HOVER }
            }
        }
        Rectangle {                               // knob
            id: knob
            width: 14; height: 14; radius: 7
            y: (parent.height - height) / 2
            x: root.frac * (parent.width - width)
            color: root.enabled ? pal.ACC : pal.TXT_MUTED
            border.width: 2; border.color: pal.BG
            scale: (ma.pressed || hov.hovered) ? 1.15 : 1.0      // §19a thumb-grow
            Behavior on scale { NumberAnimation { duration: Theme.reducedMotion ? 0 : 120; easing.type: Easing.OutCubic } }
            Rectangle {                           // soft glow ring
                anchors.centerIn: parent
                width: parent.width + 6; height: width; radius: width / 2
                color: "transparent"
                border.width: 2
                border.color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b,
                                      (ma.pressed || hov.hovered) ? 0.35 : 0.0)
                Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
            }
        }
        HoverHandler { id: hov; enabled: root.enabled; cursorShape: Qt.PointingHandCursor }
        MouseArea {
            id: ma
            anchors.fill: parent
            anchors.margins: -7                   // generous grab target
            enabled: root.enabled
            cursorShape: Qt.PointingHandCursor
            onPressed: (m) => root._apply(m.x + anchors.margins, false)
            onPositionChanged: (m) => { if (pressed) root._apply(m.x + anchors.margins, false) }
            onReleased: (m) => root._apply(m.x + anchors.margins, true)
        }
    }

    Text {
        id: valueLbl
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        width: 46
        horizontalAlignment: Text.AlignRight
        visible: root.showValue && !valueEdit.visible
        text: (root.decimals > 0 ? root.value.toFixed(root.decimals)
                                 : Math.round(root.value).toString()) + root.suffix
        color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo"
        MouseArea {
            anchors.fill: parent
            anchors.margins: -4                   // easier double-click target
            enabled: root.enabled
            cursorShape: Qt.IBeamCursor
            onDoubleClicked: root._startEdit()
        }
    }

    TextInput {
        id: valueEdit
        visible: false
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        width: 46
        horizontalAlignment: Text.AlignRight
        color: pal.ACC; font.pixelSize: sc.textXs; font.family: "Menlo"
        selectByMouse: true; clip: true
        inputMethodHints: Qt.ImhFormattedNumbersOnly
        validator: DoubleValidator {
            bottom: root.from; top: root.to; decimals: Math.max(0, root.decimals)
        }
        onAccepted: root._commitEdit()                    // Enter
        onActiveFocusChanged: if (!activeFocus && valueEdit.visible) root._commitEdit()
        Keys.onEscapePressed: valueEdit.visible = false   // cancel, no commit
    }
}
