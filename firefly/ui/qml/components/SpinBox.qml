import QtQuick

// Numeric field.  Text entry always; optional stepper arrows (`steppers`) that
// nudge by `step`, with hold-to-repeat and Up/Down keys.  Supports doubles
// (decimals), an int mode (decimals 0), a suffix (e.g. " %"), and a
// special-value label shown when the value sits at `from` (the "off"/"auto"/
// "all" cases). Emits committed(real).
Rectangle {
    id: root
    property real value: 0
    property real from: 0
    property real to: 100
    property real step: 1               // the stepper increment; also the key step
    property int decimals: 0
    property string suffix: ""
    property string special: ""         // shown instead of the number when value==from
    property int textAlign: TextInput.AlignLeft   // e.g. AlignHCenter for short values
    property bool steppers: false       // show − / + arrows either side of the number
    signal committed(real v)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property string display:
        (special !== "" && value <= from) ? special
        : (decimals > 0 ? value.toFixed(decimals) : Math.round(value).toString()) + suffix

    implicitHeight: 28
    implicitWidth: steppers ? 160 : 120
    radius: sc.radiusSm
    color: pal.PANEL_ALT
    border.width: 1
    border.color: input.activeFocus ? pal.ACC : pal.BORDER
    Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }

    function _clamp(v) { return Math.max(from, Math.min(to, v)) }
    function _fmt(v) { return decimals > 0 ? v.toFixed(decimals) : Math.round(v).toString() }

    // Emit the new value but DON'T assign root.value — the owner binds `value`
    // to its own source and writes back in onCommitted, exactly as Slider does.
    // Assigning here destroyed that binding, so a field that had been typed in
    // stopped following its source: type a detection threshold, then drag the
    // slider, and the number sat frozen at what you typed.
    function _commit(v) { root.committed(_clamp(v)) }

    // Step from an explicit base, so the keys can step from what is currently
    // TYPED rather than from the last committed value.  Returns the new value.
    function stepFrom(base, n) {
        var s = step > 0 ? step : 1
        var f = Math.pow(10, Math.max(0, decimals))
        // round on the decimals we display, or 0.45 + 0.01 commits as
        // 0.46000000000000002 and the field reads a value it never stepped to
        var v = _clamp(Math.round((base + n * s) * f) / f)
        root.committed(v)
        return v
    }
    function stepBy(n) { return stepFrom(root.value, n) }

    // Hold either arrow to repeat: 400 ms before the first repeat, then fast.
    Timer {
        id: repeater
        property int dir: 0
        interval: 400; repeat: true
        onTriggered: { interval = 70; root.stepBy(dir) }
    }

    Rectangle {                          // −
        id: stepDown
        visible: root.steppers
        width: root.steppers ? 26 : 0
        anchors { left: parent.left; top: parent.top; bottom: parent.bottom; margins: 1 }
        radius: root.radius - 1
        color: downArea.pressed
               ? Qt.rgba(root.pal.ACC.r, root.pal.ACC.g, root.pal.ACC.b, 0.22)
               : downArea.containsMouse
               ? Qt.rgba(root.pal.TXT.r, root.pal.TXT.g, root.pal.TXT.b, 0.08)
               : "transparent"
        Text { anchors.centerIn: parent; text: "−"; font.bold: true
               font.pixelSize: root.sc.textSm
               color: root.value <= root.from ? root.pal.TXT_MUTED : root.pal.TXT }
        MouseArea {
            id: downArea
            anchors.fill: parent; hoverEnabled: true
            onPressed: { root.stepBy(-1); repeater.dir = -1
                         repeater.interval = 400; repeater.restart() }
            onReleased: repeater.stop()
            onCanceled: repeater.stop()
        }
    }

    Rectangle {                          // +
        id: stepUp
        visible: root.steppers
        width: root.steppers ? 26 : 0
        anchors { right: parent.right; top: parent.top; bottom: parent.bottom; margins: 1 }
        radius: root.radius - 1
        color: upArea.pressed
               ? Qt.rgba(root.pal.ACC.r, root.pal.ACC.g, root.pal.ACC.b, 0.22)
               : upArea.containsMouse
               ? Qt.rgba(root.pal.TXT.r, root.pal.TXT.g, root.pal.TXT.b, 0.08)
               : "transparent"
        Text { anchors.centerIn: parent; text: "+"; font.bold: true
               font.pixelSize: root.sc.textSm
               color: root.value >= root.to ? root.pal.TXT_MUTED : root.pal.TXT }
        MouseArea {
            id: upArea
            anchors.fill: parent; hoverEnabled: true
            onPressed: { root.stepBy(1); repeater.dir = 1
                         repeater.interval = 400; repeater.restart() }
            onReleased: repeater.stop()
            onCanceled: repeater.stop()
        }
    }

    TextInput {
        id: input
        anchors.fill: parent
        anchors.leftMargin: root.steppers ? stepDown.width + 2 : sc.sp3
        anchors.rightMargin: root.steppers ? stepUp.width + 2 : sc.sp3
        horizontalAlignment: root.steppers ? TextInput.AlignHCenter : root.textAlign
        verticalAlignment: TextInput.AlignVCenter
        color: pal.TXT
        font.pixelSize: sc.textSm
        selectByMouse: true
        clip: true
        // show the formatted display unless the user is actively editing
        text: activeFocus ? text : root.display
        onActiveFocusChanged: if (activeFocus) {
            text = (root.special !== "" && root.value <= root.from)
                   ? "" : root._fmt(root.value)
            selectAll()
        }
        onEditingFinished: {
            var v = parseFloat(text)
            if (!isNaN(v)) root._commit(v)
            else root.committed(root.value)   // revert display
            focus = false
        }
        // Step from what is typed, not from the last commit, so ↑ after typing
        // "0.4" goes to 0.41 rather than back to the old value + one step.
        Keys.onUpPressed: (event) => {
            if (!root.steppers) { event.accepted = false; return }
            var v = parseFloat(text)
            text = root._fmt(root.stepFrom(isNaN(v) ? root.value : v, 1))
        }
        Keys.onDownPressed: (event) => {
            if (!root.steppers) { event.accepted = false; return }
            var v = parseFloat(text)
            text = root._fmt(root.stepFrom(isNaN(v) ? root.value : v, -1))
        }
    }
}
