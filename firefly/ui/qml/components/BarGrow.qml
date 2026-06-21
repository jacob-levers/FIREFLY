import QtQuick

// BarGrow — a data bar that grows from its baseline when it first appears and
// animates between values thereafter. For Compare bar charts, MSD/JDD bars,
// mobile-fraction bars. Drives `height` from `value` (0..1 of `track`).
//
//   Column { Repeater { model: Compare.groups; delegate: BarGrow {
//       value: modelData.auc; track: 160; tone: modelData.color
//       width: 28 } } }
Rectangle {
    id: root
    property real value: 0           // 0..1
    property real track: 100         // full height in px at value == 1
    property color tone: Theme.palette.ACC
    property int  delay: 0           // stagger: pass index * 40

    width: 24
    height: 0
    radius: 2
    color: tone
    anchors.bottom: parent ? parent.bottom : undefined

    Component.onCompleted: grow.start()
    onValueChanged: height = Math.max(2, value * track)

    SequentialAnimation {
        id: grow
        PauseAnimation { duration: Theme.reducedMotion ? 0 : root.delay }
        NumberAnimation { target: root; property: "height"
                          from: 0; to: Math.max(2, root.value * root.track)
                          duration: Theme.reducedMotion ? 0 : 420; easing.type: Easing.OutCubic }
    }
    Behavior on height { NumberAnimation { duration: Theme.reducedMotion ? 0 : 300; easing.type: Easing.OutCubic } }
    Behavior on color  { ColorAnimation { duration: Theme.reducedMotion ? 0 : 130 } }
}
