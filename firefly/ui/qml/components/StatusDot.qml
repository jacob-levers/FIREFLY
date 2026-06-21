import QtQuick

// StatusDot — a liveness indicator: a solid dot with a soft expanding ring.
// Use inside "Live" / "running" badges. `pulsing` gates the loop (and it
// auto-stops under reduce-motion).
//
//   RowLayout {
//       StatusDot { tone: pal.SUCCESS; pulsing: Process.running }
//       Text { text: "Live"; color: pal.SUCCESS; ... }
//   }
Item {
    id: root
    property color tone: Theme.palette.SUCCESS
    property bool  pulsing: true

    implicitWidth: 8
    implicitHeight: 8

    Rectangle {                       // the dot
        anchors.centerIn: parent
        width: 6; height: 6; radius: 3
        color: root.tone
    }

    Rectangle {                       // the expanding ring
        id: ring
        anchors.centerIn: parent
        width: 6; height: 6; radius: width / 2
        color: "transparent"
        border.width: 1.5
        border.color: root.tone
        opacity: 0

        SequentialAnimation {
            running: root.pulsing && !Theme.reducedMotion
            loops: Animation.Infinite
            ParallelAnimation {
                NumberAnimation { target: ring; property: "opacity"; from: 0.55; to: 0
                                  duration: 1100; easing.type: Easing.OutCubic }
                NumberAnimation { target: ring; property: "scale";   from: 1; to: 2.4
                                  duration: 1100; easing.type: Easing.OutCubic }
            }
            PauseAnimation { duration: 500 }
        }
    }
}
