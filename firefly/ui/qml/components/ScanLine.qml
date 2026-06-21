import QtQuick

// ScanLine — a sweeping accent line that travels down (and back) over the
// detection preview while a run is processing — reinforces "the pipeline is
// reading this frame". Place over the preview Image; gate `active`.
//
//   Item { Image { ... }  ScanLine { anchors.fill: parent; active: Process.running } }
Item {
    id: root
    property bool active: true
    clip: true
    Rectangle {
        width: parent.width; height: 2
        y: 0
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "transparent" }
            GradientStop { position: 0.5; color: Theme.palette.ACC }
            GradientStop { position: 1.0; color: "transparent" }
        }
        // fade out when the sweep is gated off (idle / reduce-motion) so the
        // static line doesn't linger as a fake "still loading" bar
        opacity: (root.active && !Theme.reducedMotion) ? 0.85 : 0
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 300; easing.type: Easing.OutCubic } }
        SequentialAnimation on y {
            running: root.active && !Theme.reducedMotion
            loops: Animation.Infinite
            NumberAnimation { from: 0; to: root.height; duration: Theme.reducedMotion ? 0 : 1400; easing.type: Easing.InOutSine }
            NumberAnimation { from: root.height; to: 0; duration: Theme.reducedMotion ? 0 : 1400; easing.type: Easing.InOutSine }
        }
    }
}
