import QtQuick

// IndeterminateShimmer — a soft highlight band that sweeps across its parent.
// Drop it INSIDE the progress bar's filled Rectangle (which should be
// `clip: true`) to signal "working" while a run is in progress.
//
//   Rectangle {                       // the progress fill
//       width: track.width * progress
//       clip: true
//       gradient: /* accent → accent-hover */
//       IndeterminateShimmer { active: Process.running }
//   }
Item {
    id: root
    property bool active: true
    anchors.fill: parent
    clip: true

    Rectangle {
        id: band
        width: parent.width * 0.30
        height: parent.height
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "transparent" }
            GradientStop { position: 0.5; color: Qt.rgba(1, 1, 1, 0.18) }
            GradientStop { position: 1.0; color: "transparent" }
        }
        XAnimator on x {
            running: root.active && !Theme.reducedMotion
            loops: Animation.Infinite
            from: -band.width
            to: root.width
            duration: 1100
            easing.type: Easing.InOutSine
        }
    }
}
