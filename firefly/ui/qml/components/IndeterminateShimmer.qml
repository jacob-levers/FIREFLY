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

    // Sweep position, 0 → 1.  The band's x is BOUND to it, so every frame the
    // sweep spans the bar's CURRENT width.  An XAnimator from -band.width to
    // root.width read those once, when it started — and a shimmer usually
    // starts on a bar that isn't at its final width yet: the update bar is
    // laid out at its status text's width for a frame after it appears, and a
    // progress fill starts near-empty and grows.  The band then looped over
    // that starting width forever (the left half of the update bar; the first
    // few percent of the Process / HYPER-FLY fills).
    property real phase: 0

    Rectangle {
        id: band
        visible: root.active
        width: parent.width * 0.30
        height: parent.height
        // reduced motion: a still band, centred, still says "working"
        x: Theme.reducedMotion ? (root.width - width) / 2
                               : -width + root.phase * (root.width + width)
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "transparent" }
            GradientStop { position: 0.5; color: Qt.rgba(1, 1, 1, 0.18) }
            GradientStop { position: 1.0; color: "transparent" }
        }
    }

    // LINEAR, deliberately.  Ease-in-out is for a motion that starts and
    // stops once; on an infinite loop it made the band crawl at the edges
    // — where it is mostly clipped — and race across the middle, a 16x
    // speed swing that reads as the animation stalling and restarting each
    // cycle.  Constant speed sweeps smoothly and keeps the band at least
    // half visible for 77% of the cycle rather than 56%.
    NumberAnimation on phase {
        running: root.active && !Theme.reducedMotion
        loops: Animation.Infinite
        from: 0
        to: 1
        duration: 1100
        easing.type: Easing.Linear
    }
}
