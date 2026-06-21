import QtQuick
import QtQuick.Effects   // Qt 6.5+  (for MultiEffect)

// FigureReveal — the right way to "animate" a matplotlib figure. The plot itself
// is a static PNG from the Python image provider, so we never animate its
// internals (bars, brackets, stars are baked pixels). Instead we animate the
// REVEAL: a Skeleton shimmer while it renders, then a blur-up — the sharp image
// dissolves in from blurred + slightly scaled + dim. Re-blurs and reveals again
// whenever `source` changes (a re-render) if you re-arm `ready` from status.
//
//   FigureReveal {
//       width: 520; height: 440
//       source: "image://figure/" + Workspace.figureToken
//       // ready auto-tracks the Image's load status:
//   }
//
// Qt 5 fallback: replace MultiEffect with QtGraphicalEffects' GaussianBlur, and
// animate its `radius` from ~16 → 0 (see note at bottom).
Item {
    id: root
    property url  source
    property real cornerRadius: 6
    // ready flips true once the image has decoded; that drives the blur-up.
    property bool ready: img.status === Image.Ready

    readonly property var pal: Theme.palette

    // ── loading skeleton (shimmer) ───────────────────────────────────────
    Skeleton {
        anchors.fill: parent
        radius: root.cornerRadius
        opacity: root.ready ? 0 : 1
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200 } }
    }

    // ── the real figure, fed through a blur effect ───────────────────────
    Image {
        id: img
        anchors.fill: parent
        source: root.source
        fillMode: Image.PreserveAspectFit
        cache: false
        asynchronous: true
        visible: false                     // shown via the effect below
    }

    MultiEffect {
        anchors.fill: img
        source: img
        opacity: root.ready ? 1 : 0
        scale:   root.ready ? 1 : 1.03
        blurEnabled: !Theme.reducedMotion
        blurMax: 48
        blur:    root.ready ? 0.0 : 1.0    // 0 = sharp, 1 = blurMax
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 320; easing.type: Easing.OutCubic } }
        Behavior on scale   { NumberAnimation { duration: Theme.reducedMotion ? 0 : 420; easing.type: Easing.OutCubic } }
        Behavior on blur    { NumberAnimation { duration: Theme.reducedMotion ? 0 : 420; easing.type: Easing.OutCubic } }
    }
}

// ── Qt 5 (QtGraphicalEffects) variant ───────────────────────────────────────
// import QtGraphicalEffects 1.15
// Image { id: img; visible: false; ... }
// GaussianBlur {
//     anchors.fill: img; source: img
//     opacity: root.ready ? 1 : 0
//     radius:  root.ready ? 0 : 16          // animate this
//     samples: 33
//     Behavior on opacity { NumberAnimation { duration: 320; easing.type: Easing.OutCubic } }
//     Behavior on radius  { NumberAnimation { duration: 420; easing.type: Easing.OutCubic } }
// }
