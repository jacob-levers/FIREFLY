import QtQuick

// Skeleton — a loading placeholder with a sweeping shimmer. Use while a figure
// renders, a file loads, or metrics are still computing. Give it the size/shape
// of the thing it stands in for; swap it out (fade) when real content arrives.
//
//   Skeleton { width: 280; height: 200; radius: sc.radiusMd; visible: !Figure.ready }
Rectangle {
    id: root
    radius: 5                          // callers set radius directly (Rectangle built-in)
    color: Theme.palette.PANEL_ALT
    clip: true

    Rectangle {                                   // sweeping highlight
        id: sweep
        width: parent.width * 0.45
        height: parent.height * 2
        rotation: 12
        anchors.verticalCenter: parent.verticalCenter
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "transparent" }
            GradientStop { position: 0.5; color: Qt.rgba(1, 1, 1, 0.05) }
            GradientStop { position: 1.0; color: "transparent" }
        }
        XAnimator on x {
            running: root.visible && !Theme.reducedMotion
            loops: Animation.Infinite
            from: -sweep.width
            to: root.width
            duration: 1300
            easing.type: Easing.InOutSine
        }
    }
}
