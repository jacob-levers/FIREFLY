import QtQuick

// The "living localisation field" behind the landing hero: two soft glow blooms
// (amber top-right, blue bottom-left) drawn on a Canvas (works under every
// backend), and a sparse field of drifting, twinkling single-molecule dots.
// All motion is gated on Theme.reducedMotion.
Item {
    id: backdrop
    clip: true

    Canvas {                                   // glow blooms
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d")
            ctx.clearRect(0, 0, width, height)
            function bloom(cx, cy, r, inner) {
                var g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r)
                g.addColorStop(0, inner)
                g.addColorStop(1, "transparent")
                ctx.fillStyle = g
                ctx.fillRect(0, 0, width, height)
            }
            var rr = Math.max(width, height)
            bloom(width * 0.86, height * 0.12, rr * 0.55, "rgba(246,166,35,0.11)")
            bloom(width * 0.08, height * 0.92, rr * 0.50, "rgba(88,166,255,0.07)")
        }
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
    }

    Repeater {                                 // drifting molecules
        model: 7
        delegate: Rectangle {
            id: dot
            required property int index
            readonly property var cols: ["#F6A623", "#27C0E8", "#4FE0A0"]
            width: 2 + (index % 3)
            height: width
            radius: width / 2
            color: cols[index % 3]
            opacity: 0.05
            x: backdrop.width * (0.12 + 0.76 * ((index * 0.197) % 1.0))
            property real baseY: backdrop.height * (0.10 + 0.80 * ((index * 0.331) % 1.0))
            y: baseY

            SequentialAnimation on opacity {
                running: !Theme.reducedMotion
                loops: Animation.Infinite
                PauseAnimation { duration: index * 350 }
                NumberAnimation { to: 0.55; duration: 1700; easing.type: Easing.InOutSine }
                NumberAnimation { to: 0.06; duration: 2300; easing.type: Easing.InOutSine }
            }
            SequentialAnimation on y {
                running: !Theme.reducedMotion
                loops: Animation.Infinite
                NumberAnimation { to: dot.baseY - 22; duration: 5200 + index * 600; easing.type: Easing.InOutSine }
                NumberAnimation { to: dot.baseY; duration: 5200 + index * 600; easing.type: Easing.InOutSine }
            }
        }
    }
}
