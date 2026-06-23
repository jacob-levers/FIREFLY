import QtQuick

// The backdrop behind the landing hero: two soft glow blooms (amber top-right,
// blue bottom-left) drawn on a Canvas (works under every backend). Static and
// motion-free.
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
}
