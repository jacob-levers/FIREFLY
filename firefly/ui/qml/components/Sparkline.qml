import QtQuick

// Sparkline — a tiny line chart that DRAWS itself in (and can scroll as new
// samples arrive). For the resource-history strip, a per-run locs/frame trend,
// or any compact time series. Feed `points` (array of 0..1); call play() to
// redraw the draw-in.
//
//   Sparkline { width: 160; height: 36; points: Monitor.gpuHistory; tone: pal.SUCCESS }
Canvas {
    id: root
    property var points: []          // array of 0..1
    property color tone: Theme.palette.ACC
    property real drawP: 0              // 0..1 reveal
    onPointsChanged: requestPaint()
    onDrawPChanged: requestPaint()
    onPaint: {
        var ctx = getContext("2d"); ctx.reset();
        var n = points.length; if (n < 2) return;
        var pad = 2, w = width - pad*2, h = height - pad*2;
        ctx.lineWidth = 1.5; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.strokeStyle = tone;
        var shown = Math.max(1, Math.floor((n-1) * drawP));
        ctx.beginPath();
        for (var i = 0; i <= shown; i++) {
            var x = pad + (i/(n-1))*w;
            var y = pad + (1 - points[i])*h;
            i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }
        ctx.stroke();
        // soft fill under the drawn portion
        ctx.lineTo(pad + (shown/(n-1))*w, pad+h); ctx.lineTo(pad, pad+h); ctx.closePath();
        ctx.globalAlpha = 0.10; ctx.fillStyle = tone; ctx.fill();
    }
    NumberAnimation on drawP { id: anim; from: 0; to: 1; duration: 700; easing.type: Easing.OutCubic; running: false }
    function play() { if (Theme.reducedMotion) { drawP = 1 } else anim.restart() }
    Component.onCompleted: play()
}
