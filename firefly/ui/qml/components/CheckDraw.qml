import QtQuick

// CheckDraw — a check-mark that DRAWS its stroke on (success confirmations:
// run finished, file saved, QC pass). Crisper than a fade. Call play() or bind
// `on` true.
//
//   CheckDraw { on: Process.done; size: 18; tone: pal.SUCCESS }
Canvas {
    id: root
    property bool on: false
    property int size: 18
    property color tone: Theme.palette.SUCCESS
    property real drawP: 0
    implicitWidth: size; implicitHeight: size
    onPaint: {
        var ctx = getContext("2d"); ctx.reset();
        var w = width, h = height;
        // checkmark polyline (relative), drawn proportionally to drawP
        var pts = [[0.2,0.55],[0.42,0.75],[0.8,0.28]];
        ctx.strokeStyle = tone; ctx.lineWidth = Math.max(2, size*0.11);
        ctx.lineCap = "round"; ctx.lineJoin = "round";
        // total length split: seg1 then seg2
        var seg1 = 0.45, prog = drawP;
        ctx.beginPath(); ctx.moveTo(pts[0][0]*w, pts[0][1]*h);
        if (prog <= seg1) {
            var t = prog/seg1;
            ctx.lineTo((pts[0][0]+(pts[1][0]-pts[0][0])*t)*w, (pts[0][1]+(pts[1][1]-pts[0][1])*t)*h);
        } else {
            ctx.lineTo(pts[1][0]*w, pts[1][1]*h);
            var t2 = (prog-seg1)/(1-seg1);
            ctx.lineTo((pts[1][0]+(pts[2][0]-pts[1][0])*t2)*w, (pts[1][1]+(pts[2][1]-pts[1][1])*t2)*h);
        }
        ctx.stroke();
    }
    onDrawPChanged: requestPaint()
    NumberAnimation on drawP { id: anim; from: 0; to: 1; duration: 360; easing.type: Easing.OutCubic; running: false }
    function play() { drawP = 0; if (Theme.reducedMotion) { drawP = 1; } else anim.restart() }
    onOnChanged: if (on) play(); else { drawP = 0; requestPaint() }
    Component.onCompleted: if (on) play()
}
