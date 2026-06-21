import QtQuick

// ProgressRing — a circular determinate progress indicator for per-file batch
// tiles, overall batch progress, or any 0..1 value where a bar doesn't fit.
// Sweeps the accent arc clockwise; the centre shows the percent (mono).
//
//   ProgressRing { value: Batch.fileProgress; size: 52 }
Item {
    id: root
    property real value: 0          // 0..1
    property int  size: 52
    property int  thickness: 4
    property color tone: Theme.palette.ACC
    readonly property var pal: Theme.palette

    implicitWidth: size; implicitHeight: size

    Canvas {
        id: cv
        anchors.fill: parent
        property real shown: 0
        onShownChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d"); ctx.reset();
            var cx = width/2, cy = height/2, r = (Math.min(width,height) - root.thickness)/2;
            ctx.lineWidth = root.thickness; ctx.lineCap = "round";
            ctx.strokeStyle = root.pal.PANEL_ALT;                 // track
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.stroke();
            ctx.strokeStyle = root.tone;                          // value arc
            ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI/2, -Math.PI/2 + Math.PI*2*shown); ctx.stroke();
        }
        Behavior on shown { NumberAnimation { duration: Theme.reducedMotion ? 0 : 300; easing.type: Easing.OutCubic } }
    }
    Binding { target: cv; property: "shown"; value: root.value }

    Text {
        anchors.centerIn: parent
        text: Math.round(root.value*100) + "%"
        color: pal.TXT; font.pixelSize: root.size*0.26; font.bold: true; font.family: "Menlo"
    }
}
