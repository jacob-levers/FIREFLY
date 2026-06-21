import QtQuick

// A dashed rounded-rectangle border — QML's Rectangle only does solid borders,
// so the drop zone / dashed buttons draw their outline here.
Canvas {
    id: root
    property color stroke: Theme.palette.BORDER_HI
    property real radius: 8
    property real dash: 4
    property real gap: 3
    property real lineWidth: 1

    onStrokeChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()
    Component.onCompleted: requestPaint()

    onPaint: {
        var ctx = getContext("2d");
        ctx.reset();
        ctx.strokeStyle = root.stroke;
        ctx.lineWidth = root.lineWidth;
        ctx.setLineDash([root.dash, root.gap]);
        var x = 0.5, y = 0.5, w = width - 1, h = height - 1, r = root.radius;
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
        ctx.stroke();
    }
}
