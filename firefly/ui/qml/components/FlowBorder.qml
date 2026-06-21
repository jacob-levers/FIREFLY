import QtQuick

// FlowBorder — an animated accent gradient that travels around a card's border to
// signal it is ACTIVE / running (the live Analysis cockpit card, a processing
// tile). Subtle, slow (2.4 s loop). Place as the card's background; put content
// on top with a 1px inset panel fill so only the border shows.
//
//   FlowBorder { anchors.fill: parent; active: Process.running }
//   Rectangle { anchors.fill: parent; anchors.margins: 1; radius: sc.radiusLg-1; color: pal.PANEL }
Item {
    id: root
    property bool active: true
    property int radius: 6
    readonly property var pal: Theme.palette

    Rectangle {                       // static border when inactive
        anchors.fill: parent; radius: root.radius
        color: "transparent"; border.width: 1; border.color: pal.BORDER
        visible: !root.active || Theme.reducedMotion
    }
    Canvas {
        id: cv
        anchors.fill: parent
        visible: root.active && !Theme.reducedMotion
        property real phase: 0
        onPhaseChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d"); ctx.reset();
            var g = ctx.createLinearGradient(0,0,width,height);
            var p = phase;
            function stop(o,c){ ctx.globalAlpha=1; }
            g.addColorStop((0.0+p)%1, root.pal.BORDER);
            g.addColorStop((0.5+p)%1, root.pal.ACC);
            g.addColorStop((0.75+p)%1, root.pal.ACC_HOVER);
            g.addColorStop((1.0+p)%1 === 0 ? 0.999 : (1.0+p)%1, root.pal.BORDER);
            ctx.strokeStyle = g; ctx.lineWidth = 1.5;
            var r = root.radius, x=1, y=1, w=width-2, h=height-2;
            ctx.beginPath();
            ctx.moveTo(x+r,y);
            ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
            ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r);
            ctx.closePath(); ctx.stroke();
        }
        NumberAnimation on phase {
            running: root.active && !Theme.reducedMotion
            from: 0; to: 1; duration: 2400; loops: Animation.Infinite
        }
    }
}
