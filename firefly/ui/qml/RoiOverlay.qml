import QtQuick
import QtQuick.Layouts
import "components"

// Manual-polygon ROI editor (Phase 6c): a full-area modal over the input file's
// max-projection. Click to drop polygon vertices, "Close polygon" to commit one,
// Done to save (per-file, into RoiStore → params_builder roi_polygon). Pure QML
// (no native island) — draws over the image with a Canvas; bound to `Roi`.
Item {
    id: root
    anchors.fill: parent
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    visible: Roi.editing

    // opaque backdrop swallows clicks to the tab beneath
    Rectangle { anchors.fill: parent; color: pal.BG; opacity: 0.98
                MouseArea { anchors.fill: parent } }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: sc.sp6
        spacing: sc.sp4

        // toolbar
        RowLayout {
            Layout.fillWidth: true
            spacing: sc.sp4
            Icon { name: "move"; size: 16; color: pal.ACC }
            Text { text: "Edit ROI"; color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true }
            Text { text: Roi.fileName; color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                   font.family: "Menlo"; Layout.preferredWidth: 0; Layout.fillWidth: true
                   elide: Text.ElideMiddle }
            Badge { text: Roi.polygonCount + (Roi.polygonCount === 1 ? " polygon" : " polygons")
                    tone: Roi.polygonCount > 0 ? pal.ACC : pal.TXT_MUTED }
            Button { variant: "secondary"; text: "Close polygon"; icon: "check"
                     enabled: Roi.canClose; onClicked: Roi.closeDraft() }
            Button { variant: "secondary"; text: "Clear"; icon: "x"; onClicked: Roi.clearPolygons() }
            Button { variant: "secondary"; text: "Cancel"; onClicked: Roi.cancel() }
            Button { variant: "primary"; text: "Done"; icon: "check"; onClicked: Roi.commit() }
        }
        Text {
            Layout.fillWidth: true
            text: "Click to add polygon points · ‘Close polygon’ needs ≥3 points · draw several for multiple ROIs."
            color: pal.TXT_MUTED; font.pixelSize: sc.textXs
        }

        // image + drawing canvas
        Rectangle {
            Layout.fillWidth: true; Layout.fillHeight: true
            radius: sc.radiusMd; color: "#05070a"
            border.width: 1; border.color: pal.BORDER; clip: true

            Item {
                id: imgArea
                anchors.fill: parent
                anchors.margins: 2
                readonly property real scale: Roi.imageWidth > 0 ? bg.paintedWidth / Roi.imageWidth : 1
                readonly property real offX: (width - bg.paintedWidth) / 2
                readonly property real offY: (height - bg.paintedHeight) / 2
                function toImg(px, py) { return [(py - offY) / scale, (px - offX) / scale] }  // [y, x]
                function toDispX(x) { return offX + x * scale }
                function toDispY(y) { return offY + y * scale }

                Image {
                    id: bg
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                    smooth: false; cache: false; asynchronous: true
                    source: Roi.hasImage ? ("image://roibg/" + Roi.imageToken) : ""
                    onPaintedWidthChanged: canvas.requestPaint()
                }
                Text {
                    anchors.centerIn: parent
                    visible: !Roi.hasImage
                    text: "No image — pick an image input file to draw a polygon ROI."
                    color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                }

                Canvas {
                    id: canvas
                    anchors.fill: parent
                    onPaint: {
                        var ctx = getContext("2d"); ctx.reset(); ctx.clearRect(0, 0, width, height)
                        ctx.lineWidth = 1.5
                        // committed polygons (accent, semi-filled)
                        var polys = Roi.polygons
                        for (var p = 0; p < polys.length; ++p) {
                            var poly = polys[p]; if (!poly.length) continue
                            ctx.beginPath()
                            for (var i = 0; i < poly.length; ++i) {
                                var dx = imgArea.toDispX(poly[i][1]); var dy = imgArea.toDispY(poly[i][0])
                                if (i === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy)
                            }
                            ctx.closePath()
                            ctx.fillStyle = Qt.rgba(0.345, 0.651, 1.0, 0.15); ctx.fill()
                            ctx.strokeStyle = pal.ACC; ctx.stroke()
                        }
                        // draft polyline + vertex dots
                        var d = Roi.draftPoints
                        if (d.length) {
                            ctx.beginPath()
                            for (var j = 0; j < d.length; ++j) {
                                var ddx = imgArea.toDispX(d[j][1]); var ddy = imgArea.toDispY(d[j][0])
                                if (j === 0) ctx.moveTo(ddx, ddy); else ctx.lineTo(ddx, ddy)
                            }
                            ctx.strokeStyle = pal.WARN; ctx.setLineDash([5, 3]); ctx.stroke(); ctx.setLineDash([])
                            for (var k = 0; k < d.length; ++k) {
                                ctx.beginPath()
                                ctx.arc(imgArea.toDispX(d[k][1]), imgArea.toDispY(d[k][0]), 3, 0, 2 * Math.PI)
                                ctx.fillStyle = pal.WARN; ctx.fill()
                            }
                        }
                    }
                }
                MouseArea {
                    anchors.fill: parent
                    enabled: Roi.hasImage
                    cursorShape: Qt.CrossCursor
                    onClicked: (m) => {
                        var yx = imgArea.toImg(m.x, m.y)
                        if (yx[0] >= 0 && yx[1] >= 0 && yx[0] <= Roi.imageHeight && yx[1] <= Roi.imageWidth)
                            Roi.addVertex(yx[0], yx[1])
                    }
                }
            }
        }
    }

    Connections {
        target: Roi
        function onPolygonsChanged() { canvas.requestPaint() }
        function onDraftChanged() { canvas.requestPaint() }
        function onImageChanged() { canvas.requestPaint() }
    }
}
