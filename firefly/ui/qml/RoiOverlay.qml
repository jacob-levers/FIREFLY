import QtQuick
import QtQuick.Layouts
import QtQuick.Controls as QQC
import "components"

// Preview & ROI viewer (Phase 6c → batch redesign): a centered modal card over
// the input file. Left = the stage (max projection OR a scrubbed raw frame) with
// the live threshold-mask overlay, a one-shot reveal scan line, the polygon ROI,
// and a Visualise-style transport scrubber. Right = the full ROI control panel
// mirroring the sidebar ROI menu (mode / auto method / threshold / mask mode /
// background σ). The sidebar holds the DEFAULT for all files; this viewer sets a
// PER-FILE override (saved on "Save ROI"). Bound to `Roi`.
Item {
    id: root
    anchors.fill: parent
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    visible: Roi.editing

    readonly property bool isPoly:   Roi.roiMode === "Manual polygon"
    readonly property bool isAuto:   Roi.roiMode === "Auto threshold"
    readonly property bool isManual: Roi.roiMode === "Manual threshold"
    readonly property bool isThresh: isAuto || isManual
    readonly property bool isSister: Roi.roiMode === "Sister TIFF"

    // Which screen is showing.  See the panel comments in the controls column.
    readonly property bool isDetect:   Roi.panel === "detect"
    readonly property bool isRoiPanel: !isDetect

    // small uppercase section label used throughout the control panel
    component PanelLabel: Text {
        color: Theme.palette.TXT_MUTED; font.pixelSize: 10
        font.bold: true; font.letterSpacing: 0.8
    }

    // entrance tween + a single reveal sweep of the scan line
    property bool shown: false
    property bool scanning: true
    Component.onCompleted: { try { Embed.setModalOpen(true) } catch (e) {} ; shown = true }
    Component.onDestruction: {
        try { Embed.setModalOpen(false) } catch (e) {}
        try { Batch.notifyRoiChanged() } catch (e) {}   // refresh the series ROI badge
    }
    Timer { running: true; interval: 2800; onTriggered: root.scanning = false }
    Timer { id: maskDebounce; interval: 160; onTriggered: Roi.refreshMask() }
    Timer { id: spotsDebounce; interval: 200; onTriggered: Roi.refreshSpots() }
    Connections {
        target: Roi
        function onSpotsChanged() { if (!Roi.spotsStale) spotsDebounce.stop() }
        function onPreviewInvalidated() {
            if (Roi.editing && Roi.detectEnabled) spotsDebounce.restart()
        }
    }

    function saveRoi() {
        if (root.isPoly && Roi.canClose) Roi.closeDraft()
        // Run-scoped: this is a FINISHED run, so there is no sidebar override to
        // write — hand the polygons straight to the post-process worker instead.
        // Postproc.start re-checks that the region only shrinks the run's own
        // ROI and refuses otherwise, because a run only stores the
        // localisations its ROI kept.
        if (Roi.runScoped) {
            var runDir = Roi.runDir
            var polys = Roi.runPolygons()
            var verdict = Postproc.canApply(runDir, polys)
            if (!verdict.ok) { roiBlocked.text = verdict.reason; roiBlocked.open(); return }
            Roi.closeRun()
            Postproc.start(runDir, polys)
            return
        }
        // Multiple-ROI handling is chosen inline via the "Analyse each ROI
        // separately" toggle, so saving just commits.
        Roi.commit()
    }

    // Refuses an ROI that reaches outside what the source run kept, rather than
    // producing a result that silently covers only the overlap.
    Modal {
        id: roiBlocked
        property alias text: roiBlockedText.text
        title: "That region can't be applied"
        Text {
            id: roiBlockedText
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            color: pal.TXT_MUTED
            font.pixelSize: sc.textSm
            lineHeight: 1.35
        }
        RowLayout {
            Layout.fillWidth: true
            Layout.topMargin: sc.sp2
            Item { Layout.fillWidth: true }
            Button { variant: "primary"; text: "Got it"; onClicked: roiBlocked.close() }
        }
    }

    Rectangle {
        anchors.fill: parent
        color: "#000000"
        opacity: root.shown ? 0.6 : 0
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200 } }
        MouseArea { anchors.fill: parent; onClicked: Roi.cancel() }
    }

    Card {
        id: vcard
        anchors.centerIn: parent
        width: Math.min(940, parent.width - sc.sp10 * 2)
        implicitHeight: vcol.implicitHeight
        raised: true
        opacity: root.shown ? 1 : 0
        scale:   root.shown ? 1 : 0.97
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
        Behavior on scale   { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
        transform: Translate {
            y: root.shown ? 0 : 10
            Behavior on y { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
        }
        MouseArea { anchors.fill: parent }

        ColumnLayout {
            id: vcol
            width: parent.width
            spacing: 0

            // ── header ──────────────────────────────────────────────────
            RowLayout {
                Layout.fillWidth: true
                Layout.margins: sc.sp4
                spacing: sc.sp3
                Icon { name: root.isDetect ? "sliders-horizontal" : "scan-search"
                       size: 15; color: pal.ACC }
                Text { text: root.isDetect ? "Detection threshold" : "Preview & ROI"
                       color: pal.TXT; font.pixelSize: sc.textMd; font.bold: true }
                Text { text: Roi.fileName; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                       font.family: "Menlo"; elide: Text.ElideMiddle
                       Layout.fillWidth: true; Layout.preferredWidth: 0 }
                IconButton { icon: "x"; tip: "Close"; onClicked: Roi.cancel() }
            }
            Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }

            // ── body: stage + controls ──────────────────────────────────
            RowLayout {
                Layout.fillWidth: true
                spacing: 0

                // left — stage + scrubber (scrubber sits BELOW the image)
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: sc.sp2

                    Rectangle {
                    id: stage
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 340
                    Layout.preferredHeight: Math.max(360, Math.min(540, root.height - 220))
                    color: pal.WELL
                    clip: true

                    Item {
                        id: imgArea
                        anchors.fill: parent
                        // A border of empty stage kept AROUND the fitted image so
                        // a polygon can be drawn slightly off every edge (the
                        // shape is clipped back to the image on close).  The
                        // image + all overlays share this inset, so they stay
                        // aligned; offX/offY derive from the painted width, so
                        // the coordinate transform is unaffected by the pad.
                        readonly property real drawPad: root.isPoly
                            ? Math.round(Math.min(width, height) * 0.07) : 0
                        readonly property real sscale: Roi.imageWidth > 0 ? bg.paintedWidth / Roi.imageWidth : 1
                        readonly property real offX: (width - bg.paintedWidth) / 2
                        readonly property real offY: (height - bg.paintedHeight) / 2
                        function toImg(px, py) { return [(py - offY) / sscale, (px - offX) / sscale] }
                        function toDispX(x) { return offX + x * sscale }
                        function toDispY(y) { return offY + y * sscale }

                        Image {
                            id: bg
                            anchors.fill: parent
                            anchors.margins: imgArea.drawPad
                            fillMode: Image.PreserveAspectFit
                            smooth: false; cache: false; asynchronous: true
                            source: Roi.hasImage ? ("image://roibg/" + Roi.imageToken) : ""
                            onPaintedWidthChanged: canvas.requestPaint()
                        }

                        // ROI mask overlay (auto / manual threshold OR sister
                        // TIFF) — constant across frames, so scrubbing shows
                        // which particles the region keeps
                        Image {
                            anchors.fill: parent
                            anchors.margins: imgArea.drawPad
                            fillMode: Image.PreserveAspectFit
                            smooth: false; cache: false; asynchronous: true
                            visible: (root.isThresh || root.isSister) && Roi.hasMask && !(Roi.detectEnabled && root.isThresh)
                            opacity: visible ? 1 : 0
                            Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                            source: Roi.hasMask ? ("image://roimask/" + Roi.maskToken) : ""
                        }

                        // detected-spot overlay (green circles at the current minmass)
                        Image {
                            anchors.fill: parent
                            anchors.margins: imgArea.drawPad
                            fillMode: Image.PreserveAspectFit
                            smooth: true; cache: false; asynchronous: true
                            visible: Roi.detectEnabled && Roi.hasSpots
                            opacity: visible ? 1 : 0
                            Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                            source: Roi.hasSpots ? ("image://roispots/" + Roi.spotToken) : ""
                        }

                        // Painted-region preview. Same PreserveAspectFit box and
                        // drawPad inset as the background, so the mask and the
                        // image cannot drift apart.
                        Image {
                            anchors.fill: parent
                            anchors.margins: imgArea.drawPad
                            fillMode: Image.PreserveAspectFit
                            smooth: false; cache: false; asynchronous: false
                            visible: Roi.brushActive && Roi.hasBrushPreview
                            source: Roi.hasBrushPreview ? ("image://roibrush/" + Roi.brushToken) : ""
                        }

                        ScanLine { anchors.fill: parent; anchors.margins: imgArea.drawPad; active: root.scanning }

                        Text {
                            anchors.centerIn: parent
                            visible: !Roi.hasImage
                            width: parent.width - sc.sp8 * 2
                            horizontalAlignment: Text.AlignHCenter; wrapMode: Text.WordWrap
                            text: "No preview for this file — pick an image recording to draw a region."
                            color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                        }

                        Canvas {
                            id: canvas
                            anchors.fill: parent
                            visible: root.isPoly
                            onPaint: {
                                var ctx = getContext("2d"); ctx.reset(); ctx.clearRect(0, 0, width, height)
                                ctx.lineWidth = 1.5
                                var polys = Roi.polygons
                                for (var p = 0; p < polys.length; ++p) {
                                    var poly = polys[p]; if (!poly.length) continue
                                    ctx.beginPath()
                                    for (var i = 0; i < poly.length; ++i) {
                                        var dx = imgArea.toDispX(poly[i][1]); var dy = imgArea.toDispY(poly[i][0])
                                        if (i === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy)
                                    }
                                    ctx.closePath()
                                    ctx.fillStyle = Qt.rgba(0.337, 0.827, 0.392, 0.16); ctx.fill()
                                    ctx.strokeStyle = pal.SUCCESS; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([])
                                }
                                var d = Roi.draftPoints
                                if (d.length) {
                                    ctx.beginPath()
                                    for (var j = 0; j < d.length; ++j) {
                                        var ddx = imgArea.toDispX(d[j][1]); var ddy = imgArea.toDispY(d[j][0])
                                        if (j === 0) ctx.moveTo(ddx, ddy); else ctx.lineTo(ddx, ddy)
                                    }
                                    ctx.strokeStyle = pal.SUCCESS; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([])
                                    for (var k = 0; k < d.length; ++k) {
                                        ctx.beginPath()
                                        ctx.arc(imgArea.toDispX(d[k][1]), imgArea.toDispY(d[k][0]), 3, 0, 2 * Math.PI)
                                        ctx.fillStyle = pal.SUCCESS; ctx.fill()
                                    }
                                }
                            }
                        }

                        // Brush-size cursor: what you are about to paint, at the
                        // same scale as the image, so the size slider is honest.
                        Canvas {
                            id: brushCursor
                            anchors.fill: parent
                            visible: Roi.brushActive && drawArea.containsMouse
                            onPaint: {
                                var ctx = getContext("2d"); ctx.reset()
                                ctx.clearRect(0, 0, width, height)
                                var rpx = Roi.brushRadius * imgArea.sscale
                                ctx.beginPath()
                                ctx.arc(drawArea.mouseX, drawArea.mouseY, rpx, 0, 2 * Math.PI)
                                ctx.strokeStyle = (Roi.tool === "eraser") ? pal.DANGER : pal.SUCCESS
                                ctx.lineWidth = 1.5; ctx.stroke()
                            }
                        }

                        MouseArea {
                            id: drawArea
                            anchors.fill: parent
                            enabled: Roi.hasImage && (root.isPoly || Roi.detectEnabled)
                            cursorShape: root.isPoly ? Qt.CrossCursor : Qt.ArrowCursor
                            hoverEnabled: Roi.brushActive
                            property real lastY: NaN
                            property real lastX: NaN
                            // Brush strokes: press -> snapshot for undo, drag ->
                            // paint the segment since the last point (so a fast
                            // drag is continuous, not dotted), release -> retrace
                            // the mask into polygons.
                            onPressed: (m) => {
                                if (!Roi.brushActive) return
                                var yx = imgArea.toImg(m.x, m.y)
                                Roi.beginStroke()
                                Roi.paintAt(yx[0], yx[1])
                                lastY = yx[0]; lastX = yx[1]
                            }
                            onPositionChanged: (m) => {
                                brushCursor.requestPaint()
                                if (!Roi.brushActive || !(m.buttons & Qt.LeftButton)) return
                                var yx = imgArea.toImg(m.x, m.y)
                                if (isNaN(lastY)) Roi.paintAt(yx[0], yx[1])
                                else              Roi.paintAt(yx[0], yx[1], lastY, lastX)
                                lastY = yx[0]; lastX = yx[1]
                            }
                            onReleased: {
                                if (!Roi.brushActive) return
                                lastY = NaN; lastX = NaN
                                Roi.endStroke()
                            }
                            // Vertices MAY sit outside the image (in the margin
                            // around it) so a region can comfortably enclose
                            // samples pressed against an edge; closeDraft() clips
                            // the finished shape back to the image rectangle.
                            onClicked: (m) => {
                                var yx = imgArea.toImg(m.x, m.y)
                                if (Roi.brushActive)
                                    return              // press/drag already painted
                                if (Roi.detectEnabled && (!root.isPoly || (m.modifiers & Qt.ControlModifier)))
                                    Roi.inspectSpot(yx[0], yx[1])
                                else
                                    Roi.addVertex(yx[0], yx[1])
                            }
                        }

                        // ── editable vertices ───────────────────────────
                        // The controller has had moveVertex / deleteVertex /
                        // deletePolygon since the editor was written, but no QML
                        // ever called them: a committed shape could only be
                        // cleared and redrawn.  These handles are what make an
                        // EXISTING region editable.
                        //
                        // Declared AFTER the drawing MouseArea so they sit above
                        // it and take the press first — a click that misses a
                        // handle still falls through and adds a vertex.
                        //
                        // Position is bound to Roi.polygons rather than moved
                        // directly: the controller stays the single source of
                        // truth, so a handle can never drift from the outline
                        // the canvas paints from the same list.
                        Repeater {
                            // Hidden while painting: a traced brush outline can
                            // carry hundreds of vertices, and dragging one of
                            // them is not how you edit a painted shape.
                            model: (root.isPoly && Roi.hasImage && !Roi.brushActive)
                                   ? Roi.polygonCount : 0
                            delegate: Item {
                                id: polyHandles
                                readonly property int polyIdx: index
                                anchors.fill: parent

                                Repeater {
                                    model: {
                                        var ps = Roi.polygons
                                        return (polyHandles.polyIdx < ps.length)
                                            ? ps[polyHandles.polyIdx].length : 0
                                    }
                                    delegate: Rectangle {
                                        id: handle
                                        readonly property int vertIdx: index
                                        readonly property var pt: {
                                            var ps = Roi.polygons
                                            if (polyHandles.polyIdx >= ps.length) return [0, 0]
                                            var poly = ps[polyHandles.polyIdx]
                                            return (vertIdx < poly.length) ? poly[vertIdx] : [0, 0]
                                        }
                                        width: 11; height: 11; radius: width / 2
                                        x: imgArea.toDispX(pt[1]) - width / 2
                                        y: imgArea.toDispY(pt[0]) - height / 2
                                        color: grab.containsMouse || grab.dragging
                                               ? pal.SUCCESS : Qt.rgba(0, 0, 0, 0.55)
                                        border.width: 1.5
                                        border.color: pal.SUCCESS

                                        MouseArea {
                                            id: grab
                                            anchors.fill: parent
                                            anchors.margins: -5     // forgiving grab target
                                            hoverEnabled: true
                                            acceptedButtons: Qt.LeftButton | Qt.RightButton
                                            cursorShape: Qt.SizeAllCursor
                                            property bool dragging: false
                                            onPressed: (m) => {
                                                if (m.button === Qt.LeftButton) dragging = true
                                            }
                                            onReleased: dragging = false
                                            onCanceled: dragging = false
                                            onPositionChanged: (m) => {
                                                if (!dragging) return
                                                var q = mapToItem(imgArea, m.x, m.y)
                                                var yx = imgArea.toImg(q.x, q.y)
                                                Roi.moveVertex(polyHandles.polyIdx,
                                                               handle.vertIdx, yx[0], yx[1])
                                            }
                                            // Right-click removes the vertex; the
                                            // controller drops the whole polygon if
                                            // that would leave fewer than three.
                                            onClicked: (m) => {
                                                if (m.button === Qt.RightButton)
                                                    Roi.deleteVertex(polyHandles.polyIdx,
                                                                     handle.vertIdx)
                                            }
                                        }
                                    }
                                }

                                // delete the whole region — anchored at its first vertex
                                Rectangle {
                                    readonly property var head: {
                                        var ps = Roi.polygons
                                        if (polyHandles.polyIdx >= ps.length) return [0, 0]
                                        var poly = ps[polyHandles.polyIdx]
                                        return poly.length ? poly[0] : [0, 0]
                                    }
                                    visible: Roi.polygonCount > 0
                                    width: 18; height: 18; radius: width / 2
                                    x: imgArea.toDispX(head[1]) + 10
                                    y: imgArea.toDispY(head[0]) - 24
                                    color: del.containsMouse ? pal.DANGER : Qt.rgba(0, 0, 0, 0.65)
                                    border.width: 1; border.color: pal.DANGER
                                    Text {
                                        anchors.centerIn: parent
                                        text: "\u00d7"; color: "#ffffff"
                                        font.pixelSize: 12; font.bold: true
                                    }
                                    MouseArea {
                                        id: del
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: Roi.deletePolygon(polyHandles.polyIdx)
                                    }                                }
                            }
                        }

                        // top-left readout (mode / coverage)
                        Rectangle {
                            anchors { left: parent.left; top: parent.top; margins: sc.sp2 }
                            visible: Roi.hasImage
                            radius: sc.radiusXs; color: Qt.rgba(0, 0, 0, 0.6)
                            width: cornlbl.implicitWidth + sc.sp3 * 2
                            height: cornlbl.implicitHeight + sc.sp1 * 2
                            Text { id: cornlbl; anchors.centerIn: parent
                                   text: Roi.frameLabel
                                         + ((root.isThresh || root.isSister) && Roi.hasMask
                                            ? "  ·  ROI " + (Roi.maskFraction * 100).toFixed(1) + "%" : "")
                                   color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo" }
                        }
                    }

                    }   // ← stage Rectangle

                    // ── transport scrubber — BELOW the image, raw view only ──
                    Rectangle {
                        id: transport
                        visible: Roi.viewMode === "raw" && Roi.nFrames > 1
                        Layout.fillWidth: true
                        Layout.preferredHeight: 40
                        radius: 12
                        color: Qt.rgba(0.05, 0.07, 0.10, 0.86)
                        border.width: 1; border.color: pal.BORDER_HI
                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: sc.sp4; anchors.rightMargin: sc.sp4
                            spacing: sc.sp3
                            Icon { name: "image"; size: 15; color: pal.TXT_MUTED }
                            Slider {                          // scrubber
                                Layout.fillWidth: true
                                showValue: false
                                from: 0; to: Math.max(1, Roi.nFrames - 1); step: 1; decimals: 0
                                value: Roi.frameIndex
                                // frame renders live; the (heavy) mask + detection
                                // recomputes debounce so they update per frame
                                onMoved: (v) => { Roi.setFrame(v); maskDebounce.restart()
                                                  if (Roi.detectEnabled) spotsDebounce.restart() }
                            }
                            Text {
                                text: (Roi.frameIndex + 1) + " / " + Roi.nFrames
                                color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                            }
                        }
                    }
                }

                // right — control panel (mirrors the sidebar ROI menu)
                Rectangle {
                    Layout.preferredWidth: 276
                    Layout.fillHeight: true
                    color: "transparent"
                    Rectangle { anchors.left: parent.left; anchors.top: parent.top
                                anchors.bottom: parent.bottom; width: 1; color: pal.BORDER }
                    Flickable {
                        id: controlsScroll
                        anchors.fill: parent
                        anchors.margins: sc.sp4
                        contentWidth: width
                        contentHeight: controlsColumn.implicitHeight
                        clip: true
                        flickableDirection: Flickable.VerticalFlick
                        QQC.ScrollBar.vertical: QQC.ScrollBar {}
                        ColumnLayout {
                            id: controlsColumn
                            width: controlsScroll.width - 8
                            spacing: sc.sp3

                        // view toggle
                        ColumnLayout {
                            visible: Roi.nFrames > 1 || Roi.hasGreenImage
                            Layout.fillWidth: true; spacing: sc.sp2
                            PanelLabel { text: "VIEW" }
                            Segmented {
                                Layout.fillWidth: true
                                // "Green" appears only when a companion image
                                // sits beside this recording
                                options: {
                                    var o = [{ v: "proj", t: "Max proj" }]
                                    if (Roi.nFrames > 1) o.push({ v: "raw", t: "Raw frames" })
                                    if (Roi.hasGreenImage) o.push({ v: "green", t: "Green" })
                                    return o
                                }
                                value: Roi.viewMode
                                onPicked: (v) => Roi.setViewMode(v)
                            }
                        }

                        // colour map
                        PanelLabel { text: "COLOUR"; Layout.topMargin: sc.sp1 }
                        Select {
                            Layout.fillWidth: true
                            model: Roi.cmaps
                            currentIndex: Math.max(0, Roi.cmaps.indexOf(Roi.cmap))
                            onPicked: (t) => Roi.cmap = t
                        }

                        // ── ROI: WHERE to analyse ───────────────────────────────────
                        // Two screens of one viewer, not one screen doing two jobs:
                        // `Roi.editFile` opens this panel and `Roi.editDetection` the
                        // one below.  They were a single scrolling column, so every
                        // visit scrolled past the other job's controls to reach its own.
                        ColumnLayout {
                            visible: root.isRoiPanel
                            Layout.fillWidth: true; spacing: sc.sp3

                            PanelLabel { text: "ROI MODE"; Layout.topMargin: sc.sp1 }
                            Select {
                                Layout.fillWidth: true
                                model: Roi.roiModes
                                currentIndex: Math.max(0, Roi.roiModes.indexOf(Roi.roiMode))
                                onPicked: (t) => Roi.roiMode = t
                            }
                            Text {
                                Layout.fillWidth: true; wrapMode: Text.WordWrap
                                color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3
                                text: root.isPoly
                                      ? "Click the image to trace a region (Close shape for more). Saved for THIS file only."
                                      : root.isThresh
                                      ? "Green mask = the threshold ROI. Scrub raw frames to see which particles it keeps. Saved for THIS file only."
                                      : root.isSister
                                      ? (Roi.hasMask
                                         ? "Green mask = the sister-image ROI FIREFLY will keep.  " + Roi.sisterStatus
                                         : (Roi.sisterStatus || "Looking for a companion ROI image…")
                                           + "  The whole frame is analysed unless a sister image is found.")
                                      : Roi.roiMode === "None"
                                      ? "Analyse the whole frame for this file (no region)."
                                      : "ROI loaded from a companion file for this file."
                            }

                            // auto-threshold method
                            ColumnLayout {
                                visible: root.isAuto
                                Layout.fillWidth: true; spacing: sc.sp2; Layout.topMargin: sc.sp1
                                PanelLabel { text: "AUTO METHOD" }
                                Select {
                                    Layout.fillWidth: true
                                    model: Roi.autoMethods
                                    currentIndex: Math.max(0, Roi.autoMethods.indexOf(Roi.autoMethod))
                                    onPicked: (t) => Roi.autoMethod = t
                                }
                            }

                            // manual-threshold slider
                            ColumnLayout {
                                visible: root.isManual
                                Layout.fillWidth: true; spacing: sc.sp2; Layout.topMargin: sc.sp1
                                PanelLabel { text: "THRESHOLD" }
                                Slider {
                                    Layout.fillWidth: true
                                    from: 0; to: 1; step: 0.005; decimals: 3
                                    value: Roi.threshold
                                    onMoved: (v) => { Roi.threshold = v; maskDebounce.restart() }
                                    onCommitted: (v) => { Roi.threshold = v; Roi.refreshMask() }
                                }
                            }

                            // mask mode + background sigma (both threshold modes)
                            ColumnLayout {
                                visible: root.isThresh
                                Layout.fillWidth: true; spacing: sc.sp2; Layout.topMargin: sc.sp1
                                PanelLabel { text: "MASK MODE" }
                                Select {
                                    Layout.fillWidth: true
                                    model: Roi.maskModes
                                    currentIndex: Math.max(0, Roi.maskModes.indexOf(Roi.maskMode))
                                    onPicked: (t) => Roi.maskMode = t
                                }
                            }
                            ColumnLayout {
                                visible: root.isThresh
                                Layout.fillWidth: true; spacing: sc.sp2
                                PanelLabel { text: "BACKGROUND σ" }
                                Slider {
                                    Layout.fillWidth: true
                                    from: 0; to: 100; step: 1; decimals: 1
                                    value: Roi.bgSigma
                                    onMoved: (v) => { Roi.bgSigma = v; maskDebounce.restart() }
                                    onCommitted: (v) => { Roi.bgSigma = v; Roi.refreshMask() }
                                }
                            }

                            // ── drawing tool: click-a-polygon vs paint ──────────
                            // Shown whenever an image is loaded, NOT only in Manual
                            // polygon mode: gating it on the mode meant a file set to
                            // "None" (the common case) offered no drawing tools and no
                            // hint that picking a tool is what reveals them.  Choosing
                            // one switches the mode.
                            ColumnLayout {
                                Layout.fillWidth: true; Layout.topMargin: sc.sp2; spacing: sc.sp2
                                visible: Roi.hasImage && !Roi.runScoped
                                Text { text: "Drawing tool"; color: pal.TXT_MUTED
                                       font.pixelSize: sc.textXs }
                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp2
                                    Button {
                                        Layout.fillWidth: true
                                        text: "Polygon"; icon: "waypoints"
                                        variant: Roi.tool === "polygon" ? "primary" : "secondary"
                                        onClicked: { if (!root.isPoly) Roi.roiMode = "Manual polygon"
                                                       Roi.setTool("polygon") }
                                    }
                                    Button {
                                        Layout.fillWidth: true
                                        text: "Brush"; icon: "palette"
                                        variant: Roi.tool === "brush" ? "primary" : "secondary"
                                        onClicked: { if (!root.isPoly) Roi.roiMode = "Manual polygon"
                                                       Roi.setTool("brush") }
                                    }
                                    Button {
                                        Layout.fillWidth: true
                                        text: "Eraser"; icon: "x"
                                        variant: Roi.tool === "eraser" ? "primary" : "secondary"
                                        onClicked: { if (!root.isPoly) Roi.roiMode = "Manual polygon"
                                                       Roi.setTool("eraser") }
                                    }
                                }
                                ColumnLayout {
                                    Layout.fillWidth: true; spacing: sc.sp1
                                    visible: Roi.brushActive
                                    RowLayout {
                                        Layout.fillWidth: true; spacing: sc.sp2
                                        Text { text: "Brush size"; color: pal.TXT
                                               font.pixelSize: sc.textSm }
                                        Item { Layout.fillWidth: true }
                                        Text { text: (Roi.brushRadius * 2).toFixed(0) + " px"
                                               color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                    }
                                    Slider {
                                        Layout.fillWidth: true
                                        showValue: false
                                        from: 1; to: 60; step: 0.5; decimals: 1
                                        value: Roi.brushRadius
                                        onMoved: (v) => Roi.brushRadius = v
                                        onCommitted: (v) => Roi.brushRadius = v
                                    }
                                    RowLayout {
                                        Layout.fillWidth: true; spacing: sc.sp2
                                        Button {
                                            Layout.fillWidth: true
                                            variant: "secondary"; text: "Undo stroke"; icon: "rotate-ccw"
                                            enabled: Roi.canUndoStroke
                                            onClicked: Roi.undoStroke()
                                        }
                                        Button {
                                            Layout.fillWidth: true
                                            variant: "secondary"; text: "Erase all"; icon: "x"
                                            onClicked: Roi.clearBrush()
                                        }
                                    }
                                    Text {
                                        Layout.fillWidth: true; wrapMode: Text.WordWrap
                                        text: "Drag to paint the region; the eraser trims it. " +
                                              "Painted shapes are stored as ordinary ROI outlines, " +
                                              "so everything downstream is unchanged."
                                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3
                                    }
                                }
                            }

                            // polygon count + close shape
                            RowLayout {
                                Layout.fillWidth: true; Layout.topMargin: sc.sp1; spacing: sc.sp3
                                visible: root.isPoly
                                Badge { text: Roi.polygonCount + (Roi.polygonCount === 1 ? " region" : " regions")
                                        tone: Roi.polygonCount > 0 ? pal.SUCCESS : pal.TXT_MUTED }
                                Item { Layout.fillWidth: true }
                                Button { variant: "secondary"; text: "Close shape"; icon: "check"
                                         enabled: Roi.canClose; onClicked: Roi.closeDraft() }
                            }

                            // Multiple ROIs → analyse each as its own replicate.
                            ColumnLayout {
                                Layout.fillWidth: true; Layout.topMargin: sc.sp2; spacing: sc.sp2
                                visible: root.isPoly && Roi.polygonCount > 1
                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp3
                                    ColumnLayout {
                                        Layout.fillWidth: true; spacing: 1
                                        Text { text: "Analyse each ROI separately"; color: pal.TXT
                                               font.pixelSize: sc.textSm }
                                        Text { Layout.fillWidth: true; wrapMode: Text.WordWrap
                                               text: "Individual replicates — one output per ROI, so the cells don't skew each other's D-values."
                                               color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3 }
                                    }
                                    Switch { checked: Roi.splitReplicates
                                             onToggled: (c) => Roi.splitReplicates = c }
                                }
                                Repeater {
                                    model: Roi.splitReplicates ? Roi.roiLabels.length : 0
                                    RowLayout {
                                        Layout.fillWidth: true; spacing: sc.sp2
                                        Text { text: "ROI " + (index + 1); color: pal.TXT_MUTED
                                               font.pixelSize: sc.textXs; Layout.preferredWidth: 44 }
                                        FieldInput {
                                            Layout.fillWidth: true
                                            placeholderText: "cell" + (index + 1)
                                            text: Roi.roiLabels[index]
                                            onEditingFinished: Roi.setRoiLabel(index, text)
                                        }
                                    }
                                }
                            }
                        }

                        // ── DETECTION: WHICH spots to keep ──────────────────────────
                        // Needs no ROI: the mass profile reads frames directly, which is
                        // why this is reachable without drawing anything first.
                        ColumnLayout {
                            visible: root.isDetect
                            Layout.fillWidth: true; spacing: sc.sp3

                            // ── DETECTION THRESHOLD ─────────────────────────
                            // Its own section, deliberately: choosing the number is
                            // a separate job from drawing spots on the image, and it
                            // needs no overlay — the profile reads frames directly.
                            // Previously this evidence was only reachable by first
                            // switching the preview on.
                            ColumnLayout {
                                id: thrSection
                                // How much one arrow press moves the threshold.  A
                                // fixed increment cannot serve both ends: a working
                                // minmass is ~0.45 on trackpy and orders of magnitude
                                // larger on another backend, because mass is in the
                                // detector's own units.
                                property real nudge: 0.01
                                visible: !Roi.runScoped
                                Layout.fillWidth: true; spacing: sc.sp2; Layout.topMargin: sc.sp2
                                RowLayout {
                                    Layout.fillWidth: true
                                    PanelLabel { text: "DETECTION THRESHOLD" }
                                    Item { Layout.fillWidth: true }
                                    Text {
                                        text: Roi.detectMinmass.toLocaleString(Qt.locale(), "f", 2)
                                        color: pal.ACC; font.pixelSize: sc.textXs; font.family: "Menlo"
                                    }
                                }
                                Slider {
                                    Layout.fillWidth: true
                                    showValue: false
                                    from: 0; to: 50; step: 0.01; decimals: 3
                                    value: Roi.detectMinmass
                                    onMoved: (v) => { Roi.detectMinmass = v; guide.refresh()
                                                      if (Roi.detectEnabled) spotsDebounce.restart() }
                                    onCommitted: (v) => { Roi.detectMinmass = v; guide.refresh()
                                                          if (Roi.detectEnabled) Roi.refreshSpots() }
                                }
                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp2
                                    Text { text: "Exact"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                    SpinBox {
                                        objectName: "minmassSpin"
                                        Layout.fillWidth: true
                                        steppers: true                  // − / + , hold to repeat, ↑↓ keys
                                        from: 0; to: 1000000; decimals: 4
                                        step: thrSection.nudge
                                        value: Roi.detectMinmass
                                        onCommitted: (v) => { Roi.detectMinmass = v; guide.refresh()
                                                              if (Roi.detectEnabled) spotsDebounce.restart() }
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp2
                                    Text { text: "Nudge by"; color: pal.TXT_MUTED
                                           font.pixelSize: sc.textXs }
                                    Segmented {
                                        objectName: "minmassStepPick"
                                        Layout.fillWidth: true
                                        options: [{ v: "0.001", t: "0.001" }, { v: "0.01", t: "0.01" },
                                                  { v: "0.1", t: "0.1" }, { v: "1", t: "1" }]
                                        value: thrSection.nudge.toString()
                                        onPicked: (v) => thrSection.nudge = parseFloat(v)
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp3
                                    ColumnLayout {
                                        Layout.fillWidth: true; spacing: 1
                                        Text { text: "Use for this file only"; color: pal.TXT
                                               font.pixelSize: sc.textSm }
                                        Text { Layout.fillWidth: true; wrapMode: Text.WordWrap
                                               text: "Saves the threshold with this file's ROI, so a batch can " +
                                                     "run one recording at a different value. Off, the slider sets " +
                                                     "the shared sidebar threshold every file uses."
                                               color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3 }
                                    }
                                    Switch {
                                        checked: Roi.minmassPerFile
                                        onToggled: (c) => Roi.setMinmassPerFile(c)
                                    }
                                }
                                Alert {
                                    Layout.fillWidth: true
                                    visible: Roi.minmassPerFile
                                    severity: "warn"
                                    text: "Per-file thresholds are not comparable by default: mass is " +
                                          "file-relative, and tuning each recording until the counts agree is " +
                                          "how a detection difference gets manufactured. Use a stated rule, " +
                                          "apply it to every condition, and report it."
                                }
                                Text {
                                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                                    text: "Sets the sidebar's Threshold (minmass) and turns Auto minmass off. " +
                                          "Mass is in the selected detector's own units and is file-relative — " +
                                          "a value from another backend or recording does not carry over."
                                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3
                                }
                            // ── threshold guidance ──────────────────
                            // Detect once at minmass 0, then re-score any
                            // threshold instantly: the histogram and the
                            // readouts follow the slider with no detection.
                            // refresh() is cheap because the harvest AND the
                            // noise floor are cached per recording, so it is
                            // safe on every slider move.
                            ColumnLayout {
                                id: guide
                                Layout.fillWidth: true; spacing: sc.sp1
                                property var prof: ({})
                                function refresh() { prof = Roi.massProfile(); massHist.requestPaint() }

                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp2
                                    Text { text: "Mass distribution"; color: pal.TXT
                                           font.pixelSize: sc.textSm }
                                    Item { Layout.fillWidth: true }
                                    Button {
                                        variant: "secondary"; text: "Profile"; icon: "chart-spline"
                                        onClicked: { Roi.invalidateMassProfile(); guide.refresh() }
                                    }
                                }

                                Canvas {
                                    id: massHist
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 92
                                    onPaint: {
                                        var ctx = getContext("2d"); ctx.reset()
                                        ctx.clearRect(0, 0, width, height)
                                        var p = guide.prof
                                        if (!p || !p.counts || p.counts.length === 0) return
                                        var e = p.edges, c = p.counts
                                        var lo = e[0], hi = e[e.length - 1]
                                        var span = (hi - lo) || 1
                                        var mx = 0
                                        for (var i = 0; i < c.length; ++i) mx = Math.max(mx, c[i])
                                        if (mx <= 0) return
                                        var toX = function (log10m) {
                                            return (log10m - lo) / span * width }
                                        // bars, split at the threshold so the
                                        // kept fraction is visible, not inferred
                                        var cut = Math.log(p.minmass) / Math.LN10
                                        var bw = width / c.length
                                        for (i = 0; i < c.length; ++i) {
                                            var h = c[i] / mx * (height - 14)
                                            var mid = (e[i] + e[i + 1]) / 2
                                            ctx.fillStyle = (mid >= cut) ? pal.SUCCESS : pal.TXT_MUTED
                                            ctx.globalAlpha = (mid >= cut) ? 0.85 : 0.30
                                            ctx.fillRect(i * bw, height - 14 - h, Math.max(1, bw - 1), h)
                                        }
                                        ctx.globalAlpha = 1
                                        // markers
                                        var mark = function (v, colour, dash) {
                                            if (!v) return
                                            var x = toX(Math.log(v) / Math.LN10)
                                            if (x < 0 || x > width) return
                                            ctx.beginPath(); ctx.setLineDash(dash)
                                            ctx.moveTo(x, 0); ctx.lineTo(x, height - 14)
                                            ctx.strokeStyle = colour; ctx.lineWidth = 1.5
                                            ctx.stroke(); ctx.setLineDash([])
                                        }
                                        mark(p.knee, pal.WARN || "#e0a33a", [4, 3])
                                        mark(p.noise_floor, pal.DANGER, [2, 2])
                                        mark(p.minmass, pal.TXT, [])
                                    }
                                }

                                Text {
                                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                                    visible: !!(guide.prof && guide.prof.n_candidates > 0)
                                    text: {
                                        var p = guide.prof
                                        if (!p || !p.n_candidates) return ""
                                        return "— threshold · " +
                                            (p.knee ? "– – knee " + p.knee.toFixed(3) + " · " : "") +
                                            (p.noise_floor ? "· · noise floor " + p.noise_floor.toFixed(3) : "no noise floor")
                                    }
                                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                }

                                Text {
                                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                                    visible: !!(guide.prof && guide.prof.n_candidates > 0)
                                    text: {
                                        var p = guide.prof
                                        if (!p || !p.n_candidates) return ""
                                        return p.per_frame_kept.toFixed(1) + " spots/frame kept of " +
                                               p.per_frame_all.toFixed(0) + " candidates (" +
                                               (p.kept_fraction * 100).toFixed(1) + "%), from " +
                                               p.n_frames_sampled + " frames across the recording."
                                    }
                                    color: pal.TXT; font.pixelSize: sc.textXs
                                }

                                Text {
                                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                                    visible: text.length > 0
                                    text: (guide.prof && guide.prof.floor_status &&
                                           !guide.prof.noise_floor) ? guide.prof.floor_status : ""
                                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3
                                }

                                Alert {
                                    Layout.fillWidth: true
                                    // !! is load-bearing: guide.prof starts as {},
                                    // so this chain yields `undefined`, the bool
                                    // binding fails, and `visible` falls back to its
                                    // default — TRUE — showing an empty red alert.
                                    visible: !!(guide.prof && guide.prof.warning &&
                                                guide.prof.warning.length > 0)
                                    severity: (guide.prof && guide.prof.below_noise_floor) ? "danger" : "warn"
                                    text: (guide.prof && guide.prof.warning) ? guide.prof.warning : ""
                                }

                                Connections {
                                    target: Roi
                                    function onDetectChanged() { if (Roi.detectEnabled) guide.refresh() }
                                }
                            }
                            }

                            // detection-threshold (minmass) preview + slider
                            ColumnLayout {
                                visible: !Roi.runScoped
                                Layout.fillWidth: true; spacing: sc.sp2; Layout.topMargin: sc.sp1
                                RowLayout {
                                    Layout.fillWidth: true
                                    PanelLabel { text: "DETECTION PREVIEW" }
                                    Item { Layout.fillWidth: true }
                                    Text {
                                        visible: Roi.detectEnabled
                                        text: Roi.spotsStale ? "outdated" : Roi.spotCount + " pass"
                                        color: pal.ACC; font.pixelSize: sc.textXs; font.family: "Menlo"
                                    }
                                    Switch {
                                        checked: Roi.detectEnabled
                                        onToggled: (c) => { Roi.detectEnabled = c }
                                    }
                                }
                                // explicitly labelled so it's clear this is the minmass
                                // overlay controls only — the threshold itself lives
                                // in its own section above
                                ColumnLayout {
                                    visible: Roi.detectEnabled
                                    Layout.fillWidth: true; spacing: 2
                                    Button {
                                        text: "Refresh overlay"
                                        Layout.fillWidth: true
                                        onClicked: { spotsDebounce.stop(); Roi.refreshSpots() }
                                    }
                                    Text {
                                        Layout.fillWidth: true; wrapMode: Text.WordWrap
                                        text: "Green: detection + ROI. Orange: contrast rejected. Red: outside ROI. Blue: ROI unchecked."
                                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                    }
                                    Text {
                                        Layout.fillWidth: true; wrapMode: Text.WordWrap
                                        text: "Click a spot to inspect. The slider above sets the threshold; Save threshold to keep it."
                                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                    }
                                    Text {
                                        Layout.fillWidth: true; wrapMode: Text.WordWrap
                                        text: Roi.spotInspection
                                        visible: text.length > 0
                                        color: pal.TXT; font.pixelSize: sc.textXs
                                    }

                                    Text {
                                        Layout.fillWidth: true; wrapMode: Text.WordWrap
                                        text: Roi.spotSummary
                                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3
                                    }
                                }
                            }
                        }

                        Item { Layout.fillHeight: true }

                        RowLayout {
                            Layout.fillWidth: true; spacing: sc.sp3
                            Button {
                                Layout.fillWidth: true
                                visible: root.isRoiPanel && root.isPoly
                                variant: "secondary"; text: "Clear"; icon: "rotate-ccw"
                                onClicked: Roi.clearPolygons()
                            }
                            Button {
                                Layout.fillWidth: true
                                variant: "primary"; icon: "check"
                                text: root.isDetect ? "Save threshold" : "Save ROI"
                                onClicked: root.saveRoi()
                            }
                        }
                    }
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

    Keys.onEscapePressed: Roi.cancel()
}
