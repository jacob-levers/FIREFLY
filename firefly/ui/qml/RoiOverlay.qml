import QtQuick
import QtQuick.Layouts
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

    function saveRoi() {
        if (root.isPoly && Roi.canClose) Roi.closeDraft()
        Roi.commit()
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
                Icon { name: "scan-search"; size: 15; color: pal.ACC }
                Text { text: "Preview & ROI"; color: pal.TXT; font.pixelSize: sc.textMd; font.bold: true }
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
                    color: "#05070a"
                    clip: true

                    Item {
                        id: imgArea
                        anchors.fill: parent
                        readonly property real sscale: Roi.imageWidth > 0 ? bg.paintedWidth / Roi.imageWidth : 1
                        readonly property real offX: (width - bg.paintedWidth) / 2
                        readonly property real offY: (height - bg.paintedHeight) / 2
                        function toImg(px, py) { return [(py - offY) / sscale, (px - offX) / sscale] }
                        function toDispX(x) { return offX + x * sscale }
                        function toDispY(y) { return offY + y * sscale }

                        Image {
                            id: bg
                            anchors.fill: parent
                            fillMode: Image.PreserveAspectFit
                            smooth: false; cache: false; asynchronous: true
                            source: Roi.hasImage ? ("image://roibg/" + Roi.imageToken) : ""
                            onPaintedWidthChanged: canvas.requestPaint()
                        }

                        // threshold-mask overlay (auto / manual) — constant across
                        // frames, so scrubbing shows which particles it selects
                        Image {
                            anchors.fill: parent
                            fillMode: Image.PreserveAspectFit
                            smooth: false; cache: false; asynchronous: true
                            visible: root.isThresh && Roi.hasMask
                            opacity: visible ? 1 : 0
                            Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                            source: Roi.hasMask ? ("image://roimask/" + Roi.maskToken) : ""
                        }

                        // detected-spot overlay (green circles at the current minmass)
                        Image {
                            anchors.fill: parent
                            fillMode: Image.PreserveAspectFit
                            smooth: true; cache: false; asynchronous: true
                            visible: Roi.detectEnabled && Roi.hasSpots
                            opacity: visible ? 1 : 0
                            Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                            source: Roi.hasSpots ? ("image://roispots/" + Roi.spotToken) : ""
                        }

                        ScanLine { anchors.fill: parent; active: root.scanning }

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

                        MouseArea {
                            anchors.fill: parent
                            enabled: Roi.hasImage && root.isPoly
                            cursorShape: root.isPoly ? Qt.CrossCursor : Qt.ArrowCursor
                            onClicked: (m) => {
                                var yx = imgArea.toImg(m.x, m.y)
                                if (yx[0] >= 0 && yx[1] >= 0 && yx[0] <= Roi.imageHeight && yx[1] <= Roi.imageWidth)
                                    Roi.addVertex(yx[0], yx[1])
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
                                         + (root.isThresh && Roi.hasMask
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
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: sc.sp4
                        spacing: sc.sp3

                        // view toggle
                        ColumnLayout {
                            visible: Roi.nFrames > 1
                            Layout.fillWidth: true; spacing: sc.sp2
                            PanelLabel { text: "VIEW" }
                            Segmented {
                                Layout.fillWidth: true
                                options: [{ v: "proj", t: "Max proj" }, { v: "raw", t: "Raw frames" }]
                                value: Roi.viewMode
                                onPicked: (v) => Roi.setViewMode(v === "raw" ? "Raw frames" : "Max projection")
                            }
                        }

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

                        // colour map
                        PanelLabel { text: "COLOUR"; Layout.topMargin: sc.sp1 }
                        Select {
                            Layout.fillWidth: true
                            model: Roi.cmaps
                            currentIndex: Math.max(0, Roi.cmaps.indexOf(Roi.cmap))
                            onPicked: (t) => Roi.cmap = t
                        }

                        // detection threshold (minmass) preview + slider
                        ColumnLayout {
                            Layout.fillWidth: true; spacing: sc.sp2; Layout.topMargin: sc.sp1
                            RowLayout {
                                Layout.fillWidth: true
                                PanelLabel { text: "DETECTIONS" }
                                Item { Layout.fillWidth: true }
                                Text {
                                    visible: Roi.detectEnabled
                                    text: Roi.hasSpots ? (Roi.spotCount.toLocaleString(Qt.locale(), "f", 0) + " spots") : "0 spots"
                                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                                }
                                Switch {
                                    checked: Roi.detectEnabled
                                    onToggled: (c) => { Roi.detectEnabled = c
                                                        if (c) Roi.refreshSpots() }
                                }
                            }
                            Slider {
                                Layout.fillWidth: true
                                visible: Roi.detectEnabled
                                from: 0; to: 50; step: 0.25; decimals: 2
                                value: Roi.detectMinmass
                                onMoved: (v) => { Roi.detectMinmass = v; spotsDebounce.restart() }
                                onCommitted: (v) => { Roi.detectMinmass = v; Roi.refreshSpots() }
                            }
                            Text {
                                visible: Roi.detectEnabled
                                Layout.fillWidth: true; wrapMode: Text.WordWrap
                                text: "Green circles = spots detected at this minmass. This sets the run's detection threshold."
                                color: pal.TXT_MUTED; font.pixelSize: sc.textXs; lineHeight: 1.3
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

                        Item { Layout.fillHeight: true }

                        RowLayout {
                            Layout.fillWidth: true; spacing: sc.sp3
                            Button {
                                Layout.fillWidth: true
                                visible: root.isPoly
                                variant: "secondary"; text: "Clear"; icon: "rotate-ccw"
                                onClicked: Roi.clearPolygons()
                            }
                            Button {
                                Layout.fillWidth: true
                                variant: "primary"; text: "Save ROI"; icon: "check"
                                onClicked: root.saveRoi()
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
