import QtQuick
import QtQuick.Layouts
import "../components"

// Import tab. Single mode: a drop zone, a Recording row, an Output row (+ auto-
// name toggle), a metadata stat strip, and a live-detection preview. Batch mode:
// a Source row, an Output row, and a run queue with per-series status. Calibration
// (pixel size / frame interval) lives in the left parameter sidebar.
// Bound to ImportController / BatchController.
Flickable {
    id: root
    property bool batchMode: false
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    contentWidth: width
    contentHeight: col.implicitHeight + 56
    clip: true

    // ── reusable: labelled value row (icon-chip + label + value + badge) + Browse
    component InfoRow: RowLayout {
        id: infoRow
        property string icon: "folder-open"
        property color iconTint: Theme.palette.ACC
        property string label: ""
        property string value: ""
        property bool mono: true
        property bool placeholder: false
        property string badgeText: ""
        property color badgeTone: Theme.palette.ACC
        property string browseText: "Browse"
        signal browse()
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        Layout.fillWidth: true
        spacing: sc.sp4
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 60
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                spacing: sc.sp4
                Rectangle {
                    width: 40; height: 40; radius: sc.radiusLg
                    color: Qt.rgba(infoRow.iconTint.r, infoRow.iconTint.g, infoRow.iconTint.b, 0.12)
                    border.width: 1
                    border.color: Qt.rgba(infoRow.iconTint.r, infoRow.iconTint.g, infoRow.iconTint.b, 0.22)
                    Icon { anchors.centerIn: parent; name: infoRow.icon
                           color: infoRow.iconTint; size: 19 }
                }
                Text { text: infoRow.label; color: pal.TXT_MUTED
                       font.pixelSize: sc.textSm; Layout.preferredWidth: 76 }
                Text {
                    text: infoRow.value
                    color: infoRow.placeholder ? pal.TXT_MUTED : pal.TXT
                    font.pixelSize: sc.textSm
                    font.family: (infoRow.mono && !infoRow.placeholder)
                                 ? "Menlo" : Qt.application.font.family
                    elide: Text.ElideMiddle
                    Layout.fillWidth: true; Layout.preferredWidth: 0
                }
                // slides in from the right (+12px) + fades when the badge appears
                Badge {
                    readonly property bool present: infoRow.badgeText !== ""
                    visible: present || opacity > 0.001
                    text: infoRow.badgeText
                    tone: infoRow.badgeTone; Layout.alignment: Qt.AlignVCenter
                    opacity: present ? 1 : 0
                    transform: Translate {
                        x: present ? 0 : 12
                        Behavior on x { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
                    }
                    Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
                }
            }
        }
        Button { visible: infoRow.browseText !== ""; variant: "secondary"
                 text: infoRow.browseText; icon: "folder-open"
                 onClicked: infoRow.browse() }
    }

    // ── reusable: metadata stat tile ──────────────────────────────────────
    component StatCard: Card {
        id: scard
        property string icon: ""
        property string label: ""
        property string value: ""
        property real numValue: -1        // ≥0 → animated count-up instead of `value`
        property int numDecimals: 0
        property string numSuffix: ""
        property int enterDelay: 0         // stagger (ms) for the entrance
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        Layout.fillWidth: true
        Layout.preferredHeight: 66
        // fade + rise entrance, staggered by enterDelay
        opacity: 0
        transform: Translate { id: scT; y: 8 }
        Component.onCompleted: scEnter.start()
        SequentialAnimation {
            id: scEnter
            PauseAnimation { duration: Theme.reducedMotion ? 0 : scard.enterDelay }
            ParallelAnimation {
                NumberAnimation { target: scard; property: "opacity"; to: 1
                                  duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic }
                NumberAnimation { target: scT; property: "y"; to: 0
                                  duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic }
            }
        }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp4
            spacing: sc.sp4
            Icon { name: scard.icon; size: 18; color: pal.TXT_MUTED }
            ColumnLayout {
                spacing: 1
                Text { text: scard.label; color: pal.TXT_MUTED; font.pixelSize: 10
                       font.bold: true; font.letterSpacing: 0.8 }
                Text { visible: scard.numValue < 0
                       text: scard.value; color: pal.TXT; font.pixelSize: sc.textLg
                       font.family: "Menlo" }
                CountUp { visible: scard.numValue >= 0
                          value: Math.max(0, scard.numValue)
                          decimals: scard.numDecimals; suffix: scard.numSuffix
                          color: pal.TXT; font.pixelSize: sc.textLg; font.family: "Menlo" }
            }
            Item { Layout.fillWidth: true }
        }
    }

    // ── reusable: tri-state selection box (on / off / indeterminate) ──────
    component SelBox: Rectangle {
        id: sb
        property string state: "off"          // on | off | ind
        signal clicked()
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        readonly property bool filled: state === "on" || state === "ind"
        implicitWidth: 16; implicitHeight: 16; radius: sc.radiusXs
        color: filled ? pal.ACC : pal.PANEL_ALT
        border.width: 1
        border.color: filled ? pal.ACC : (sbHov.hovered ? pal.BORDER_HI : pal.BORDER)
        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
        Icon {
            anchors.centerIn: parent
            visible: sb.filled
            name: sb.state === "ind" ? "minus" : "check"
            size: 11; color: pal.ACC_FG
            scale: sb.filled ? 1 : 0.6
            Behavior on scale { NumberAnimation { duration: Theme.reducedMotion ? 0 : 90; easing.type: Easing.OutCubic } }
        }
        HoverHandler { id: sbHov; cursorShape: Qt.PointingHandCursor }
        TapHandler { onTapped: sb.clicked() }
    }

    // ── reusable: header text-link (Expand all / Select all …) ────────────
    component LinkBtn: Text {
        id: lb
        signal clicked()
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        color: lbHov.hovered ? pal.TXT : pal.TXT_MUTED
        font.pixelSize: sc.textSm
        padding: 4
        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
        HoverHandler { id: lbHov; cursorShape: Qt.PointingHandCursor }
        TapHandler { onTapped: lb.clicked() }
    }

    // ── reusable: run-queue row (series header + constituent-file drawer) ──
    component SeriesRow: Rectangle {
        id: srow
        property var item
        property int rowIndex: 0
        property real collapse: 1                       // 1 → full height, 0 → removed
        readonly property bool removing: item.removing === true
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        readonly property string st: item.status
        Layout.fillWidth: true
        implicitHeight: bodyCol.implicitHeight * collapse
        clip: true
        radius: sc.radiusMd
        color: st === "running" ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.06) : "transparent"
        border.width: 1
        border.color: st === "running" ? pal.ACC : (rowHov.hovered ? pal.BORDER_HI : pal.BORDER)
        Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
        HoverHandler { id: rowHov }

        // staggered fade-rise as rows populate / get added (one-shot on mount;
        // selection + expand are dataChanged, not re-creates, so no replay)
        opacity: 0
        transform: Translate { id: rowT; y: 6 }
        Component.onCompleted: rowEnter.start()
        SequentialAnimation {
            id: rowEnter
            PauseAnimation { duration: Theme.reducedMotion ? 0 : Math.min(srow.rowIndex, 14) * 35 }
            ParallelAnimation {
                NumberAnimation { target: srow; property: "opacity"; to: 1
                                  duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic }
                NumberAnimation { target: rowT; property: "y"; to: 0
                                  duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic }
            }
        }

        // collapse + fade out on remove, THEN drop from the model (§4 row remove)
        onRemovingChanged: if (removing) removeAnim.start()
        ParallelAnimation {
            id: removeAnim
            NumberAnimation { target: srow; property: "opacity"; to: 0
                              duration: Theme.reducedMotion ? 0 : 200; easing.type: Easing.OutCubic }
            NumberAnimation { target: srow; property: "collapse"; to: 0
                              duration: Theme.reducedMotion ? 0 : 200; easing.type: Easing.OutCubic }
            onFinished: Batch.finalizeRemove(srow.item.key)
        }

        ColumnLayout {
            id: bodyCol
            anchors.left: parent.left; anchors.right: parent.right
            spacing: 0

            // ── series header ──────────────────────────────────────────
            RowLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: 44
                Layout.leftMargin: sc.sp3; Layout.rightMargin: sc.sp3
                spacing: sc.sp3

                Item {                          // chevron (toggles the drawer)
                    Layout.preferredWidth: 16; Layout.preferredHeight: 16
                    Icon { anchors.centerIn: parent; name: "chevron-right"; size: 14
                           color: srow.item.open ? pal.ACC : pal.TXT_MUTED
                           rotation: srow.item.open ? 90 : 0
                           Behavior on rotation { NumberAnimation { duration: Theme.reducedMotion ? 0 : 90; easing.type: Easing.OutCubic } } }
                    TapHandler { onTapped: Batch.setOpen(srow.item.key, !srow.item.open) }
                    HoverHandler { cursorShape: Qt.PointingHandCursor }
                }
                SelBox {                        // series tri-state checkbox
                    state: srow.item.selState
                    onClicked: Batch.setChecked(srow.item.key, srow.item.selState !== "on")
                }
                Item {                          // name + ROI badge (toggles drawer)
                    Layout.fillWidth: true; Layout.preferredHeight: 30
                    RowLayout {
                        anchors.fill: parent; spacing: sc.sp3
                        Text { text: srow.item.name; color: pal.TXT; font.pixelSize: sc.textSm
                               font.family: "Menlo"; elide: Text.ElideMiddle
                               Layout.fillWidth: true; Layout.preferredWidth: 0 }
                        Pop {                   // ROI badge (pops in on save)
                            visible: srow.item.hasRoi
                            Rectangle {
                                implicitHeight: 18; implicitWidth: roiRow.implicitWidth + sc.sp4 * 2
                                radius: sc.radiusPill
                                color: Qt.rgba(pal.SUCCESS.r, pal.SUCCESS.g, pal.SUCCESS.b, 0.14)
                                border.width: 1
                                border.color: Qt.rgba(pal.SUCCESS.r, pal.SUCCESS.g, pal.SUCCESS.b, 0.32)
                                RowLayout {
                                    id: roiRow; anchors.centerIn: parent; spacing: sc.sp1
                                    Icon { name: "scan-search"; size: 10; color: pal.SUCCESS }
                                    Text { text: srow.item.roiLabel || "ROI"; color: pal.SUCCESS
                                           font.pixelSize: sc.textXs; font.bold: true }
                                }
                            }
                        }
                    }
                    TapHandler { onTapped: Batch.setOpen(srow.item.key, !srow.item.open) }
                }
                Button {                        // Preview & ROI — left of the ×N/size meta
                    variant: "secondary"; text: "Preview & ROI"; icon: "scan-search"
                    visible: rowHov.hovered || srow.item.hasRoi || srow.st === "running"
                    onClicked: Roi.editFile(srow.item.primaryPath)
                }
                Text { visible: srow.item.fileCount > 1; text: "×" + srow.item.fileCount
                       color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                Text {
                    text: (srow.item.framesTotal > 0
                           ? srow.item.framesTotal.toLocaleString(Qt.locale(), "f", 0) + " fr · " : "")
                          + srow.item.sizeStr
                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                }
                Text {                          // status chip
                    text: srow.st === "running" ? (srow.item.progress + "%")
                        : srow.st === "done" ? "Done" : srow.st === "error" ? "Failed"
                        : srow.st === "skipped" ? "Skipped" : "Queued"
                    color: srow.st === "done" ? pal.SUCCESS : srow.st === "error" ? pal.DANGER
                         : srow.st === "running" ? pal.ACC : pal.TXT_MUTED
                    font.pixelSize: sc.textXs
                    font.family: srow.st === "running" ? "Menlo" : Qt.application.font.family
                    Layout.preferredWidth: 56; horizontalAlignment: Text.AlignRight
                }
                IconButton {                    // remove from queue (hover)
                    icon: "x"; tip: "Remove"; danger: true; size: 24
                    visible: rowHov.hovered && !Batch.running
                    onClicked: Batch.removeSeries(srow.item.key)
                }
            }

            // ── constituent-file drawer (animated height) ───────────────
            Item {
                Layout.fillWidth: true
                clip: true
                implicitHeight: srow.item.open ? filesCol.implicitHeight : 0
                Behavior on implicitHeight { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
                Rectangle { anchors.top: parent.top; anchors.left: parent.left
                            anchors.right: parent.right; height: 1; color: pal.BORDER }
                ColumnLayout {
                    id: filesCol
                    anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
                    spacing: 0
                    Repeater {
                        model: srow.item.parts
                        delegate: RowLayout {
                            required property var modelData
                            required property int index
                            Layout.fillWidth: true
                            Layout.preferredHeight: 32
                            Layout.leftMargin: 40; Layout.rightMargin: sc.sp4
                            spacing: sc.sp3
                            SelBox {
                                state: modelData.checked ? "on" : "off"
                                onClicked: Batch.setFileChecked(srow.item.key, index, !modelData.checked)
                            }
                            Icon { name: "image"; size: 13; color: pal.TXT_MUTED }
                            Text { text: modelData.name
                                   color: modelData.checked ? pal.TXT : pal.TXT_MUTED
                                   font.pixelSize: sc.textXs; font.family: "Menlo"
                                   elide: Text.ElideMiddle; Layout.fillWidth: true; Layout.preferredWidth: 0 }
                            Text { text: (modelData.frames > 0
                                          ? modelData.frames.toLocaleString(Qt.locale(), "f", 0) + " fr · " : "")
                                         + modelData.sizeStr
                                   color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo" }
                        }
                    }
                }
            }
        }
    }

    ColumnLayout {
        id: col
        x: 28; y: 20
        width: Math.min(960, root.width - 56)
        spacing: sc.sp6

        // ── single / batch mode toggle ────────────────────────────────
        RowLayout {
            spacing: sc.sp2
            Repeater {
                model: [{ t: "Single analysis", b: false, ic: "scan-search" },
                        { t: "Batch analysis", b: true, ic: "layers" }]
                delegate: Rectangle {
                    required property var modelData
                    readonly property bool active: root.batchMode === modelData.b
                    implicitWidth: pillRow.implicitWidth + sc.sp6 * 2
                    implicitHeight: 34; radius: sc.radiusLg
                    color: active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14) : pal.PANEL_ALT
                    border.width: 1; border.color: active ? pal.ACC : pal.BORDER
                    RowLayout {
                        id: pillRow; anchors.centerIn: parent; spacing: sc.sp2
                        Icon { name: modelData.ic; size: 14; color: active ? pal.ACC : pal.TXT_MUTED }
                        Text { text: modelData.t; color: active ? pal.ACC : pal.TXT_MUTED
                               font.pixelSize: sc.textSm; font.bold: active }
                    }
                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: root.batchMode = modelData.b }
                }
            }
        }

        // ══════════════════════ SINGLE ANALYSIS ══════════════════════════
        // ── drop zone ────────────────────────────────────────────────
        Item {
            visible: !root.batchMode
            Layout.fillWidth: true
            Layout.preferredHeight: 96
            Canvas {                                 // dashed rounded border
                id: dash
                anchors.fill: parent
                property color stroke: dropArea.containsDrag ? pal.ACC : pal.BORDER_HI
                Behavior on stroke { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                onStrokeChanged: requestPaint()
                onWidthChanged: requestPaint()
                onHeightChanged: requestPaint()
                onPaint: {
                    var ctx = getContext("2d"); ctx.reset();
                    ctx.clearRect(0, 0, width, height);
                    ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.setLineDash([6, 5]);
                    var r = sc.radiusXl, x = 1, y = 1, w = width - 2, h = height - 2;
                    ctx.beginPath();
                    ctx.moveTo(x + r, y);
                    ctx.arcTo(x + w, y,     x + w, y + h, r);
                    ctx.arcTo(x + w, y + h, x,     y + h, r);
                    ctx.arcTo(x,     y + h, x,     y,     r);
                    ctx.arcTo(x,     y,     x + w, y,     r);
                    ctx.closePath(); ctx.stroke();
                }
            }
            Rectangle {                              // hover/drag fill
                anchors.fill: parent; radius: sc.radiusXl
                color: dropArea.containsDrag ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.06) : "transparent"
            }
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: sc.sp6; anchors.rightMargin: sc.sp6
                spacing: sc.sp5
                Rectangle {
                    id: dropChip
                    width: 48; height: 48; radius: sc.radiusLg
                    color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                    border.width: 1; border.color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.22)
                    Icon { anchors.centerIn: parent; name: "arrow-up-right"
                           rotation: -45; color: pal.ACC; size: 22 }
                    // gentle "breathing" while a file is dragged over the zone
                    readonly property bool breathing: dropArea.containsDrag && !Theme.reducedMotion
                    onBreathingChanged: if (!breathing) dropChip.scale = 1
                    SequentialAnimation {
                        running: dropChip.breathing
                        loops: Animation.Infinite
                        NumberAnimation { target: dropChip; property: "scale"; from: 1.0; to: 1.06
                                          duration: Theme.reducedMotion ? 0 : 700; easing.type: Easing.InOutSine }
                        NumberAnimation { target: dropChip; property: "scale"; to: 1.0
                                          duration: Theme.reducedMotion ? 0 : 700; easing.type: Easing.InOutSine }
                    }
                }
                ColumnLayout {
                    spacing: sc.sp1
                    Text { text: "Drop a recording to analyse"; color: pal.TXT
                           font.pixelSize: sc.textLg; font.bold: true }
                    Text { text: ".tif / .nd2 / .czi · or pick a file below"
                           color: pal.TXT_MUTED; font.pixelSize: sc.textSm }
                }
                Item { Layout.fillWidth: true }
                Badge { text: "single file"; tone: pal.TXT_MUTED }
            }
            DropArea {
                id: dropArea
                anchors.fill: parent
                onDropped: (drop) => {
                    if (drop.hasUrls && drop.urls.length > 0) Import.dropFile("" + drop.urls[0])
                }
            }
            TapHandler { onTapped: Import.browseFile() }
            HoverHandler { cursorShape: Qt.PointingHandCursor }
        }

        // ── recording + output rows ──────────────────────────────────
        InfoRow {
            visible: !root.batchMode
            icon: Import.isCsv ? "circle-dot" : "microscope"
            label: "Recording"
            value: Import.hasFile ? Import.filePath : "No file selected"
            placeholder: !Import.hasFile
            badgeText: Import.hasFile
                       ? ("." + ("" + Import.fileFormat).toLowerCase()
                          + (Import.frameCount > 0
                             ? "  ·  " + Import.frameCount.toLocaleString(Qt.locale(), "f", 0) + " fr" : ""))
                       : ""
            onBrowse: Import.browseFile()
        }
        InfoRow {
            visible: !root.batchMode
            icon: "folder-open"; iconTint: pal.SUCCESS
            label: "Output"
            value: Import.outDir || "(beside the recording)"
            placeholder: !Import.outDir
            badgeText: (!Import.outputExplicit && Import.hasFile) ? "auto" : ""
            badgeTone: pal.TXT_MUTED
            browseText: "Browse"
            onBrowse: Import.browseOutDir()
        }

        // ── external-CSV options (localisation table only) ────────────
        ColumnLayout {
            visible: !root.batchMode && Import.isCsv
            Layout.fillWidth: true
            spacing: sc.sp4

            // source-format picker — auto-detect, or override when it fails
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 60
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                    spacing: sc.sp4
                    Rectangle {
                        width: 40; height: 40; radius: sc.radiusLg
                        color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                        border.width: 1
                        border.color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.22)
                        Icon { anchors.centerIn: parent; name: "circle-dot"; color: pal.ACC; size: 19 }
                    }
                    Text { text: "Source format"; color: pal.TXT_MUTED
                           font.pixelSize: sc.textSm; Layout.preferredWidth: 76 }
                    Select {
                        Layout.fillWidth: true
                        property var labels: ["Auto-detect", "PALM-Tracer", "ThunderSTORM", "Picasso", "TrackMate"]
                        property var vals: ["auto", "PALM-Tracer", "ThunderSTORM", "Picasso", "TrackMate"]
                        model: labels
                        currentIndex: Math.max(0, vals.indexOf(Import.csvPreset))
                        onPicked: (t) => Import.csvPreset = vals[labels.indexOf(t)]
                    }
                }
            }

            // optional background image → a real max-projection in the figure
            InfoRow {
                icon: "image"
                label: "Background"
                value: Import.bgImagePath || "(optional — none)"
                placeholder: !Import.bgImagePath
                onBrowse: Import.browseBgImage()
            }

            Text {
                Layout.fillWidth: true
                text: "Localisation table — detection & linking are skipped. Set the "
                    + "source format if auto-detect fails; a background image gives the "
                    + "figure a real max-projection instead of a blank canvas."
                wrapMode: Text.WordWrap
                color: pal.TXT_MUTED; font.pixelSize: sc.textSm
            }
        }

        // ── metadata stat strip ──────────────────────────────────────
        GridLayout {
            visible: !root.batchMode
            Layout.fillWidth: true
            columns: width < 560 ? 2 : 4
            columnSpacing: sc.sp4; rowSpacing: sc.sp4
            StatCard { icon: "image";    label: "FRAMES"; enterDelay: 0
                       value: "—"
                       numValue: Import.frameCount > 0 ? Import.frameCount : -1 }
            StatCard { icon: "move";     label: "PIXEL"; enterDelay: 35
                       numValue: Math.round(Import.pixelSize * 1000); numSuffix: " nm" }
            StatCard { icon: "clock";    label: "EXPOSURE"; enterDelay: 70
                       numValue: Math.round(Import.frameInterval * 1000); numSuffix: " ms" }
            StatCard { icon: "database"; label: "SIZE"; enterDelay: 105
                       value: Import.fileSize || "—" }
        }

        // ── detection preview ─────────────────────────────────────────
        Card {
            visible: !root.batchMode
            Layout.fillWidth: true
            Layout.preferredHeight: previewCol.implicitHeight + sc.sp5 * 2
            ColumnLayout {
                id: previewCol
                x: sc.sp5; y: sc.sp5
                width: parent.width - sc.sp5 * 2
                spacing: sc.sp4
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp3
                    Icon { name: "scan-search"; size: 15; color: pal.ACC }
                    Text { text: "Detection preview"; color: pal.TXT
                           font.pixelSize: sc.textSm; font.bold: true }
                    Item { Layout.fillWidth: true }
                    Badge { text: Process.running ? "Live" : "Idle"
                            tone: Process.running ? pal.SUCCESS : pal.TXT_MUTED
                            dot: true }
                }
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp5
                    Rectangle {
                        id: thumbBox
                        // Width tracks the frame's aspect ratio (height fixed) so a
                        // square recording fills the box edge-to-edge with no black
                        // letterbox bars; falls back to the original 8:5 box for the
                        // empty-state placeholder.
                        readonly property real ar:
                            (thumbImg.implicitWidth > 0 && thumbImg.implicitHeight > 0)
                                ? thumbImg.implicitWidth / thumbImg.implicitHeight : 1.6
                        Layout.preferredHeight: 150
                        Layout.preferredWidth: Math.round(Math.max(120, Math.min(300, 150 * ar)))
                        radius: sc.radiusMd; color: pal.WELL
                        border.width: 1; border.color: pal.BORDER; clip: true
                        readonly property bool showLive: Process.running && Process.hasLiveFrame
                        Image {
                            id: thumbImg
                            anchors.fill: parent; anchors.margins: 2
                            visible: thumbBox.showLive || Import.hasThumb
                            fillMode: Image.PreserveAspectFit
                            smooth: false; cache: false; asynchronous: true
                            source: thumbBox.showLive
                                    ? ("image://liveframe/" + Process.frameToken)
                                    : (Import.hasThumb ? ("image://importthumb/" + Import.thumbToken) : "")
                            // Fade the max-proj thumbnail in once it's ready; the
                            // live frame stays at full opacity (never animate it).
                            opacity: thumbBox.showLive ? 1 : (status === Image.Ready ? 1 : 0)
                            Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200; easing.type: Easing.OutCubic } }
                        }
                        Text {
                            anchors.centerIn: parent
                            visible: !thumbBox.showLive && !Import.hasThumb
                            width: parent.width - sc.sp6 * 2
                            horizontalAlignment: Text.AlignHCenter; wrapMode: Text.WordWrap
                            text: Import.hasFile ? "No preview for this file type"
                                                 : "Pick a recording to preview it here"
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                        }
                        Rectangle {                          // corner status chip
                            visible: thumbBox.showLive || Import.hasThumb
                            anchors { left: parent.left; bottom: parent.bottom; margins: sc.sp2 }
                            radius: sc.radiusXs; color: Qt.rgba(0, 0, 0, 0.6)
                            width: flbl.implicitWidth + sc.sp3 * 2
                            height: flbl.implicitHeight + sc.sp1 * 2
                            Text { id: flbl; anchors.centerIn: parent
                                   text: thumbBox.showLive ? "live" : "max proj"
                                   color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo" }
                        }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true; spacing: sc.sp4
                        Text {
                            Layout.fillWidth: true; wrapMode: Text.WordWrap
                            text: "A max-intensity projection of the recording. Open the "
                                + "preview to see it at full size and draw a region of "
                                + "interest before you run."
                            color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                        }
                        RowLayout {
                            spacing: sc.sp3
                            Button {
                                variant: "secondary"; text: "Open preview & ROI"; icon: "scan-search"
                                enabled: Import.hasFile && !Import.isCsv
                                onClicked: Roi.editFile(Import.filePath)
                            }
                            Button {
                                variant: "secondary"; text: "Load run manifest"; icon: "rotate-ccw"
                                onClicked: Sidebar.loadManifest()
                            }
                            Button {
                                variant: "secondary"; text: "Process"; icon: "arrow-up-right"
                                onClicked: App.setTab(1)
                            }
                        }
                    }
                }
            }
        }

        // ══════════════════════ BATCH ANALYSIS ═══════════════════════════
        InfoRow {
            visible: root.batchMode
            icon: "layers"
            label: "Source"
            value: Batch.folder || "No folder selected"
            placeholder: !Batch.folder
            badgeText: Batch.series.length > 0 ? (Batch.series.length + " found") : ""
            onBrowse: Batch.browseFolder()
        }
        InfoRow {
            visible: root.batchMode
            icon: "folder-open"; iconTint: pal.SUCCESS
            label: "Output"
            value: Batch.outputDir || (Batch.folder ? "(source folder) / batch_results" : "—")
            placeholder: !Batch.outputDir
            badgeText: ""
            badgeTone: pal.TXT_MUTED
            onBrowse: Batch.browseOutputDir()
        }
        RowLayout {
            visible: root.batchMode
            spacing: sc.sp3
            Switch { checked: Batch.recursive; onToggled: (c) => Batch.recursive = c }
            Text { text: "Include subfolders"; color: pal.TXT
                   font.pixelSize: sc.textSm; Layout.alignment: Qt.AlignVCenter }
        }

        // ── run queue ─────────────────────────────────────────────────
        Card {
            id: queueCard
            visible: root.batchMode
            Layout.fillWidth: true
            Layout.preferredHeight: queueCol.implicitHeight + sc.sp5 * 2
            // accent the whole queue while a folder/file is dragged over it
            border.color: queueDrop.containsDrag ? pal.ACC : pal.BORDER
            Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }

            ColumnLayout {
                id: queueCol
                x: sc.sp5; y: sc.sp5
                width: parent.width - sc.sp5 * 2
                spacing: sc.sp4

                // header: title · counts · expand/select · add
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp3
                    Icon { name: "list-checks"; size: 15; color: pal.ACC }
                    Text { text: "Run queue"; color: pal.TXT; font.pixelSize: sc.textSm; font.bold: true }
                    Badge {
                        visible: Batch.seriesCount > 0
                        text: Batch.seriesCount + " series · " + Batch.fileCountTotal + " files"
                        tone: pal.ACC
                    }
                    Item { Layout.fillWidth: true }
                    LinkBtn {
                        visible: Batch.seriesCount > 0
                        text: Batch.allExpanded ? "Collapse all" : "Expand all"
                        onClicked: Batch.expandAll(!Batch.allExpanded)
                    }
                    LinkBtn {
                        visible: Batch.seriesCount > 0
                        text: Batch.allFilesSelected ? "Deselect all" : "Select all"
                        onClicked: Batch.selectAll(!Batch.allFilesSelected)
                    }
                    Button { variant: "secondary"; text: "Add folder"; icon: "folder-plus"
                             onClicked: Batch.addFolder() }
                    Button { variant: "secondary"; text: "Add files"; icon: "file-plus"
                             onClicked: Batch.addFiles() }
                }

                // drag-and-drop hint zone
                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    DashedRect {
                        anchors.fill: parent
                        stroke: queueDrop.containsDrag ? pal.ACC : pal.BORDER_HI
                    }
                    Rectangle {
                        anchors.fill: parent; radius: sc.radiusXl
                        color: queueDrop.containsDrag ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.06) : "transparent"
                    }
                    RowLayout {
                        anchors.centerIn: parent; spacing: sc.sp3
                        Icon { name: "upload"; size: 16
                               color: queueDrop.containsDrag ? pal.ACC : pal.TXT_MUTED }
                        Text { text: "Drag folders or .tif / .czi files here to add them to the queue"
                               color: queueDrop.containsDrag ? pal.ACC : pal.TXT_MUTED
                               font.pixelSize: sc.textSm }
                    }
                }

                Text {
                    visible: Batch.seriesCount === 0
                    text: Batch.folder ? "No analysable files found in this folder."
                                       : "Pick a source folder, add files, or drop them above."
                    color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                }

                // the series list (stable delegates → smooth expand / badges)
                Repeater {
                    model: Batch.seriesModel
                    delegate: SeriesRow {
                        required property var model
                        required property int index
                        item: model
                        rowIndex: index
                    }
                }

                // footer: selection summary + Clear
                RowLayout {
                    visible: Batch.seriesCount > 0
                    Layout.fillWidth: true; Layout.topMargin: sc.sp1; spacing: sc.sp3
                    Text {
                        textFormat: Text.StyledText
                        text: "<b>" + Batch.selectedFileCount + "</b> of " + Batch.fileCountTotal
                              + " files selected · <b>" + Batch.seriesCount + "</b> series"
                        color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                    }
                    Item { Layout.fillWidth: true }
                    Button { variant: "secondary"; text: "Clear"; icon: "rotate-ccw"
                             enabled: !Batch.running; onClicked: Batch.clear() }
                }

                Text {
                    visible: Batch.status !== ""
                    text: Batch.status; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                }
            }

            // queue-wide drop target (folders + files appended via addPaths)
            DropArea {
                id: queueDrop
                anchors.fill: parent
                onDropped: (drop) => {
                    if (drop.hasUrls && drop.urls.length > 0) {
                        var us = []
                        for (var i = 0; i < drop.urls.length; ++i) us.push("" + drop.urls[i])
                        Batch.addPaths(us)
                    }
                }
            }
        }
        Alert { visible: root.batchMode && Batch.generateError !== ""; Layout.fillWidth: true
                severity: "warn"; text: Batch.generateError }

        // ── start / stop ──────────────────────────────────────────────
        RowLayout {
            Layout.topMargin: sc.sp2
            spacing: sc.sp4
            Button {
                visible: !root.batchMode
                variant: "primary"; text: "Start analysis"; icon: "play"
                enabled: Import.hasFile && !Process.running
                onClicked: { App.setTab(1); Process.start(); }
            }
            Button {
                visible: root.batchMode
                variant: Batch.running ? "danger" : "primary"
                text: Batch.running ? "Stop batch" : "Start batch"
                icon: Batch.running ? "x" : "play"
                enabled: Batch.running || Batch.canRun
                onClicked: Batch.running ? Batch.stop() : Batch.generate()
            }
            Text {
                text: root.batchMode
                      ? (Batch.canRun || Batch.running ? "Processes the queued series."
                                                       : "Pick a folder and queue series.")
                      : (Import.hasFile ? "Runs on the Analysis tab."
                                        : "Pick an input file to begin.")
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                Layout.alignment: Qt.AlignVCenter
            }
        }
    }
}
