import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "../components"

// Analysis cockpit (Phase 3): a connected pipeline stepper, progress + elapsed,
// a live detection frame, a streaming mass histogram, CPU/RAM meters, and the
// run result. Bound to AnalysisController; Start/Stop drive a real worker run.
Flickable {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    contentWidth: width
    contentHeight: col.implicitHeight + 56
    clip: true

    // ── status-pill mapping ──────────────────────────────────────────────
    readonly property color statusTone:
        Process.running ? pal.ACC
        : Process.resultSeverity === "ok"    ? pal.SUCCESS
        : Process.resultSeverity === "error" ? pal.DANGER
        : Process.resultSeverity === "warn"  ? pal.WARN
        : pal.TXT_MUTED
    readonly property string statusText:
        Process.running ? "Running"
        : Process.resultSeverity === "ok"    ? "Complete"
        : Process.resultSeverity === "error" ? "Error"
        : Process.resultSeverity === "warn"  ? "Finished"
        : "Idle"

    ColumnLayout {
        id: col
        x: 28; y: 20
        width: root.width - 56
        spacing: sc.sp6

        // ── header: title · status · elapsed · Start/Stop ────────────────
        RowLayout {
            Layout.fillWidth: true
            spacing: sc.sp4
            Text { text: "Process"; color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true }
            StatusDot { tone: root.statusTone; pulsing: Process.running
                        Layout.alignment: Qt.AlignVCenter }
            Badge { text: root.statusText; tone: root.statusTone; dot: false
                    Layout.alignment: Qt.AlignVCenter }
            Item { Layout.fillWidth: true }
            ProgressRing {
                value: Process.progress / 100
                size: 30
                thickness: 3
                tone: root.statusTone
                visible: Process.running
                Layout.alignment: Qt.AlignVCenter
            }
            RowLayout {
                spacing: sc.sp2
                visible: Process.running || Process.elapsed !== "00:00"
                Icon { name: "clock"; size: 13; color: pal.TXT_MUTED }
                Text { text: Process.elapsed; color: pal.TXT_MUTED
                       font.pixelSize: sc.textSm; font.family: "Menlo" }
            }
            Button {
                variant: Process.running ? "danger" : "primary"
                text: Process.running ? "Stop" : "Start analysis"
                icon: Process.running ? "x" : "play"
                enabled: Process.running || Import.hasFile
                onClicked: Process.running ? Process.stop() : Process.start()
            }
        }

        // ── connected pipeline stepper ───────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 88
            RowLayout {
                id: stepper
                anchors.fill: parent
                anchors.leftMargin: sc.sp8
                anchors.rightMargin: sc.sp8
                spacing: 0
                Repeater {
                    model: Process.stages
                    delegate: RowLayout {
                        id: node
                        required property int index
                        required property string modelData
                        readonly property bool done: Process.complete || index < Process.stage
                        readonly property bool active: index === Process.stage && !Process.complete
                        Layout.fillWidth: index > 0
                        spacing: 0

                        // connector to the previous node — pinned to the circle's
                        // centre-line (the circle is a 30px item at the TOP of the
                        // node column, so its centre is 15px down), NOT the vertical
                        // centre of the circle+label column below it.
                        Rectangle {
                            visible: index > 0
                            Layout.fillWidth: true
                            Layout.preferredHeight: 2
                            Layout.alignment: Qt.AlignTop
                            Layout.topMargin: 14        // (30 − 2) / 2 → centre at y = 15
                            radius: 1
                            color: (Process.complete || index <= Process.stage) ? pal.SUCCESS
                                 : pal.BORDER
                        }

                        // node + label
                        ColumnLayout {
                            spacing: sc.sp2
                            Layout.alignment: Qt.AlignVCenter
                            Item {
                                Layout.alignment: Qt.AlignHCenter
                                implicitWidth: 30; implicitHeight: 30
                                // pulsing ring on the active node
                                Rectangle {
                                    anchors.centerIn: parent
                                    width: 30; height: 30; radius: 15
                                    color: "transparent"
                                    border.width: 2; border.color: pal.ACC
                                    visible: node.active
                                    opacity: 0.0
                                    SequentialAnimation on opacity {
                                        running: node.active && Process.running && !Theme.reducedMotion
                                        loops: Animation.Infinite
                                        NumberAnimation { from: 0.6; to: 0.0; duration: 1100; easing.type: Easing.OutQuad }
                                    }
                                    SequentialAnimation on scale {
                                        running: node.active && Process.running && !Theme.reducedMotion
                                        loops: Animation.Infinite
                                        NumberAnimation { from: 1.0; to: 1.7; duration: 1100; easing.type: Easing.OutQuad }
                                    }
                                }
                                Rectangle {
                                    anchors.centerIn: parent
                                    width: 28; height: 28; radius: 14
                                    color: node.done ? pal.SUCCESS : node.active ? pal.ACC : pal.PANEL_ALT
                                    border.width: 1
                                    border.color: node.done ? pal.SUCCESS : node.active ? pal.ACC : pal.BORDER
                                    Icon {
                                        anchors.centerIn: parent
                                        visible: node.done
                                        name: "check"; size: 16; color: pal.ACC_FG
                                    }
                                    Text {
                                        anchors.centerIn: parent
                                        visible: !node.done
                                        text: node.index + 1
                                        color: node.active ? pal.ACC_FG : pal.TXT_MUTED
                                        font.pixelSize: sc.textSm; font.bold: node.active
                                    }
                                }
                            }
                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: node.modelData
                                color: node.done || node.active ? pal.TXT : pal.TXT_MUTED
                                font.pixelSize: sc.textXs
                                font.bold: node.active
                            }
                        }
                    }
                }
            }
        }

        // ── progress bar + stage label ───────────────────────────────────
        ColumnLayout {
            Layout.fillWidth: true
            spacing: sc.sp2
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 8
                radius: 4
                color: pal.PANEL_ALT
                border.width: 1; border.color: pal.BORDER
                clip: true
                Rectangle {
                    id: progressFill
                    height: parent.height
                    width: Math.max(0, Math.min(1, Process.progress / 100)) * parent.width
                    radius: 4
                    clip: true
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: pal.SUCCESS }
                        GradientStop { position: 1.0; color: pal.ACC }
                    }
                    Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                    // sweeping "working" highlight while a run is in progress
                    IndeterminateShimmer { active: Process.running }
                }
                // completion flash — one SUCCESS pulse across the whole track
                Rectangle {
                    id: doneFlash
                    anchors.fill: parent
                    radius: 4; color: pal.SUCCESS; opacity: 0
                    SequentialAnimation {
                        id: progressFlash
                        NumberAnimation { target: doneFlash; property: "opacity"; to: 0.55
                                          duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic }
                        NumberAnimation { target: doneFlash; property: "opacity"; to: 0
                                          duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic }
                    }
                }
                Connections {
                    target: Process
                    function onRunningChanged() {
                        if (!Process.running && Process.resultSeverity === "ok") progressFlash.restart()
                    }
                }
            }
            Text {
                text: Process.progressText || (Process.running ? "Working…" : "Ready")
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
            }
        }

        // ── live frame + histogram ───────────────────────────────────────
        GridLayout {
            id: cockpit
            Layout.fillWidth: true
            columns: width < 720 ? 1 : 2
            columnSpacing: sc.sp6
            rowSpacing: sc.sp6

            // live detection view — spans both right-column rows (histogram +
            // meters) so the preview stays tall and boxy.
            Card {
                Layout.fillWidth: true
                Layout.rowSpan: cockpit.columns === 2 ? 2 : 1
                Layout.fillHeight: true
                Layout.preferredHeight: 280
                Layout.minimumHeight: 280
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: sc.sp4
                    spacing: sc.sp2
                    RowLayout {
                        Layout.fillWidth: true
                        Icon { name: "image"; size: 14; color: pal.TXT_MUTED }
                        Text { text: "Live detection"; color: pal.TXT; font.pixelSize: sc.textSm; font.bold: true }
                        Item { Layout.fillWidth: true }
                    }
                    // The bordered frame hugs the image's aspect ratio (contained
                    // in the available cell) so a square movie leaves no black
                    // letterbox bars inside the border.  Before a frame loads, the
                    // box fills the cell so the placeholder text sits comfortably.
                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Rectangle {
                            id: liveFrame
                            anchors.centerIn: parent
                            readonly property real ar:
                                (liveImg.implicitWidth > 0 && liveImg.implicitHeight > 0)
                                    ? liveImg.implicitWidth / liveImg.implicitHeight
                                    : (parent.height > 0 ? parent.width / parent.height : 1.0)
                            width: Math.max(1, Math.min(parent.width, parent.height * ar))
                            height: Math.max(1, Math.min(parent.height, parent.width / ar))
                            radius: sc.radiusMd
                            color: pal.WELL
                            border.width: 1; border.color: pal.BORDER
                            clip: true
                            Image {
                                id: liveImg
                                anchors.fill: parent
                                anchors.margins: 2
                                visible: Process.hasLiveFrame
                                fillMode: Image.PreserveAspectFit
                                smooth: false
                                cache: false
                                asynchronous: true
                                source: Process.hasLiveFrame
                                        ? ("image://liveframe/" + Process.frameToken) : ""
                            }
                            Text {
                                anchors.centerIn: parent
                                width: parent.width - sc.sp6
                                horizontalAlignment: Text.AlignHCenter
                                wrapMode: Text.WordWrap
                                visible: !Process.hasLiveFrame
                                text: Process.running ? "Waiting for first frame…"
                                                       : "Detection preview appears here while running"
                                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                            }
                        }
                    }
                }
            }

            // mass histogram (top of the right column)
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 280
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: sc.sp4
                    spacing: sc.sp2
                    RowLayout {
                        Layout.fillWidth: true
                        Icon { name: "zap"; size: 14; color: pal.TXT_MUTED }
                        Text { text: "Localisation mass"; color: pal.TXT; font.pixelSize: sc.textSm; font.bold: true }
                        Item { Layout.fillWidth: true }
                        RowLayout {
                            visible: hist.total > 0
                            spacing: sc.sp1
                            Odometer {
                                value: hist.total
                                digits: Math.max(3, String(hist.total).length)
                                pixelSize: sc.textXs
                            }
                            Text {
                                text: "spots"; color: pal.TXT_MUTED
                                font.pixelSize: sc.textXs
                                Layout.alignment: Qt.AlignVCenter
                            }
                        }
                    }
                    Canvas {
                        id: hist
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        property var bins: []
                        property int total: 0
                        property real vmin: 0
                        property real vmax: 1
                        property bool dirty: false

                        function addChunk(vals) {
                            if (!vals || vals.length === 0) return;
                            // Lazily fix the range from the first values, then bin
                            // incrementally so we never store the full stream.
                            if (bins.length === 0) {
                                var lo = vals[0], hi = vals[0];
                                for (var i = 0; i < vals.length; ++i) {
                                    if (vals[i] < lo) lo = vals[i];
                                    if (vals[i] > hi) hi = vals[i];
                                }
                                vmin = lo; vmax = (hi > lo ? hi : lo + 1);
                                var b = []; for (var k = 0; k < 48; ++k) b.push(0);
                                bins = b;
                            }
                            var span = vmax - vmin;
                            for (var j = 0; j < vals.length; ++j) {
                                var t = (vals[j] - vmin) / span;
                                var idx = Math.floor(t * 48);
                                if (idx < 0) idx = 0; if (idx > 47) idx = 47;
                                bins[idx] += 1; total += 1;
                            }
                            dirty = true;
                        }
                        function clearHist() { bins = []; total = 0; dirty = false; requestPaint(); }

                        Timer {
                            interval: 120; running: Process.running; repeat: true
                            onTriggered: if (hist.dirty) { hist.dirty = false; hist.requestPaint(); }
                        }

                        onPaint: {
                            var ctx = getContext("2d");
                            ctx.reset();
                            ctx.clearRect(0, 0, width, height);
                            if (bins.length === 0) {
                                ctx.fillStyle = pal.TXT_MUTED;
                                ctx.font = "11px sans-serif";
                                ctx.textAlign = "center";
                                ctx.fillText(Process.running ? "Accumulating localisations…"
                                                              : "Mass distribution appears here",
                                             width / 2, height / 2);
                                return;
                            }
                            var peak = 1;
                            for (var i = 0; i < bins.length; ++i) if (bins[i] > peak) peak = bins[i];
                            var pad = 6;
                            var w = (width - pad * 2) / bins.length;
                            for (var b = 0; b < bins.length; ++b) {
                                var h = (bins[b] / peak) * (height - pad * 2);
                                var x = pad + b * w;
                                var y = height - pad - h;
                                ctx.fillStyle = pal.ACC;
                                var bw = Math.max(1, w - 1.5);
                                ctx.fillRect(x, y, bw, h);
                            }
                            // dashed minmass threshold line
                            if (Process.minmassThreshold >= 0 && vmax > vmin) {
                                var tt = (Process.minmassThreshold - vmin) / (vmax - vmin);
                                if (tt >= 0 && tt <= 1) {
                                    var lx = pad + tt * (width - pad * 2);
                                    ctx.strokeStyle = pal.WARN;
                                    ctx.lineWidth = 1.5;
                                    ctx.setLineDash([4, 3]);
                                    ctx.beginPath(); ctx.moveTo(lx, pad); ctx.lineTo(lx, height - pad); ctx.stroke();
                                    ctx.setLineDash([]);
                                }
                            }
                        }
                    }
                }
            }

            // ── resource meters (right column, under the histogram) ───────
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: meters.implicitHeight + sc.sp5 * 2
                ColumnLayout {
                    id: meters
                    x: sc.sp5; y: sc.sp5
                    width: parent.width - sc.sp5 * 2
                    spacing: sc.sp4
                    component Meter: RowLayout {
                        property string label
                        property string iconName
                        property real value: -1       // 0..100 → progress bar; <0 → text mode
                        property string valueText: "" // shown instead of a bar when value < 0
                        readonly property bool barMode: value >= 0
                        width: meters.width
                        spacing: sc.sp4
                        Icon { name: iconName; size: 15
                               color: !barMode ? pal.ACC : value > 95 ? pal.DANGER : value > 80 ? pal.WARN : pal.ACC }
                        Text { text: label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.preferredWidth: 44 }
                        // ── percent mode: gradient bar + % readout ──────────
                        Rectangle {
                            visible: barMode
                            Layout.fillWidth: true
                            implicitHeight: 8; radius: 4
                            color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                            Rectangle {
                                height: parent.height
                                width: Math.max(0, Math.min(1, value / 100)) * parent.width
                                radius: 4
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: value > 95 ? pal.DANGER : pal.ACC }
                                    GradientStop { position: 1.0; color: value > 95 ? pal.DANGER : value > 80 ? pal.WARN : pal.ACC_HOVER }
                                }
                                // 1 Hz samples interpolate smoothly (a GPU bar dropping to 0 is then visible)
                                Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 300; easing.type: Easing.OutCubic } }
                            }
                        }
                        Text {
                            visible: barMode
                            text: Math.round(value) + "%"
                            color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo"
                            Layout.preferredWidth: 38; horizontalAlignment: Text.AlignRight
                        }
                        // ── text mode: right-aligned label (e.g. "Unified") ─
                        Text {
                            visible: !barMode
                            Layout.fillWidth: true
                            text: valueText
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                            horizontalAlignment: Text.AlignRight
                        }
                    }
                    Meter { label: "CPU";  iconName: "cpu";          value: Process.cpuPercent }
                    Meter { label: "RAM";  iconName: "memory-stick"; value: Process.memPercent }
                    Meter { label: "GPU";  iconName: "zap";          value: Process.gpuPercent
                            valueText: Process.gpuText }
                    Meter { label: "VRAM"; iconName: "database";     value: -1
                            valueText: Process.vramText }
                }
            }
        }

        // ── result summary ───────────────────────────────────────────────
        // Fades + rises in when a run finishes (the "done" cue), out when cleared.
        Card {
            Layout.fillWidth: true
            readonly property bool present: Process.resultHeadline !== ""
            visible: present || opacity > 0.001
            opacity: present ? 1 : 0
            transform: Translate {
                y: present ? 0 : 8
                Behavior on y { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
            }
            Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
            Layout.preferredHeight: resCol.implicitHeight + sc.sp5 * 2
            ColumnLayout {
                id: resCol
                x: sc.sp5; y: sc.sp5
                width: parent.width - sc.sp5 * 2
                spacing: sc.sp3
                RowLayout {
                    Layout.fillWidth: true
                    Item {
                        Layout.preferredWidth: 16; Layout.preferredHeight: 16
                        Layout.alignment: Qt.AlignVCenter
                        CheckDraw {
                            anchors.centerIn: parent
                            size: 16
                            tone: pal.SUCCESS
                            visible: Process.resultSeverity === "ok"
                            on: Process.resultSeverity === "ok"
                        }
                        Icon {
                            anchors.centerIn: parent
                            visible: Process.resultSeverity !== "ok"
                            name: Process.resultSeverity === "error" ? "triangle-alert" : "info"
                            size: 16; color: root.statusTone
                        }
                    }
                    Text {
                        text: Process.resultHeadline
                        color: pal.TXT; font.pixelSize: sc.textMd; font.bold: true
                        Layout.fillWidth: true; elide: Text.ElideRight
                    }
                }
                // ── post-analysis summary stats (animated; on a successful run) ──
                // A Loader so the metric grid is re-instantiated each completion —
                // that replays the staggered FadeRise entrance + the count-up
                // (CountUp tweens 0 → value once `reveal` flips after layout).
                Loader {
                    id: statsLoader
                    Layout.fillWidth: true
                    active: Process.resultSeverity === "ok"
                    visible: active
                    Layout.preferredHeight: (active && item) ? item.implicitHeight : 0
                    sourceComponent: Component {
                        ColumnLayout {
                            id: sg
                            width: statsLoader.width
                            spacing: sc.sp4
                            property bool reveal: false
                            Component.onCompleted: reveal = true

                            Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
                            Text { text: "Run summary"; color: pal.TXT_MUTED
                                   font.pixelSize: sc.textXs; font.bold: true }
                            GridLayout {
                                Layout.fillWidth: true
                                columns: 4
                                columnSpacing: sc.sp6; rowSpacing: sc.sp5
                                Repeater {
                                    model: [
                                        { label: "Trajectories",    key: "n_tracks",            dec: 0, unit: "",      mul: 1, hero: true },
                                        { label: "Localisations",   key: "n_locs",              dec: 0, unit: "",      mul: 1, hero: true },
                                        { label: "Median D",        key: "median_d",            dec: 3, unit: "µm²/s", mul: 1, hero: false },
                                        { label: "Median α",        key: "median_alpha",        dec: 2, unit: "",      mul: 1, hero: false },
                                        { label: "Mobile fraction", key: "mobile_fraction",     dec: 0, unit: "%",     mul: 100, hero: false },
                                        { label: "Loc precision",   key: "median_loc_sigma_nm", dec: 0, unit: "nm",    mul: 1, hero: false },
                                        { label: "Clusters",        key: "n_clusters",          dec: 0, unit: "",      mul: 1, hero: false },
                                        { label: "Frames",          key: "frames",              dec: 0, unit: "",      mul: 1, hero: false }
                                    ]
                                    delegate: FadeRise {
                                        required property var modelData
                                        required property int index
                                        Layout.fillWidth: true
                                        delay: index * 55
                                        ColumnLayout {
                                            id: cell
                                            spacing: 2
                                            readonly property var raw:
                                                (Process.stats && Process.stats[modelData.key] !== undefined
                                                 && Process.stats[modelData.key] !== null)
                                                    ? Process.stats[modelData.key] : null
                                            Text { text: modelData.label; color: pal.TXT_MUTED
                                                   font.pixelSize: sc.textXs }
                                            RowLayout {
                                                spacing: sc.sp1
                                                CountUp {
                                                    visible: cell.raw !== null
                                                    value: (sg.reveal && cell.raw !== null) ? cell.raw * modelData.mul : 0
                                                    decimals: modelData.dec
                                                    color: modelData.hero ? pal.ACC : pal.TXT
                                                    font.pixelSize: sc.textXl; font.bold: true; font.family: "Menlo"
                                                }
                                                Text { visible: cell.raw === null; text: "—"
                                                       color: pal.TXT_MUTED; font.pixelSize: sc.textXl; font.bold: true }
                                                Text { visible: modelData.unit !== "" && cell.raw !== null
                                                       text: modelData.unit; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                                       Layout.alignment: Qt.AlignBottom; bottomPadding: 3 }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    visible: Process.resultOutDir !== ""
                    spacing: sc.sp4
                    Text {
                        text: Process.resultOutDir
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                        elide: Text.ElideMiddle; Layout.fillWidth: true
                    }
                    Button {
                        variant: "secondary"; text: "Open folder"; icon: "folder-open"
                        onClicked: Qt.openUrlExternally("file://" + Process.resultOutDir)
                    }
                }
            }
        }

        // ── batch queue (only while a serial batch mirrors this cockpit) ──
        // Compact, at-a-glance progress through the queued series — the Process
        // screen otherwise shows only the current file.
        Card {
            Layout.fillWidth: true
            visible: Batch.running
            Layout.preferredHeight: bqCol.implicitHeight + sc.sp4 * 2
            ColumnLayout {
                id: bqCol
                x: sc.sp4; y: sc.sp4
                width: parent.width - sc.sp4 * 2
                spacing: sc.sp3
                RowLayout {
                    Layout.fillWidth: true
                    Icon { name: "layers"; size: 13; color: pal.TXT_MUTED }
                    Text { text: "Batch queue"; color: pal.TXT
                           font.pixelSize: sc.textSm; font.bold: true }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: Batch.filesDone + " / " + Batch.filesTotal + " done"
                              + (Batch.filesFailed > 0 ? " · " + Batch.filesFailed + " failed" : "")
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                    }
                }
                // The list is capped + scrollable so a big batch stays compact.
                ListView {
                    id: bqList
                    Layout.fillWidth: true
                    Layout.preferredHeight: Math.min(contentHeight, 132)
                    interactive: contentHeight > height
                    clip: true
                    spacing: sc.sp2
                    model: Batch.runQueue
                    delegate: RowLayout {
                        required property var modelData
                        width: bqList.width
                        spacing: sc.sp3
                        Rectangle {
                            width: 8; height: 8; radius: 4
                            Layout.alignment: Qt.AlignVCenter
                            color: modelData.status === "done"    ? pal.SUCCESS
                                 : modelData.status === "error"   ? pal.DANGER
                                 : modelData.status === "running" ? pal.ACC
                                 : pal.BORDER
                        }
                        Text {
                            text: modelData.name
                            color: modelData.current ? pal.TXT : pal.TXT_MUTED
                            font.pixelSize: sc.textXs
                            font.bold: modelData.current
                            elide: Text.ElideRight
                            Layout.preferredWidth: 160
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignVCenter
                            implicitHeight: 4; radius: 2
                            color: pal.PANEL_ALT
                            clip: true
                            Rectangle {
                                height: parent.height; radius: 2
                                width: Math.max(0, Math.min(1, modelData.progress / 100)) * parent.width
                                color: modelData.status === "error" ? pal.DANGER
                                     : modelData.status === "done"  ? pal.SUCCESS : pal.ACC
                                Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                            }
                        }
                        Text {
                            visible: modelData.current
                            text: Math.round(modelData.progress) + "%"
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                            Layout.preferredWidth: 34; horizontalAlignment: Text.AlignRight
                        }
                    }
                }
            }
        }

        // ── compact run log ──────────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 160
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: sc.sp4
                spacing: sc.sp2
                RowLayout {
                    Layout.fillWidth: true
                    Icon { name: "info"; size: 13; color: pal.TXT_MUTED }
                    Text { text: "Log"; color: pal.TXT; font.pixelSize: sc.textSm; font.bold: true }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: "Clear"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                    onClicked: logArea.text = "" }
                    }
                }
                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    TextArea {
                        id: logArea
                        readOnly: true
                        wrapMode: TextEdit.NoWrap
                        color: pal.TXT_MUTED
                        font.pixelSize: sc.textXs
                        font.family: "Menlo"
                        background: Rectangle { color: "transparent" }
                        text: ""
                    }
                }
            }
        }
    }

    // ── controller signal wiring ─────────────────────────────────────────
    Connections {
        target: Process          // the run cockpit owns logLine/massChunk/running
        function onMassChunk(vals) { hist.addChunk(vals); }
        function onLogLine(line) {
            // Cap the log so a long run doesn't grow the document unbounded.
            var t = logArea.text + (logArea.text ? "\n" : "") + line;
            if (t.length > 60000) t = t.substring(t.length - 60000);
            logArea.text = t;
            logArea.cursorPosition = logArea.length;
        }
        function onRunningChanged() {
            if (Process.running) { hist.clearHist(); logArea.text = ""; }
        }
    }
}
