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
        Analysis.running ? pal.ACC
        : Analysis.resultSeverity === "ok"    ? pal.SUCCESS
        : Analysis.resultSeverity === "error" ? pal.DANGER
        : Analysis.resultSeverity === "warn"  ? pal.WARN
        : pal.TXT_MUTED
    readonly property string statusText:
        Analysis.running ? "Running"
        : Analysis.resultSeverity === "ok"    ? "Complete"
        : Analysis.resultSeverity === "error" ? "Error"
        : Analysis.resultSeverity === "warn"  ? "Finished"
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
            Text { text: "Analysis"; color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true }
            Badge { text: root.statusText; tone: root.statusTone; dot: true
                    Layout.alignment: Qt.AlignVCenter }
            Item { Layout.fillWidth: true }
            RowLayout {
                spacing: sc.sp2
                visible: Analysis.running || Analysis.elapsed !== "00:00"
                Icon { name: "clock"; size: 13; color: pal.TXT_MUTED }
                Text { text: Analysis.elapsed; color: pal.TXT_MUTED
                       font.pixelSize: sc.textSm; font.family: "Menlo" }
            }
            Button {
                variant: Analysis.running ? "danger" : "primary"
                text: Analysis.running ? "Stop" : "Start analysis"
                icon: Analysis.running ? "x" : "play"
                enabled: Analysis.running || Import.hasFile
                onClicked: Analysis.running ? Analysis.stop() : Analysis.start()
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
                    model: Analysis.stages
                    delegate: RowLayout {
                        id: node
                        required property int index
                        required property string modelData
                        readonly property bool done: Analysis.complete || index < Analysis.stage
                        readonly property bool active: index === Analysis.stage && !Analysis.complete
                        Layout.fillWidth: index > 0
                        spacing: 0

                        // connector to the previous node
                        Rectangle {
                            visible: index > 0
                            Layout.fillWidth: true
                            Layout.preferredHeight: 2
                            radius: 1
                            color: (Analysis.complete || index <= Analysis.stage) ? pal.SUCCESS
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
                                        running: node.active && Analysis.running && !Theme.reducedMotion
                                        loops: Animation.Infinite
                                        NumberAnimation { from: 0.6; to: 0.0; duration: 1100; easing.type: Easing.OutQuad }
                                    }
                                    SequentialAnimation on scale {
                                        running: node.active && Analysis.running && !Theme.reducedMotion
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
                Rectangle {
                    height: parent.height
                    width: Math.max(0, Math.min(1, Analysis.progress / 100)) * parent.width
                    radius: 4
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: pal.SUCCESS }
                        GradientStop { position: 1.0; color: pal.ACC }
                    }
                    Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                }
            }
            Text {
                text: Analysis.progressText || (Analysis.running ? "Working…" : "Ready")
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
            }
        }

        // ── live frame + histogram ───────────────────────────────────────
        GridLayout {
            Layout.fillWidth: true
            columns: width < 720 ? 1 : 2
            columnSpacing: sc.sp6
            rowSpacing: sc.sp6

            // live detection view
            Card {
                Layout.fillWidth: true
                Layout.preferredHeight: 280
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
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: sc.radiusMd
                        color: "#05070a"
                        border.width: 1; border.color: pal.BORDER
                        clip: true
                        Image {
                            anchors.fill: parent
                            anchors.margins: 2
                            visible: Analysis.hasLiveFrame
                            fillMode: Image.PreserveAspectFit
                            smooth: false
                            cache: false
                            asynchronous: true
                            source: Analysis.hasLiveFrame
                                    ? ("image://liveframe/" + Analysis.frameToken) : ""
                        }
                        Text {
                            anchors.centerIn: parent
                            visible: !Analysis.hasLiveFrame
                            text: Analysis.running ? "Waiting for first frame…"
                                                   : "Detection preview appears here while running"
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                        }
                    }
                }
            }

            // mass histogram
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
                        Text {
                            text: hist.total > 0 ? hist.total.toLocaleString(Qt.locale(), "f", 0) + " spots" : ""
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs
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
                            interval: 120; running: Analysis.running; repeat: true
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
                                ctx.fillText(Analysis.running ? "Accumulating localisations…"
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
                            if (Analysis.minmassThreshold >= 0 && vmax > vmin) {
                                var tt = (Analysis.minmassThreshold - vmin) / (vmax - vmin);
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
        }

        // ── resource meters ──────────────────────────────────────────────
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
                    property real value         // 0..100
                    width: meters.width
                    spacing: sc.sp4
                    Icon { name: iconName; size: 15; color: value > 80 ? pal.WARN : pal.ACC }
                    Text { text: label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.preferredWidth: 44 }
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 8; radius: 4
                        color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                        Rectangle {
                            height: parent.height
                            width: Math.max(0, Math.min(1, value / 100)) * parent.width
                            radius: 4
                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0.0; color: pal.ACC }
                                GradientStop { position: 1.0; color: value > 80 ? pal.WARN : pal.ACC_HOVER }
                            }
                            Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200 } }
                        }
                    }
                    Text {
                        text: Math.round(value) + "%"
                        color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo"
                        Layout.preferredWidth: 38; horizontalAlignment: Text.AlignRight
                    }
                }
                Meter { label: "CPU"; iconName: "cpu"; value: Analysis.cpuPercent }
                Meter { label: "RAM"; iconName: "memory-stick"; value: Analysis.memPercent }
            }
        }

        // ── result summary ───────────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            visible: Analysis.resultHeadline !== ""
            Layout.preferredHeight: resCol.implicitHeight + sc.sp5 * 2
            ColumnLayout {
                id: resCol
                x: sc.sp5; y: sc.sp5
                width: parent.width - sc.sp5 * 2
                spacing: sc.sp3
                RowLayout {
                    Layout.fillWidth: true
                    Icon {
                        name: Analysis.resultSeverity === "ok" ? "circle-check"
                            : Analysis.resultSeverity === "error" ? "triangle-alert"
                            : "info"
                        size: 16; color: root.statusTone
                    }
                    Text {
                        text: Analysis.resultHeadline
                        color: pal.TXT; font.pixelSize: sc.textMd; font.bold: true
                        Layout.fillWidth: true; elide: Text.ElideRight
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    visible: Analysis.resultOutDir !== ""
                    spacing: sc.sp4
                    Text {
                        text: Analysis.resultOutDir
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                        elide: Text.ElideMiddle; Layout.fillWidth: true
                    }
                    Button {
                        variant: "secondary"; text: "Open folder"; icon: "folder-open"
                        onClicked: Qt.openUrlExternally("file://" + Analysis.resultOutDir)
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
        target: Analysis
        function onMassChunk(vals) { hist.addChunk(vals); }
        function onLogLine(line) {
            // Cap the log so a long run doesn't grow the document unbounded.
            var t = logArea.text + (logArea.text ? "\n" : "") + line;
            if (t.length > 60000) t = t.substring(t.length - 60000);
            logArea.text = t;
            logArea.cursorPosition = logArea.length;
        }
        function onRunningChanged() {
            if (Analysis.running) { hist.clearHist(); logArea.text = ""; }
        }
    }
}
