import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "../components"

// HYPER-FLY dashboard (tab 4): the live view of a parallel-batch run. A command
// bar (overall progress + elapsed/throughput/eta), a machine resource strip
// (CPU/RAM/GPU/VRAM, reused from the always-on Process monitor), and a grid of
// fixed worker tiles — files flow through `n_concurrent` stable slots, each a
// mini cockpit (live frame + stage + locs + progress). Bound to
// `Hyperfly` (the parallel dashboard model) + `Batch` (run control) + `Process`
// (resource meters). Populated only during a real HYPER-FLY batch run.
Flickable {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property string amber: "#f6a623"
    contentWidth: width
    contentHeight: col.implicitHeight + 56
    clip: true

    function resColor(v) { return v > 95 ? pal.DANGER : v > 82 ? pal.WARN : pal.ACC }

    // ── reusable: resource meter ──────────────────────────────────────────
    component Meter: RowLayout {
        property string label: ""
        property string iconName: ""
        property real value: -1            // ≥0 → bar; <0 → text
        property string valueText: ""
        property bool gpu: false
        readonly property bool barMode: value >= 0
        Layout.fillWidth: true
        spacing: sc.sp3
        Icon { name: iconName; size: 14; color: pal.TXT_MUTED }
        Text { text: label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.preferredWidth: 42 }
        Rectangle {
            visible: barMode
            Layout.fillWidth: true; implicitHeight: 6; radius: 3
            color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER; clip: true
            Rectangle {
                height: parent.height; radius: 3
                width: Math.max(0, Math.min(1, value / 100)) * parent.width
                color: gpu ? pal.SUCCESS : root.resColor(value)
                Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 900; easing.type: Easing.Linear } }
                Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 300 } }
            }
        }
        Text { visible: barMode; text: Math.round(value) + "%"
               color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo"
               Layout.preferredWidth: 38; horizontalAlignment: Text.AlignRight }
        Text { visible: !barMode; text: valueText
               color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
               Layout.fillWidth: true; horizontalAlignment: Text.AlignRight }
    }

    // ── reusable: worker tile ─────────────────────────────────────────────
    component WorkerTile: Rectangle {
        id: tile
        property var item
        property int idx: 0
        readonly property string st: item ? item.state : "idle"
        Layout.fillWidth: true
        Layout.preferredHeight: 188
        radius: sc.radiusLg
        clip: true
        color: pal.PANEL
        border.width: 1
        border.color: st === "running" ? Qt.rgba(0.345, 0.651, 1.0, 0.5)
                    : st === "done"    ? Qt.rgba(0.337, 0.827, 0.392, 0.45)
                    : st === "failed"  ? pal.DANGER : pal.BORDER
        Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 250 } }

        // RAM-staggered spawn (fade + rise, i·90 ms apart)
        opacity: 0
        transform: Translate { id: tT; y: 10 }
        Component.onCompleted: tEnter.start()
        SequentialAnimation {
            id: tEnter
            PauseAnimation { duration: Theme.reducedMotion ? 0 : Math.min(tile.idx, 15) * 90 }
            ParallelAnimation {
                NumberAnimation { target: tile; property: "opacity"; to: 1
                                  duration: Theme.reducedMotion ? 0 : 300; easing.type: Easing.OutCubic }
                NumberAnimation { target: tT; property: "y"; to: 0
                                  duration: Theme.reducedMotion ? 0 : 300; easing.type: Easing.OutCubic }
            }
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            // ── thumb: live frame + scan line + badges ──
            Rectangle {
                id: thumb
                Layout.fillWidth: true
                Layout.preferredHeight: 104
                color: pal.WELL
                clip: true
                Image {
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectCrop
                    smooth: false; cache: false; asynchronous: true
                    visible: tile.item && tile.item.hasFrame
                    opacity: tile.st === "idle" ? 0.12 : 0.92
                    source: (tile.item && tile.item.hasFrame)
                            ? ("image://hfworker/" + tile.idx + "_" + tile.item.frameToken) : ""
                }
                // status badge (top-left)
                Rectangle {
                    anchors { left: parent.left; top: parent.top; margins: sc.sp2 }
                    radius: sc.radiusPill; color: Qt.rgba(0, 0, 0, 0.55)
                    height: 19; width: bRow.implicitWidth + sc.sp3 * 2
                    RowLayout {
                        id: bRow; anchors.centerIn: parent; spacing: sc.sp1
                        StatusDot { tone: tile.st === "done" ? pal.SUCCESS
                                          : tile.st === "failed" ? pal.DANGER
                                          : tile.st === "running" ? pal.ACC : pal.TXT_MUTED
                                    pulsing: tile.st === "running" }
                        Text {
                            text: tile.st === "running" ? "Running" : tile.st === "done" ? "Done"
                                : tile.st === "failed" ? "Failed" : "Idle"
                            color: tile.st === "done" ? pal.SUCCESS : tile.st === "failed" ? pal.DANGER
                                 : tile.st === "running" ? pal.ACC : pal.TXT_MUTED
                            font.pixelSize: 10; font.bold: true
                        }
                    }
                }
                // worker id (top-right)
                Rectangle {
                    anchors { right: parent.right; top: parent.top; margins: sc.sp2 }
                    radius: sc.radiusXs; color: Qt.rgba(0, 0, 0, 0.5)
                    height: 15; width: widLbl.implicitWidth + sc.sp2 * 2
                    Text { id: widLbl; anchors.centerIn: parent
                           text: "W" + (tile.item ? tile.item.slot : tile.idx + 1)
                           color: pal.TXT_MUTED; font.pixelSize: 9; font.family: "Menlo" }
                }
                // done check (bottom-right, pops in)
                Pop {
                    anchors { right: parent.right; bottom: parent.bottom; margins: sc.sp2 }
                    visible: tile.st === "done"
                    Rectangle {
                        width: 18; height: 18; radius: 9; color: pal.SUCCESS
                        Icon { anchors.centerIn: parent; name: "check"; size: 11; color: "#06140a" }
                    }
                }
                Text {
                    anchors.fill: parent
                    anchors.margins: sc.sp4
                    visible: !(tile.item && tile.item.hasFrame)
                    horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                    wrapMode: Text.WrapAtWordBoundaryOrAnywhere; elide: Text.ElideRight
                    maximumLineCount: 4
                    text: tile.st === "failed"
                          ? ((tile.item && tile.item.error) ? tile.item.error : "failed")
                          : tile.st === "idle" ? "idle" : "…"
                    color: tile.st === "failed" ? pal.DANGER : pal.TXT_MUTED
                    font.pixelSize: tile.st === "failed" ? 10 : sc.textXs
                    font.family: tile.st === "failed" ? "Menlo" : Qt.application.font.family
                }
            }

            // ── info: filename · stage/locs · progress ──
            ColumnLayout {
                Layout.fillWidth: true
                Layout.margins: sc.sp3
                spacing: sc.sp2
                Text {
                    text: (tile.item && tile.item.stem) ? tile.item.stem : "—"
                    color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo"
                    elide: Text.ElideMiddle; Layout.fillWidth: true
                }
                RowLayout {
                    Layout.fillWidth: true
                    Text {
                        text: tile.st === "done" ? "done" : tile.st === "failed" ? "failed"
                            : (tile.item && tile.item.stage) ? tile.item.stage + "…" : "—"
                        color: tile.st === "done" ? pal.SUCCESS : tile.st === "failed" ? pal.DANGER : pal.ACC
                        font.pixelSize: 10; font.family: "Menlo"
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        visible: tile.item && tile.item.locs > 0
                        text: tile.item ? (tile.item.locs.toLocaleString(Qt.locale(), "f", 0) + " locs") : ""
                        color: pal.TXT_MUTED; font.pixelSize: 10; font.family: "Menlo"
                    }
                }
                Rectangle {
                    Layout.fillWidth: true; implicitHeight: 5; radius: 3
                    color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER; clip: true
                    Rectangle {
                        height: parent.height; radius: 3
                        width: Math.max(0, Math.min(1, (tile.item ? tile.item.pct : 0) / 100)) * parent.width
                        color: tile.st === "done" ? pal.SUCCESS : pal.ACC
                        Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 250; easing.type: Easing.OutCubic } }
                        IndeterminateShimmer { active: tile.st === "running" }
                    }
                }
            }
        }
    }

    ColumnLayout {
        id: col
        x: 28; y: 20
        width: root.width - 56
        spacing: sc.sp5

        // ── command bar ───────────────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 64
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                spacing: sc.sp5
                Icon { name: "zap"; size: 18; color: root.amber }
                Text { text: "HYPER-FLY"; color: root.amber; font.pixelSize: sc.textLg
                       font.bold: true; font.letterSpacing: 0.5 }
                Rectangle {
                    visible: Hyperfly.workerCount > 0
                    radius: sc.radiusPill; height: 22
                    width: wpRow.implicitWidth + sc.sp4 * 2
                    color: Qt.rgba(0.965, 0.651, 0.137, 0.14)
                    border.width: 1; border.color: Qt.rgba(0.965, 0.651, 0.137, 0.32)
                    RowLayout { id: wpRow; anchors.centerIn: parent; spacing: sc.sp1
                        Icon { name: "cpu"; size: 12; color: root.amber }
                        Text { text: Hyperfly.workerCount + " workers"; color: root.amber
                               font.pixelSize: sc.textXs; font.bold: true } }
                }
                // overall progress
                ColumnLayout {
                    Layout.fillWidth: true; Layout.preferredWidth: 160; spacing: sc.sp1
                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "Overall progress"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: Hyperfly.done + " / " + Hyperfly.total + " series · " + Hyperfly.overallPct + "%"
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                        }
                    }
                    Rectangle {
                        Layout.fillWidth: true; implicitHeight: 8; radius: 4
                        color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER; clip: true
                        Rectangle {
                            height: parent.height; radius: 4
                            width: Math.max(0, Math.min(1, Hyperfly.overallPct / 100)) * parent.width
                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0.0; color: pal.ACC }
                                GradientStop { position: 1.0; color: pal.ACC_HOVER }
                            }
                            Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 400; easing.type: Easing.OutCubic } }
                            IndeterminateShimmer { active: Batch.running }
                        }
                    }
                }
                component Stat: ColumnLayout {
                    property string k: ""
                    property string v: ""
                    spacing: 1
                    Text { text: k; color: pal.TXT_MUTED; font.pixelSize: 9; font.bold: true
                           font.letterSpacing: 0.5; Layout.alignment: Qt.AlignHCenter }
                    Text { text: v; color: pal.TXT; font.pixelSize: sc.textMd; font.bold: true
                           font.family: "Menlo"; Layout.alignment: Qt.AlignHCenter }
                }
                Stat { k: "ELAPSED";    v: Hyperfly.elapsed }
                Stat { k: "THROUGHPUT"; v: Hyperfly.throughput }
                Stat { k: "ETA";        v: Hyperfly.eta }
                Button {
                    variant: "danger"; text: "Stop"; icon: "x"
                    visible: Batch.running
                    onClicked: Batch.stop()
                }
            }
        }

        // ── resource strip ────────────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 48
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                spacing: sc.sp6
                Meter { label: "CPU";  iconName: "cpu";          value: Process.cpuPercent }
                Meter { label: "RAM";  iconName: "memory-stick"; value: Process.memPercent }
                Meter { label: "GPU";  iconName: "zap";          value: Process.gpuPercent
                        valueText: Process.gpuText; gpu: true }
                Meter { label: "VRAM"; iconName: "database";     value: -1; valueText: Process.vramText }
            }
        }

        // ── workers head ──────────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true; Layout.topMargin: sc.sp1; spacing: sc.sp3
            Text { text: "WORKERS"; color: pal.TXT_MUTED; font.pixelSize: 11
                   font.bold: true; font.letterSpacing: 1.0 }
            Badge { visible: Hyperfly.workerCount > 0
                    text: Hyperfly.runningCount + " active"; tone: pal.ACC }
        }

        // ── worker grid ───────────────────────────────────────────────────
        GridLayout {
            Layout.fillWidth: true
            visible: Hyperfly.workerCount > 0
            columns: root.width < 560 ? 1 : root.width < 880 ? 2 : 4
            columnSpacing: sc.sp4; rowSpacing: sc.sp4
            Repeater {
                model: Hyperfly.workerModel
                delegate: WorkerTile {
                    required property var model
                    required property int index
                    item: model
                    idx: index
                }
            }
        }

        // ── idle / not-engaged placeholder ────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 132
            visible: Hyperfly.workerCount === 0
            ColumnLayout {
                anchors.centerIn: parent
                spacing: sc.sp3
                width: parent.width - sc.sp8 * 2
                Icon { name: "zap"; size: 26; color: root.amber; Layout.alignment: Qt.AlignHCenter }
                Text {
                    text: "HYPER-FLY isn't engaged"
                    color: pal.TXT; font.pixelSize: sc.textMd; font.bold: true
                    Layout.alignment: Qt.AlignHCenter
                }
                Text {
                    Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter; wrapMode: Text.WordWrap
                    text: "Run a folder batch on an eligible workstation (≥ 32 cores · ≥ 192 GB RAM) "
                        + "and the queue fans out across worker processes — each one appears here as a "
                        + "live tile. On smaller machines the batch runs one file at a time."
                    color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                }
            }
        }

        // ── footer ────────────────────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 46
            visible: Hyperfly.workerCount > 0
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                spacing: sc.sp6
                component Tally: RowLayout {
                    property color swatch: Theme.palette.TXT_MUTED
                    property string label: ""
                    property int n: 0
                    spacing: sc.sp2
                    Rectangle { width: 8; height: 8; radius: 2; color: swatch
                                Layout.alignment: Qt.AlignVCenter }
                    Text { text: n + " " + label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                }
                Tally { swatch: pal.SUCCESS;   label: "done";    n: Hyperfly.done }
                Tally { swatch: pal.ACC;       label: "running"; n: Hyperfly.runningCount }
                Tally { swatch: pal.TXT_MUTED; label: "queued";  n: Hyperfly.queuedCount }
                Item { Layout.fillWidth: true }
                Text {
                    text: "RAM-staggered loading keeps the box off swap"
                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                }
            }
        }

        // ── run console log ───────────────────────────────────────────────
        Card {
            Layout.fillWidth: true
            Layout.preferredHeight: 220
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: sc.sp4
                spacing: sc.sp2
                RowLayout {
                    Layout.fillWidth: true
                    Icon { name: "terminal"; size: 13; color: pal.TXT_MUTED }
                    Text { text: "Console"; color: pal.TXT; font.pixelSize: sc.textSm; font.bold: true }
                    Badge { visible: Batch.running; text: "live"; tone: pal.ACC; dot: true }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: "Clear"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                    onClicked: hfLog.text = "" }
                    }
                }
                ScrollView {
                    Layout.fillWidth: true; Layout.fillHeight: true; clip: true
                    TextArea {
                        id: hfLog
                        readOnly: true; wrapMode: TextEdit.NoWrap
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                        background: Rectangle { color: "transparent" }
                        text: ""
                        placeholderText: "Batch worker output appears here during a run…"
                    }
                }
            }
        }
    }

    // stream the batch worker's log into the HYPER-FLY console
    Connections {
        target: Batch
        function onLogLine(line) {
            var t = hfLog.text + (hfLog.text ? "\n" : "") + line
            if (t.length > 60000) t = t.substring(t.length - 60000)
            hfLog.text = t
            hfLog.cursorPosition = hfLog.length
        }
        function onRunningChanged() { if (Batch.running) hfLog.text = "" }
    }
}
