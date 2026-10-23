import QtQuick
import QtQuick.Layouts

// StageFocus — the pipeline shown one stage at a time: a step counter, the
// stage name rolling up into place with what that stage does underneath, and
// a slim step bar (one segment per stage) whose running segment shimmers.
// Terminal states take over the readout: Complete / Finished (no
// trajectories) / Stopped / Failed, with the run's headline underneath.
//
//   StageFocus { stages: Process.stages; stage: Process.stage
//                complete: Process.complete; running: Process.running
//                severity: Process.resultSeverity; headline: Process.resultHeadline }
Item {
    id: root
    property var stages: []
    property int stage: -1
    property bool complete: false
    property bool running: false
    property string severity: ""            // "" | ok | warn | error (last run)
    property string headline: ""

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property bool motion: !Theme.reducedMotion
    // What each stage covers, matching AnalysisController._index_for_msg.
    // Keyed by name so a change to the stage order can't mislabel them.
    readonly property var blurbs: ({ "Preprocess": "Loading and preparing the recording",
                                     "Detect": "Finding molecules in each frame",
                                     "Drift": "Correcting stage drift",
                                     "Link": "Linking spots into trajectories",
                                     "Diffuse": "Fitting diffusion",
                                     "Classify": "Classifying motion and saving results" })

    readonly property bool failed: !running && severity === "error"
    readonly property bool halted: !running && !complete && severity === "warn"   // stopped
    readonly property color tone: running ? pal.ACC
                                : failed ? pal.DANGER
                                : complete ? (severity === "warn" ? pal.WARN : pal.SUCCESS)
                                : halted ? pal.WARN
                                : pal.TXT_MUTED
    readonly property string stageName: stage >= 0 && stage < stages.length ? stages[stage] : ""
    readonly property string title: running ? (stageName || "Starting")
                                  : failed ? "Failed"
                                  : complete ? (severity === "warn" ? "Finished" : "Complete")
                                  : halted ? "Stopped"
                                  : "Ready"
    readonly property string blurb: running ? (stageName ? (blurbs[stageName] || "") : "Launching the analysis")
                                  : (failed || halted) ? (headline || title) + (stageName ? " · during " + stageName : "")
                                  : complete ? (headline || "All " + stages.length + " stages finished")
                                  : stages.length + " stages · press Start analysis"
    implicitHeight: 88

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: sc.sp8 + sc.sp2; anchors.rightMargin: sc.sp8 + sc.sp2
        spacing: 22

        // ── step counter ─────────────────────────────────────────────────
        Rectangle {
            Layout.alignment: Qt.AlignVCenter
            implicitWidth: 52; implicitHeight: 52; radius: 26
            color: root.severity === "" && !root.running ? pal.PANEL_ALT
                 : Qt.rgba(root.tone.r, root.tone.g, root.tone.b, 0.13)
            border.width: 1
            border.color: root.severity === "" && !root.running ? pal.BORDER : root.tone
            Behavior on color { ColorAnimation { duration: root.motion ? 300 : 0 } }
            Behavior on border.color { ColorAnimation { duration: root.motion ? 300 : 0 } }
            CheckDraw { anchors.centerIn: parent; size: 24; tone: root.tone
                        on: root.complete && !root.running }
            Icon { anchors.centerIn: parent; visible: root.failed && !root.complete
                   name: "x"; size: 22; color: root.tone }
            Row {
                anchors.centerIn: parent
                visible: !root.complete && !root.failed
                Text { id: num
                       text: root.stage >= 0 ? root.stage + 1 : "–"
                       color: root.running || root.halted ? root.tone : pal.TXT_MUTED
                       font.pixelSize: 22; font.bold: true }
                Text { text: "/" + root.stages.length; color: pal.TXT_MUTED
                       font.pixelSize: sc.textSm; anchors.baseline: num.baseline }
            }
        }

        // ── stage name (rolls up on change) + what it is doing ───────────
        Item {
            id: readout
            Layout.fillWidth: true; Layout.preferredWidth: 1
            Layout.alignment: Qt.AlignVCenter
            implicitHeight: 52
            clip: true
            property string shown: root.title
            property string leaving: ""
            Text {
                id: outgoing
                text: readout.leaving
                color: pal.TXT_MUTED; font.pixelSize: 22; font.bold: true
                opacity: 0
            }
            Text {
                id: incoming
                width: parent.width; elide: Text.ElideRight
                text: readout.shown
                color: root.running ? pal.TXT : root.severity === "" ? pal.TXT_MUTED : root.tone
                font.pixelSize: 22; font.bold: true
            }
            RowLayout {
                y: 31
                spacing: sc.sp3
                Text { Layout.maximumWidth: readout.width - 24; elide: Text.ElideRight
                       text: root.blurb; color: pal.TXT_MUTED; font.pixelSize: sc.textSm }
                LoadingDots { visible: root.running; tone: pal.ACC }
            }
            ParallelAnimation {
                id: roll
                NumberAnimation { target: outgoing; property: "y"; from: 0; to: -22; duration: 320; easing.type: Easing.InCubic }
                NumberAnimation { target: outgoing; property: "opacity"; from: 1; to: 0; duration: 320 }
                NumberAnimation { target: incoming; property: "y"; from: 22; to: 0; duration: 380; easing.type: Easing.OutCubic }
                NumberAnimation { target: incoming; property: "opacity"; from: 0; to: 1; duration: 380 }
            }
            Connections {
                target: root
                function onTitleChanged() {
                    readout.leaving = readout.shown
                    readout.shown = root.title
                    if (root.motion) roll.restart()
                }
            }
        }

        // ── step bar: one segment per stage ──────────────────────────────
        ColumnLayout {
            Layout.fillWidth: true; Layout.preferredWidth: 1.25
            Layout.alignment: Qt.AlignVCenter
            spacing: sc.sp3
            RowLayout {
                Layout.fillWidth: true; spacing: 5
                Repeater {
                    model: root.stages
                    Rectangle {
                        id: seg
                        required property int index
                        readonly property bool done: root.complete || index < root.stage
                        readonly property bool current: index === root.stage && !root.complete
                        Layout.fillWidth: true; Layout.preferredWidth: 1
                        implicitHeight: 6; radius: 3
                        clip: true
                        color: done ? pal.SUCCESS
                             : current ? (root.running ? Qt.rgba(root.tone.r, root.tone.g, root.tone.b, 0.35) : root.tone)
                             : pal.BORDER
                        Behavior on color { ColorAnimation { duration: root.motion ? 300 : 0 } }
                        Rectangle {
                            id: glint
                            visible: seg.current && root.running
                            width: parent.width * 0.45; height: parent.height; radius: 3
                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0.0; color: "transparent" }
                                GradientStop { position: 0.6; color: pal.ACC }
                                GradientStop { position: 1.0; color: "transparent" }
                            }
                            XAnimator on x {
                                running: glint.visible && root.motion; loops: Animation.Infinite
                                from: -glint.width; to: seg.width; duration: 1000
                            }
                        }
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true; spacing: 5
                Repeater {
                    model: root.stages
                    Text {
                        required property int index
                        required property string modelData
                        readonly property bool current: index === root.stage && !root.complete
                        Layout.fillWidth: true; Layout.preferredWidth: 1
                        horizontalAlignment: Text.AlignHCenter
                        elide: Text.ElideRight
                        text: modelData
                        font.pixelSize: 10; font.bold: current && root.running
                        color: current && (root.running || root.halted || root.failed) ? root.tone
                             : (root.complete || index < root.stage) ? pal.TXT : pal.TXT_MUTED
                    }
                }
            }
        }
    }
}
