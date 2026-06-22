import QtQuick
import "components"

// Transparent HUD overlay (L3) hosted by a translucent QQuickWidget raised above
// the native FireflyViewer island.  Display-only (the whole widget is mouse-
// transparent so pan/zoom fall through to the viewer).  Glass track-count pill
// top-left + a floating track/cluster inspector card bottom-right, bound to the
// VisualiseController. Sized to the viewer rect by EmbedController.
Item {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property var insp: Vis.inspector

    // ── rounded frame outline ────────────────────────────────────────────
    // The native island is masked to a rounded rect by EmbedController; this
    // antialiased border traces the same edge so the viewer reads as a floating
    // card (matching the side-panel cards). Mouse-transparent like the HUD.
    Rectangle {
        anchors.fill: parent
        color: "transparent"
        radius: 14
        border.width: 1
        border.color: pal.BORDER_HI
    }

    // ── glass HUD pill (top-left) ────────────────────────────────────────
    Rectangle {
        x: sc.sp6; y: sc.sp6
        width: pillRow.implicitWidth + sc.sp5 * 2
        height: 38; radius: sc.radius2xl
        color: Qt.rgba(0.05, 0.07, 0.10, 0.62)
        border.width: 1; border.color: Qt.rgba(1, 1, 1, 0.10)
        visible: Vis.hasRun
        Row {
            id: pillRow
            anchors.centerIn: parent
            spacing: sc.sp2
            Icon { name: "waypoints"; size: 14; color: pal.ACC
                   anchors.verticalCenter: parent.verticalCenter }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: "FireflyViewer · " + Vis.hudTrackCount.toLocaleString(Qt.locale(), "f", 0) + " tracks"
                color: pal.TXT; font.pixelSize: sc.textSm; font.family: "Menlo"
            }
        }
    }

    // ── inspector card (bottom-right) ────────────────────────────────────
    Rectangle {
        id: card
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.rightMargin: sc.sp6
        anchors.bottomMargin: sc.sp6
        width: 190
        height: col.implicitHeight + sc.sp5 * 2
        radius: sc.radius2xl
        color: Qt.rgba(0.05, 0.07, 0.10, 0.74)
        border.width: 1; border.color: Qt.rgba(1, 1, 1, 0.10)
        // fade + scale in when a track/cluster is selected
        visible: Vis.inspectorVisible || opacity > 0.001
        opacity: Vis.inspectorVisible ? 1 : 0
        scale:   Vis.inspectorVisible ? 1 : 0.95
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 180; easing.type: Easing.OutCubic } }
        Behavior on scale   { NumberAnimation { duration: Theme.reducedMotion ? 0 : 180; easing.type: Easing.OutCubic } }

        Column {
            id: col
            anchors.fill: parent
            anchors.margins: sc.sp5
            spacing: sc.sp2

            Text {
                text: root.insp.mode === "cluster"
                      ? (root.insp.cluster_id === -1 ? "NOISE" : "CLUSTER #" + root.insp.cluster_id)
                      : "TRACK #" + (root.insp.particle_id !== undefined ? root.insp.particle_id : "")
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                font.bold: true; font.letterSpacing: 1.0
            }

            // track rows
            Column {
                id: trackRows
                visible: root.insp.mode === "track"
                spacing: sc.sp1
                width: parent.width
                // Fixed-length Repeater so each CountUp instance persists and
                // animates from the old value to the new one on track switch.
                readonly property var fields: [
                    { k: "Length", value: root.insp.length,  decimals: 0, suffix: " frames", show: root.insp.length !== undefined },
                    { k: "D",      value: root.insp.d,        decimals: 4, suffix: " µm²/s",  show: root.insp.d !== undefined },
                    { k: "α",      value: root.insp.alpha,    decimals: 3, suffix: "",         show: root.insp.alpha !== undefined },
                    { k: "Net",    value: root.insp.net_displacement_um !== undefined
                                          ? Math.round(root.insp.net_displacement_um * 1000) : 0,
                                   decimals: 0, suffix: " nm", show: root.insp.net_displacement_um !== undefined }
                ]
                Repeater {
                    model: 4
                    delegate: Row {
                        required property int index
                        readonly property var f: trackRows.fields[index]
                        visible: f.show
                        width: col.width; spacing: sc.sp2
                        Text { text: f.k; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                               width: 42 }
                        CountUp { value: f.value !== undefined ? f.value : 0
                                  decimals: f.decimals; suffix: f.suffix
                                  color: pal.TXT; font.pixelSize: sc.textXs; font.family: "Menlo" }
                    }
                }
                // motion badge
                Row {
                    visible: root.insp.motion !== undefined && root.insp.motion !== ""
                    spacing: sc.sp2
                    topPadding: sc.sp1
                    Rectangle { width: 8; height: 8; radius: 4
                                color: root.insp.motionColor !== undefined ? root.insp.motionColor : pal.ACC
                                anchors.verticalCenter: parent.verticalCenter }
                    Text { text: root.insp.motion !== undefined ? root.insp.motion : ""
                           color: root.insp.motionColor !== undefined ? root.insp.motionColor : pal.TXT
                           font.pixelSize: sc.textXs; font.bold: true }
                }
            }

            // cluster note — wrap inside the card so a long "Dominant motion:
            // …" string can't overflow the fixed-width card past the screen edge
            Text {
                visible: root.insp.mode === "cluster" && root.insp.note !== undefined
                text: root.insp.note !== undefined ? root.insp.note : ""
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                width: col.width
                wrapMode: Text.WordWrap
            }
        }
    }
}
