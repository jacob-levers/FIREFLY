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

    // ── inspector card (bottom-right, above the native control bar) ──────
    Rectangle {
        id: card
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.rightMargin: sc.sp6
        anchors.bottomMargin: 70
        width: 168
        height: col.implicitHeight + sc.sp5 * 2
        radius: sc.radius2xl
        color: Qt.rgba(0.05, 0.07, 0.10, 0.74)
        border.width: 1; border.color: Qt.rgba(1, 1, 1, 0.10)
        visible: Vis.inspectorVisible

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
                visible: root.insp.mode === "track"
                spacing: sc.sp1
                width: parent.width
                Repeater {
                    model: [
                        { k: "Length", v: root.insp.length !== undefined ? root.insp.length + " frames" : "" },
                        { k: "D",      v: root.insp.d !== undefined ? root.insp.d.toFixed(4) + " µm²/s" : "" },
                        { k: "α",      v: root.insp.alpha !== undefined ? root.insp.alpha.toFixed(3) : "" },
                        { k: "Net",    v: root.insp.net_displacement_um !== undefined
                                          ? Math.round(root.insp.net_displacement_um * 1000) + " nm" : "" }
                    ]
                    delegate: Row {
                        required property var modelData
                        visible: modelData.v !== ""
                        width: col.width; spacing: sc.sp2
                        Text { text: modelData.k; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                               width: 42 }
                        Text { text: modelData.v; color: pal.TXT; font.pixelSize: sc.textXs
                               font.family: "Menlo" }
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

            // cluster note
            Text {
                visible: root.insp.mode === "cluster" && root.insp.note !== undefined
                text: root.insp.note !== undefined ? root.insp.note : ""
                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
            }
        }
    }
}
