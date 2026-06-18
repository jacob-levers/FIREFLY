import QtQuick
import QtQuick.Layouts
import "../components"

// Visualise tab: a QML layer rail + load/controls on the left, and an invisible
// "anchor" on the right whose scene rect the native FireflyViewer island is
// positioned over (EmbedController).  The viewer's own control bar handles
// transport / tail / head / background; the glass HUD + track inspector live in
// the transparent HUD overlay (HudOverlay.qml). Bound to the `Vis` controller.
Item {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ── left rail ────────────────────────────────────────────────────
        Rectangle {
            Layout.preferredWidth: 220
            Layout.fillHeight: true
            color: pal.PANEL
            Rectangle {     // right hairline
                anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                width: 1; color: pal.BORDER
            }
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: sc.sp5
                spacing: sc.sp4

                RowLayout {
                    spacing: sc.sp2
                    Icon { name: "layers"; size: 14; color: pal.ACC }
                    Text {
                        text: "LAYERS"; color: pal.TXT_MUTED
                        font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5
                    }
                }

                // load + view buttons
                Button {
                    Layout.fillWidth: true
                    variant: "primary"; text: "Open run…"; icon: "folder-open"
                    onClicked: Vis.loadRun()
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: sc.sp2
                    Button { variant: "secondary"; text: "Tracks"; icon: "waypoints"
                             Layout.fillWidth: true; onClicked: Vis.loadTracks() }
                    Button { variant: "secondary"; text: "Stack"; icon: "image"
                             Layout.fillWidth: true; onClicked: Vis.loadStack() }
                }
                Button {
                    Layout.fillWidth: true
                    variant: "secondary"; text: "Reset view"; icon: "scan-search"
                    onClicked: Vis.resetView()
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }

                // layer list
                Text {
                    visible: Vis.layers.length === 0
                    text: "Open a run to see its layers."
                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                    wrapMode: Text.WordWrap; Layout.fillWidth: true
                }
                Flickable {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contentHeight: layerCol.implicitHeight
                    clip: true
                    ColumnLayout {
                        id: layerCol
                        width: parent.width
                        spacing: sc.sp3
                        Repeater {
                            model: Vis.layers
                            delegate: LayerRow {
                                required property var modelData
                                Layout.fillWidth: true
                                name: modelData.name
                                kind: modelData.kind
                                tone: modelData.colorHex
                                layerVisible: modelData.visible
                                opacity_: modelData.opacity
                                count: modelData.count
                                onToggled: (on) => Vis.setLayerVisible(modelData.id, on)
                                onOpacitySet: (v) => Vis.setLayerOpacity(modelData.id, v)
                            }
                        }
                    }
                }

                // motion-colour toggle (segmented)
                Text {
                    text: "Motion colours"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: sc.sp2
                    Repeater {
                        model: Vis.motionColourModes
                        delegate: Rectangle {
                            required property string modelData
                            readonly property bool active: Vis.motionColourMode === modelData
                            Layout.fillWidth: true
                            implicitHeight: 26
                            radius: sc.radiusMd
                            color: active ? Qt.rgba(0.345, 0.651, 1.0, 0.14) : pal.PANEL_ALT
                            border.width: 1
                            border.color: active ? pal.ACC : pal.BORDER
                            Text {
                                anchors.centerIn: parent
                                text: modelData === "Colour-blind safe" ? "CB-safe" : modelData
                                color: active ? pal.ACC : pal.TXT_MUTED
                                font.pixelSize: sc.textXs
                            }
                            MouseArea {
                                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: Vis.motionColourMode = modelData
                            }
                        }
                    }
                }

                // minimum track length
                RowLayout {
                    Layout.fillWidth: true
                    spacing: sc.sp3
                    Text { text: "Min length"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                           Layout.fillWidth: true }
                    FieldInput {
                        Layout.preferredWidth: 64
                        horizontalAlignment: TextInput.AlignRight
                        text: "" + Vis.minLen
                        onEditingFinished: {
                            var v = parseInt(text)
                            if (!isNaN(v)) Vis.minLen = v
                        }
                    }
                }
            }
        }

        // ── right canvas (native viewer island anchors here) ─────────────
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Rectangle { anchors.fill: parent; color: "#0b0e13" }

            // empty-state hint (shows through until a run is loaded)
            ColumnLayout {
                anchors.centerIn: parent
                visible: !Vis.hasRun
                spacing: sc.sp3
                Icon { name: "waypoints"; size: 40; color: pal.TXT_MUTED
                       Layout.alignment: Qt.AlignHCenter }
                Text {
                    text: "Open a FIREFLY run to explore its tracks"
                    color: pal.TXT_MUTED; font.pixelSize: sc.textMd
                    Layout.alignment: Qt.AlignHCenter
                }
            }

            // invisible geometry source the native viewer is positioned over
            Item {
                id: viewerAnchor
                anchors.fill: parent
                function pushRect() {
                    var p = mapToItem(null, 0, 0)
                    Embed.setAnchorRect(p.x, p.y, width, height)
                }
                onWidthChanged: Qt.callLater(pushRect)
                onHeightChanged: Qt.callLater(pushRect)
                onXChanged: Qt.callLater(pushRect)
                onYChanged: Qt.callLater(pushRect)
                Component.onCompleted: Qt.callLater(pushRect)
            }
        }
    }
}
