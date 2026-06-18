import QtQuick
import QtQuick.Layouts
import "../components"

// Visualise tab: a scrollable QML control rail on the left (layers + Tracks /
// Clusters / Super-resolution / Explorer accordions) and an invisible "anchor"
// on the right whose scene rect the native FireflyViewer island is positioned
// over (EmbedController). The viewer's own bar handles transport / background;
// the glass HUD + inspector live in HudOverlay.qml. Bound to `Vis`.
Item {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    // label + numeric field row (compact, reused across the panels)
    component NumRow: RowLayout {
        property string label
        property string value
        property bool enabled_: true
        signal committed(string text)
        width: parent ? parent.width : implicitWidth
        spacing: sc.sp3
        Text { text: label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs; Layout.fillWidth: true }
        FieldInput {
            Layout.preferredWidth: 72
            enabled: enabled_
            opacity: enabled_ ? 1.0 : 0.45
            horizontalAlignment: TextInput.AlignRight
            text: value
            onEditingFinished: committed(text)
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ── left rail (scrollable) ───────────────────────────────────────
        Rectangle {
            Layout.preferredWidth: 264
            Layout.fillHeight: true
            color: pal.PANEL
            Rectangle {
                anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                width: 1; color: pal.BORDER
            }
            Flickable {
                anchors.fill: parent
                contentHeight: rail.implicitHeight + sc.sp6
                clip: true
                ColumnLayout {
                    id: rail
                    x: sc.sp5; y: sc.sp5
                    width: parent.width - sc.sp5 * 2
                    spacing: sc.sp4

                    RowLayout {
                        spacing: sc.sp2
                        Icon { name: "layers"; size: 14; color: pal.ACC }
                        Text { text: "LAYERS"; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                    }
                    Button { Layout.fillWidth: true; variant: "primary"; text: "Open run…"
                             icon: "folder-open"; onClicked: Vis.loadRun() }
                    RowLayout {
                        Layout.fillWidth: true; spacing: sc.sp2
                        Button { variant: "secondary"; text: "Tracks"; icon: "waypoints"
                                 Layout.fillWidth: true; onClicked: Vis.loadTracks() }
                        Button { variant: "secondary"; text: "Stack"; icon: "image"
                                 Layout.fillWidth: true; onClicked: Vis.loadStack() }
                    }
                    Button { Layout.fillWidth: true; variant: "secondary"; text: "Reset view"
                             icon: "scan-search"; onClicked: Vis.resetView() }

                    Text {
                        visible: Vis.layers.length === 0
                        text: "Open a run to see its layers."
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                        wrapMode: Text.WordWrap; Layout.fillWidth: true
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: sc.sp3
                        Repeater {
                            model: Vis.layers
                            delegate: LayerRow {
                                required property var modelData
                                Layout.fillWidth: true
                                name: modelData.name; kind: modelData.kind; tone: modelData.colorHex
                                layerVisible: modelData.visible; opacity_: modelData.opacity
                                count: modelData.count
                                onToggled: (on) => Vis.setLayerVisible(modelData.id, on)
                                onOpacitySet: (v) => Vis.setLayerOpacity(modelData.id, v)
                            }
                        }
                    }

                    // ── Tracks ───────────────────────────────────────────
                    CollapsibleSection {
                        Layout.fillWidth: true
                        title: "Tracks"; icon: "waypoints"; expanded: true
                        Text { text: "Motion colours"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                        RowLayout {
                            width: parent.width; spacing: sc.sp2
                            Repeater {
                                model: Vis.motionColourModes
                                delegate: Rectangle {
                                    required property string modelData
                                    readonly property bool active: Vis.motionColourMode === modelData
                                    Layout.fillWidth: true; implicitHeight: 26; radius: sc.radiusMd
                                    color: active ? Qt.rgba(0.345, 0.651, 1.0, 0.14) : pal.PANEL_ALT
                                    border.width: 1; border.color: active ? pal.ACC : pal.BORDER
                                    Text { anchors.centerIn: parent
                                           text: modelData === "Colour-blind safe" ? "CB-safe" : modelData
                                           color: active ? pal.ACC : pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                                onClicked: Vis.motionColourMode = modelData }
                                }
                            }
                        }
                        NumRow { label: "Min length"; value: "" + Vis.minLen
                                 onCommitted: (t) => { var v = parseInt(t); if (!isNaN(v)) Vis.minLen = v } }
                    }

                    // ── Clusters ─────────────────────────────────────────
                    CollapsibleSection {
                        Layout.fillWidth: true
                        title: "Clusters"; icon: "circle-dot"; expanded: false
                        Button { width: parent.width; variant: "secondary"; text: "Load cluster map"
                                 icon: "folder-open"; onClicked: Vis.loadClusters() }
                        Alert {
                            visible: Vis.noClustersBanner
                            width: parent.width
                            severity: "warn"
                            text: "No clusters at this eps — lower it."
                        }
                        NumRow { label: "eps (nm)"; value: "" + Vis.clusterEpsNm
                                 onCommitted: (t) => { var v = parseInt(t); if (!isNaN(v)) Vis.clusterEpsNm = v } }
                        NumRow { label: "min samples"; value: "" + Vis.clusterMinSamples
                                 onCommitted: (t) => { var v = parseInt(t); if (!isNaN(v)) Vis.clusterMinSamples = v } }
                        NumRow { label: "point size"; value: "" + Vis.clusterPointSize
                                 onCommitted: (t) => { var v = parseInt(t); if (!isNaN(v)) Vis.clusterPointSize = v } }
                        RowLayout {
                            width: parent.width; spacing: sc.sp2
                            Button { variant: "secondary"; text: "Suggest eps"; Layout.fillWidth: true
                                     onClicked: Vis.suggestEps() }
                            Button { variant: "secondary"; text: "Export"; Layout.fillWidth: true
                                     onClicked: Vis.exportTunedClusters() }
                        }
                        RowLayout {
                            width: parent.width; spacing: sc.sp2
                            Text { text: "Colour by"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                   Layout.fillWidth: true }
                            Repeater {
                                model: Vis.clusterColorModes
                                delegate: Rectangle {
                                    required property string modelData
                                    readonly property bool active: Vis.clusterColorMode === modelData
                                    implicitWidth: 52; implicitHeight: 24; radius: sc.radiusMd
                                    color: active ? Qt.rgba(0.345, 0.651, 1.0, 0.14) : pal.PANEL_ALT
                                    border.width: 1; border.color: active ? pal.ACC : pal.BORDER
                                    Text { anchors.centerIn: parent; text: modelData
                                           color: active ? pal.ACC : pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                                onClicked: Vis.clusterColorMode = modelData }
                                }
                            }
                        }
                        Text { width: parent.width; text: Vis.clusterStatus; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; wrapMode: Text.WordWrap }
                    }

                    // ── Super-resolution ─────────────────────────────────
                    CollapsibleSection {
                        Layout.fillWidth: true
                        title: "Super-resolution"; icon: "zap"; expanded: false
                        NumRow { label: "px size (nm)"; value: "" + Vis.srPixelNm
                                 onCommitted: (t) => { var v = parseInt(t); if (!isNaN(v)) Vis.srPixelNm = v } }
                        NumRow { label: "blur σ (nm)"; value: "" + Vis.srBlurNm
                                 onCommitted: (t) => { var v = parseInt(t); if (!isNaN(v)) Vis.srBlurNm = v } }
                        RowLayout {
                            width: parent.width; spacing: sc.sp2
                            Button { variant: "primary"; text: "Render"; Layout.fillWidth: true
                                     onClicked: Vis.renderSuperres() }
                            Button { variant: "secondary"; text: "Save PNG"; Layout.fillWidth: true
                                     enabled: Vis.hasSuperresRender; onClicked: Vis.saveSuperres() }
                        }
                        Text { width: parent.width; text: Vis.srStatus; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; wrapMode: Text.WordWrap }
                    }

                    // ── Explorer ─────────────────────────────────────────
                    CollapsibleSection {
                        Layout.fillWidth: true
                        title: "Track explorer"; icon: "sliders-horizontal"; expanded: false
                        RowLayout {
                            width: parent.width; spacing: sc.sp2
                            NumRow { Layout.fillWidth: true; label: "D ≥"; value: Vis.expDMin.toFixed(2)
                                     onCommitted: (t) => { var v = parseFloat(t); if (!isNaN(v)) Vis.expDMin = v } }
                            NumRow { Layout.fillWidth: true; label: "D ≤"; value: Vis.expDMax.toFixed(2)
                                     onCommitted: (t) => { var v = parseFloat(t); if (!isNaN(v)) Vis.expDMax = v } }
                        }
                        NumRow { label: "min length"; value: "" + Vis.expMinLen
                                 onCommitted: (t) => { var v = parseInt(t); if (!isNaN(v)) Vis.expMinLen = v } }
                        Text { text: "Motion"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                        Flow {
                            width: parent.width; spacing: sc.sp2
                            Repeater {
                                model: ["Immobile", "Confined", "Brownian", "Directed"]
                                delegate: Rectangle {
                                    required property string modelData
                                    readonly property bool on_: Vis.expMotionMask[modelData] === true
                                    implicitWidth: glyph.implicitWidth + sc.sp4 * 2; implicitHeight: 24
                                    radius: sc.radiusMd
                                    color: on_ ? Qt.rgba(0.345, 0.651, 1.0, 0.14) : pal.PANEL_ALT
                                    border.width: 1; border.color: on_ ? pal.ACC : pal.BORDER
                                    Text { id: glyph; anchors.centerIn: parent; text: modelData
                                           color: on_ ? pal.ACC : pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                                onClicked: Vis.setExpMotion(modelData, !on_) }
                                }
                            }
                        }
                        Text { width: parent.width; text: Vis.explorerCount; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs }
                        // compact result list (click a row to centre the viewer)
                        ColumnLayout {
                            width: parent.width; spacing: 1
                            Repeater {
                                model: Vis.explorerRows.slice(0, 60)
                                delegate: Rectangle {
                                    required property var modelData
                                    Layout.fillWidth: true; implicitHeight: 22; radius: sc.radiusXs
                                    color: rowHov.hovered ? pal.PANEL_ALT : "transparent"
                                    RowLayout {
                                        anchors.fill: parent; anchors.leftMargin: sc.sp2
                                        anchors.rightMargin: sc.sp2; spacing: sc.sp2
                                        Text { text: "#" + modelData.particle; color: pal.TXT
                                               font.pixelSize: sc.textXs; font.family: "Menlo"
                                               Layout.preferredWidth: 44 }
                                        Text { text: modelData.d !== null ? modelData.d.toFixed(3) : "—"
                                               color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                               font.family: "Menlo"; Layout.fillWidth: true }
                                        Text { text: modelData.motion; color: pal.TXT_MUTED
                                               font.pixelSize: sc.textXs }
                                    }
                                    HoverHandler { id: rowHov }
                                    TapHandler { onTapped: Vis.selectTrack(modelData.particle) }
                                }
                            }
                        }
                        Button { width: parent.width; variant: "secondary"; text: "Export filtered"
                                 icon: "arrow-up-right"; onClicked: Vis.exportFilteredTracks() }
                    }
                }
            }
        }

        // ── right canvas (native viewer island anchors here) ─────────────
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Rectangle { anchors.fill: parent; color: "#0b0e13" }

            ColumnLayout {
                anchors.centerIn: parent
                visible: !Vis.hasRun
                spacing: sc.sp3
                Icon { name: "waypoints"; size: 40; color: pal.TXT_MUTED
                       Layout.alignment: Qt.AlignHCenter }
                Text { text: "Open a FIREFLY run to explore its tracks"
                       color: pal.TXT_MUTED; font.pixelSize: sc.textMd
                       Layout.alignment: Qt.AlignHCenter }
            }

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
