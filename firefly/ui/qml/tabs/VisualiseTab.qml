import QtQuick
import QtQuick.Layouts
import "../components"

// Visualise tab: a SETTINGS rail on the left (Tracks / Clusters / Super-res /
// Explorer accordions), the native FireflyViewer island in the centre (anchored
// over an invisible Item whose scene rect EmbedController tracks), and a LAYERS
// panel on the right (load actions + per-run layer toggles, grouped by file).
// The viewer's own bar handles transport/background; the glass HUD + inspector
// live in HudOverlay.qml. Bound to `Vis`.
Item {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    // This layer is always loaded (see Main.qml) so its floating SETTINGS /
    // LAYERS cards can animate out when you leave the tab.  Off-tab it must be
    // inert: disabled (no input capture) and its painted surfaces hidden, so
    // the tab rendered underneath shows through untouched.
    readonly property bool onTab: App.currentTab === 3
    enabled: onTab

    // dark canvas behind the viewer + the side gutters the panels float over
    Rectangle { anchors.fill: parent; color: pal.BG; visible: root.onTab }

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

    // label-over-slider row (gradient track + glowing knob + value readout)
    component SliderRow: ColumnLayout {
        id: srow
        property string label
        property real from: 0
        property real to: 1
        property real step: 0
        property int decimals: 0
        property real value: 0
        signal moved(real v)
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        width: parent ? parent.width : implicitWidth
        spacing: sc.sp1
        Text { text: srow.label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
        Slider {
            Layout.fillWidth: true
            from: srow.from; to: srow.to; step: srow.step; decimals: srow.decimals
            value: srow.value
            onMoved: (v) => srow.moved(v)
        }
    }

    // Floating, rounded side panel.  Sits inset inside its column so the dark
    // canvas frames it on every edge, and slides + fades in from its own edge
    // when `shown` flips true (and back out when it flips false).  `edge`:
    // -1 = anchored to the left side, +1 = the right.  Default children land
    // in the card and should anchor.fill it.
    component FloatCard: Rectangle {
        id: fc
        property int edge: -1
        property bool shown: false
        readonly property var pal: Theme.palette
        anchors.top: parent.top; anchors.bottom: parent.bottom
        anchors.topMargin: 14; anchors.bottomMargin: 14
        width: parent.width - 24
        x: shown ? 12 : (edge < 0 ? -(width + 28) : parent.width + 28)
        opacity: shown ? 1 : 0
        visible: opacity > 0.01
        radius: 14
        color: pal.PANEL
        border.width: 1; border.color: pal.BORDER
        Behavior on x { NumberAnimation {
            duration: Theme.reducedMotion ? 0 : 340; easing.type: Easing.OutCubic } }
        Behavior on opacity { NumberAnimation {
            duration: Theme.reducedMotion ? 0 : 240; easing.type: Easing.OutCubic } }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ══════════════ LEFT: floating SETTINGS card ══════════════════════
        Item {
            Layout.preferredWidth: 320
            Layout.fillHeight: true
            FloatCard {
                edge: -1                       // slides in from the left
                shown: root.onTab
                Flickable {
                    anchors.fill: parent; anchors.margins: 3
                    contentHeight: rail.implicitHeight + sc.sp6
                    clip: true
                ColumnLayout {
                    id: rail
                    x: sc.sp5; y: sc.sp5
                    width: parent.width - sc.sp5 * 2
                    spacing: sc.sp4

                    RowLayout {
                        spacing: sc.sp2
                        Icon { name: "sliders-horizontal"; size: 14; color: pal.ACC }
                        Text { text: "SETTINGS"; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
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
                                    color: active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14) : pal.PANEL_ALT
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
                        Text { text: "Colour by" + (Vis.colourBy === "Auto"
                                     ? "  ·  auto → " + Vis.effectiveColourBy.toLowerCase() : "")
                               color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                        RowLayout {
                            width: parent.width; spacing: sc.sp2
                            Repeater {
                                model: Vis.colourByModes
                                delegate: Rectangle {
                                    required property string modelData
                                    readonly property bool active: Vis.colourBy === modelData
                                    Layout.fillWidth: true; implicitHeight: 26; radius: sc.radiusMd
                                    color: active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14) : pal.PANEL_ALT
                                    border.width: 1; border.color: active ? pal.ACC : pal.BORDER
                                    Text { anchors.centerIn: parent; text: modelData
                                           color: active ? pal.ACC : pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                                onClicked: Vis.setColourBy(modelData) }
                                }
                            }
                        }
                        SliderRow { label: "Track width"; from: 0.5; to: 12; step: 0.5; decimals: 1
                                    value: Vis.trackWidth; onMoved: (v) => Vis.trackWidth = v }
                        SliderRow { label: "Tail (frames)"; from: 0; to: 100; step: 1
                                    value: Vis.tail; onMoved: (v) => Vis.tail = v }
                        SliderRow { label: "Head (frames)"; from: 0; to: 100; step: 1
                                    value: Vis.head; onMoved: (v) => Vis.head = v }
                    }

                    // ── Clusters ─────────────────────────────────────────
                    CollapsibleSection {
                        Layout.fillWidth: true
                        title: "Clusters"; icon: "circle-dot"; expanded: false
                        // One click for the common case — the clusters that
                        // belong to the run whose tracks are already open — with
                        // "Load cluster map" kept beside it so any number of
                        // further maps can still be added from anywhere.
                        Button { width: parent.width; variant: "primary"
                                 visible: Vis.openRunHasClusters
                                 text: "Load clusters for " + Vis.openRunClusterName
                                 icon: "circle-dot"
                                 onClicked: Vis.loadClustersForOpenRun() }
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
                                    implicitWidth: cbLabel.implicitWidth + sc.sp3 * 2   // fit the label
                                    implicitHeight: 24; radius: sc.radiusMd
                                    color: active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14) : pal.PANEL_ALT
                                    border.width: 1; border.color: active ? pal.ACC : pal.BORDER
                                    Text { id: cbLabel; anchors.centerIn: parent; text: modelData
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
                            Button { variant: "primary"; Layout.fillWidth: true
                                     icon: Vis.srRendering ? "refresh-cw" : "sparkles"
                                     spin: Vis.srRendering
                                     text: Vis.srRendering ? "Rendering…" : "Render"
                                     enabled: !Vis.srRendering
                                     onClicked: Vis.renderSuperres() }
                            Button { variant: "secondary"; text: "Save PNG"; Layout.fillWidth: true
                                     enabled: Vis.hasSuperresRender && !Vis.srRendering
                                     onClicked: Vis.saveSuperres() }
                        }
                        Text { width: parent.width; text: Vis.srStatus; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; wrapMode: Text.WordWrap }
                    }

                    // ── Explorer ─────────────────────────────────────────
                    CollapsibleSection {
                        Layout.fillWidth: true
                        title: "Track explorer"; icon: "sliders-horizontal"; expanded: false
                        RowLayout {
                            width: parent.width; spacing: sc.sp3
                            Text { text: "D ≥"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                            FieldInput {
                                Layout.fillWidth: true; horizontalAlignment: TextInput.AlignRight
                                text: Vis.expDMin.toFixed(2)
                                onEditingFinished: { var v = parseFloat(text); if (!isNaN(v)) Vis.expDMin = v }
                            }
                            Text { text: "D ≤"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                            FieldInput {
                                Layout.fillWidth: true; horizontalAlignment: TextInput.AlignRight
                                text: Vis.expDMax.toFixed(2)
                                onEditingFinished: { var v = parseFloat(text); if (!isNaN(v)) Vis.expDMax = v }
                            }
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
                                    color: on_ ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14) : pal.PANEL_ALT
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
        }

        // ══════════════ CENTRE: viewer (native island anchors here) ════════
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.centerIn: parent
                visible: !Vis.hasContent && root.onTab
                spacing: sc.sp3
                Icon { name: "waypoints"; size: 40; color: pal.TXT_MUTED
                       Layout.alignment: Qt.AlignHCenter }
                Text { text: "Open a FIREFLY run to explore its tracks"
                       color: pal.TXT_MUTED; font.pixelSize: sc.textMd
                       Layout.alignment: Qt.AlignHCenter }
            }

            Item {
                id: viewerAnchor
                // Inset so the native viewer island floats as a rounded card,
                // matching the side panels (the EmbedController rounds its corners
                // + the HUD draws the border at this rect).  Extra bottom inset
                // reserves a strip for the QML transport bar below (which must sit
                // in the chrome, clear of the island, to receive mouse events).
                anchors.fill: parent
                anchors.topMargin: 14; anchors.bottomMargin: 64
                anchors.leftMargin: 2; anchors.rightMargin: 2
                function pushRect() {
                    var p = mapToItem(null, 0, 0)
                    Embed.setAnchorRect(p.x, p.y, width, height)
                }
                onWidthChanged: Qt.callLater(pushRect)
                onHeightChanged: Qt.callLater(pushRect)
                onXChanged: Qt.callLater(pushRect)
                onYChanged: Qt.callLater(pushRect)
                Component.onCompleted: Qt.callLater(pushRect)
                // The tab is always-loaded now, so it no longer re-mounts on
                // entry — re-push the rect when we return to Visualise so the
                // island re-anchors to the (possibly resized) centre.
                property bool tabActive: root.onTab
                onTabActiveChanged: if (tabActive) Qt.callLater(pushRect)
            }

            // ── transport scrubber (QML; replaces the dated native bar) ────
            Rectangle {
                id: transport
                visible: root.onTab && Vis.nFrames > 1
                anchors.left: parent.left;     anchors.leftMargin: 2
                anchors.right: parent.right;   anchors.rightMargin: 2
                anchors.bottom: parent.bottom; anchors.bottomMargin: 14
                height: 40; radius: 12
                color: Qt.rgba(0.05, 0.07, 0.10, 0.86)
                border.width: 1; border.color: pal.BORDER_HI

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: sc.sp3; anchors.rightMargin: sc.sp4
                    spacing: sc.sp3

                    Rectangle {                       // play / pause
                        Layout.preferredWidth: 28; Layout.preferredHeight: 28; radius: 7
                        color: phov.hovered ? pal.PANEL_ALT : "transparent"
                        Icon { anchors.centerIn: parent; name: Vis.playing ? "pause" : "play"
                               size: 15; color: pal.ACC }
                        HoverHandler { id: phov }
                        TapHandler { onTapped: Vis.playPause() }
                    }
                    Slider {                          // scrubber
                        Layout.fillWidth: true
                        showValue: false
                        from: 0; to: Math.max(1, Vis.nFrames - 1); step: 1; decimals: 0
                        value: Vis.currentFrame
                        onMoved: (v) => Vis.seek(v)
                    }
                    Text {                            // frame counter
                        text: (Vis.currentFrame + 1) + " / " + Vis.nFrames
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                    }
                    SpinBox {                         // playback rate
                        Layout.preferredWidth: 60
                        radius: sc.radius2xl          // match the bar's rounding
                        textAlign: TextInput.AlignHCenter
                        from: 1; to: 60; value: Vis.fps; suffix: " fps"
                        onCommitted: (v) => Vis.fps = v
                    }
                }
            }
        }

        // ══════════════ RIGHT: floating LAYERS card ═══════════════════════
        Item {
            Layout.preferredWidth: 264
            Layout.fillHeight: true
            FloatCard {
                edge: 1                        // slides in from the right
                shown: root.onTab
                Flickable {
                    anchors.fill: parent; anchors.margins: 3
                    contentHeight: layersCol.implicitHeight + sc.sp6
                    clip: true
                ColumnLayout {
                    id: layersCol
                    x: sc.sp5; y: sc.sp5
                    width: parent.width - sc.sp5 * 2
                    spacing: sc.sp4

                    RowLayout {
                        spacing: sc.sp2
                        Icon { name: "layers"; size: 14; color: pal.ACC }
                        Text { text: "LAYERS"; color: pal.TXT_MUTED
                               font.pixelSize: sc.textXs; font.bold: true; font.letterSpacing: 1.5 }
                    }
                    Button { Layout.fillWidth: true; variant: "primary"
                             text: Vis.hasRun ? "Add run…" : "Open run…"
                             icon: "folder-open"; onClicked: Vis.loadRun() }
                    RowLayout {
                        Layout.fillWidth: true; spacing: sc.sp2
                        Button { variant: "secondary"; text: "Tracks"; icon: "waypoints"
                                 Layout.fillWidth: true; onClicked: Vis.loadTracks() }
                        Button { variant: "secondary"; text: "Stack"; icon: "image"
                                 Layout.fillWidth: true; onClicked: Vis.loadStack() }
                    }
                    RowLayout {
                        Layout.fillWidth: true; spacing: sc.sp2
                        Button { variant: "secondary"; text: "Reset view"; icon: "scan-search"
                                 Layout.fillWidth: true; onClicked: Vis.resetView() }
                        // Clear everything loaded/generated back to an empty tab.
                        Button {
                            variant: "secondary"; text: "Clear"; icon: "trash-2"
                            Layout.fillWidth: true; enabled: Vis.hasContent
                            onClicked: clearConfirm.open()
                        }
                    }

                    // ── background layer (above the per-layer list) ──────────
                    ColumnLayout {
                        visible: Vis.backgroundOptions.length > 0
                        Layout.fillWidth: true; spacing: sc.sp2
                        Text { text: "Background"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                        Select {
                            Layout.fillWidth: true
                            model: Vis.backgroundOptions
                            currentIndex: Math.max(0, Vis.backgroundOptions.indexOf(Vis.backgroundMode))
                            onPicked: (t) => Vis.selectBackground(t)
                        }
                    }

                    Text {
                        visible: Vis.layers.length === 0
                        text: "Open a run to see its layers."
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                        wrapMode: Text.WordWrap; Layout.fillWidth: true
                    }
                    // layer list — grouped by file when the controller tags
                    // layers with a `file`/`run` name (multi-run compare).
                    Repeater {
                        model: Vis.layerGroups
                        delegate: ColumnLayout {
                            id: grpItem
                            required property var modelData
                            readonly property bool grouped: modelData.file !== ""
                            readonly property bool allVisible: {
                                var ls = grpItem.modelData.layers, ok = true
                                for (var i = 0; i < ls.length; ++i) ok = ok && ls[i].visible
                                return ok
                            }
                            Layout.fillWidth: true
                            spacing: sc.sp2
                            RowLayout {                  // file/run header (multi-run only)
                                visible: grpItem.grouped
                                Layout.fillWidth: true; Layout.topMargin: sc.sp2; spacing: sc.sp2
                                Rectangle { width: 8; height: 8; radius: 2
                                            color: grpItem.modelData.colorHex !== "" ? grpItem.modelData.colorHex : pal.TXT_MUTED }
                                Text { text: grpItem.modelData.file; color: pal.TXT; font.pixelSize: sc.textXs
                                       font.weight: Font.DemiBold; font.family: "Menlo"
                                       elide: Text.ElideMiddle; Layout.fillWidth: true; Layout.preferredWidth: 0 }
                                Icon {
                                    name: grpItem.allVisible ? "eye" : "eye-off"; size: 14
                                    color: grpItem.allVisible ? pal.ACC : pal.TXT_MUTED
                                    MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                                onClicked: Vis.setRunVisible(grpItem.modelData.file, !grpItem.allVisible) }
                                }
                            }
                            Repeater {
                                model: grpItem.modelData.layers
                                delegate: LayerRow {
                                    required property var modelData
                                    Layout.fillWidth: true
                                    Layout.leftMargin: grpItem.grouped ? sc.sp3 : 0
                                    name: modelData.name; kind: modelData.kind; tone: modelData.colorHex
                                    layerVisible: modelData.visible; opacity_: modelData.opacity
                                    count: modelData.count
                                    onToggled: (on) => Vis.setLayerVisible(modelData.id, on)
                                    onOpacitySet: (v) => Vis.setLayerOpacity(modelData.id, v)
                                }
                            }
                        }
                    }
                }
            }
        }
        }
    }

    // ══ blocking "loading movie…" popup ══
    // A large raw movie (a multi-GB .czi) can take a minute to decode.  It's
    // decoded off the GUI thread so the window never hard-freezes, but this
    // modal blocks interaction until it's ready — so playback is only reached
    // once the movie is fully loaded (never mid-decode, when it would stutter).
    // Not dismissable; "Skip movie" cancels the decode.
    Modal {
        id: movieLoadingModal
        title: ""                        // headerless → clean centred body
        dismissable: false
        opened: Vis.movieLoading

        // progress ring rotating around a static play glyph
        Item {
            Layout.alignment: Qt.AlignHCenter
            implicitWidth: 56; implicitHeight: 56
            Icon {
                anchors.fill: parent
                name: "loader-circle"; size: 56; color: pal.ACC
                RotationAnimator on rotation {
                    running: Vis.movieLoading && !Theme.reducedMotion
                    loops: Animation.Infinite; from: 0; to: 360; duration: 1000
                }
            }
            Icon { anchors.centerIn: parent; name: "play"; size: 20; color: pal.ACC }
        }
        Text {
            Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter
            text: "Loading movie…"; color: pal.TXT
            font.pixelSize: sc.textLg; font.weight: Font.DemiBold
        }
        Text {
            Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter
            text: Vis.movieLoadingLabel; color: pal.TXT_MUTED
            font.pixelSize: sc.textXs; font.family: "Menlo"; elide: Text.ElideMiddle
        }
        Text {
            Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter
            text: "This can take a minute."; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
        }
        Button {
            Layout.alignment: Qt.AlignHCenter; Layout.topMargin: sc.sp2
            variant: "ghost"; text: "Skip movie"; onClicked: Vis.cancelMovieLoad()
        }
    }

    // confirm before wiping loaded runs / movie / clusters / super-res
    ConfirmModal {
        id: clearConfirm
        title: "Clear the Visualise tab?"
        message: "This removes every loaded run, movie, cluster map and super-res "
                 + "layer from the viewer. Your files aren't deleted — you can load "
                 + "them again. Settings (colours, cluster params) are kept."
        confirmText: "Clear"
        action: function () { Vis.clearAll() }
    }
}
