import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import "../components"

// ── Analysis workspace (merged Compare + Results) ─────────────────────────
// Right rail = inputs (conditions / timepoints / settings); left = a live
// readout (figure → headline → stats → significance → methods).  Everything
// recomputes the instant data or settings change — no Generate button.
Item {
    id: root
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property color faint: "#5b636e"        // dim text; same on both dark themes
    // layered shades (darkest→lightest): page = BG < recessed well / figure mat
    // ≈ BG < card body = PANEL < card header < chip = PANEL_ALT.  mat + cSunken
    // track the theme via BG (so AMOLED goes pure-black) rather than a near-black
    // WELL, which made the Dark theme read as AMOLED in the Analysis cards.
    readonly property color mat: pal.BG
    readonly property color cSunken: pal.BG
    readonly property color cCardHead: pal.PANEL_ALT  // header band — theme-aware (was a hardcoded dark navy that stayed dark in Light mode)
    // design spacing: cards are 12px-radius with 14px interior padding and sit
    // 14px apart; columns are 16px apart with a 20px page inset.
    readonly property int gRad: 12
    readonly property int gPad: 14
    readonly property int gGap: 14

    // header'd card matching the design (uppercase title + count + right slot)
    component WCard: Rectangle {
        id: wc
        default property alias content: body.data
        property string title
        property string count
        property Item headerRight
        property int pad: root.gPad
        property int enterDelay: 0        // §18 staggered-entrance offset (ms)
        color: pal.PANEL
        radius: root.gRad
        border.width: 1
        border.color: pal.BORDER
        implicitHeight: head.height + body.implicitHeight + body.anchors.margins * 2
        clip: true
        // §18 — the card fades + rises in as the tab/view assembles itself.
        // opacity/transform are visual only, so layout sizing is unaffected.
        opacity: 0
        transform: Translate { id: wcT; y: 8 }
        Component.onCompleted: wcEnter.start()
        SequentialAnimation {
            id: wcEnter
            PauseAnimation { duration: Theme.reducedMotion ? 0 : wc.enterDelay }
            ParallelAnimation {
                NumberAnimation { target: wc; property: "opacity"; to: 1
                                  duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic }
                NumberAnimation { target: wcT; property: "y"; to: 0
                                  duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic }
            }
        }
        Rectangle {                                  // header band (a touch lighter)
            visible: wc.title !== ""
            anchors {
                left: parent.left; right: parent.right; top: parent.top
                leftMargin: 1; rightMargin: 1; topMargin: 1   // sit inside the 1px border
            }
            height: head.height - 1
            color: root.cCardHead
            radius: root.gRad - 1                    // round only the top corners
            bottomLeftRadius: 0
            bottomRightRadius: 0
        }
        RowLayout {
            id: head
            visible: wc.title !== ""
            height: visible ? 40 : 0
            anchors { left: parent.left; right: parent.right; top: parent.top }
            anchors.leftMargin: root.gPad; anchors.rightMargin: root.gPad
            spacing: sc.sp3
            Text {
                text: wc.title; color: pal.TXT_MUTED
                font.pixelSize: 11; font.bold: true
                font.letterSpacing: 1.0; font.capitalization: Font.AllUppercase
            }
            Text {
                visible: wc.count !== ""; text: wc.count
                color: root.faint; font.pixelSize: 11; font.family: "Menlo"
            }
            Item { Layout.fillWidth: true }
        }
        // header right-slot host — sized to its child so anchors.right
        // right-aligns the content instead of letting it run off the edge.
        Item {
            anchors { right: parent.right; rightMargin: root.gPad; verticalCenter: head.verticalCenter }
            width: childrenRect.width
            height: childrenRect.height
            children: wc.headerRight ? [wc.headerRight] : []
        }
        Item {
            id: body
            anchors { left: parent.left; right: parent.right; top: head.bottom; margins: wc.pad }
            implicitHeight: childrenRect.height
        }
        // top hairline under header
        Rectangle {
            visible: wc.title !== ""
            anchors { left: parent.left; right: parent.right; top: head.bottom }
            height: 1; color: pal.BORDER
        }
    }

    // ── layout ────────────────────────────────────────────────────────────
    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 20; anchors.rightMargin: 20
        anchors.topMargin: sc.sp4; anchors.bottomMargin: sc.sp5
        spacing: root.gGap

        // ════ control band ════ — fixed height so the body doesn't shift when
        // the metric bar (Comparison only) appears/disappears.
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 34
            Layout.minimumHeight: 34
            spacing: sc.sp4
            Segmented {
                solid: true
                Layout.alignment: Qt.AlignVCenter
                Layout.preferredWidth: 320
                options: [
                    { v: "comparison", t: "Comparison", icon: "git-compare" },
                    { v: "panels", t: "All panels · " + Analysis.panelCount, icon: "layout-grid" }
                ]
                value: Analysis.view
                onPicked: (v) => Analysis.setView(v)
            }
            // metric switcher (comparison only)
            Item {
                Layout.fillWidth: true
                Layout.preferredHeight: 34
                visible: Analysis.view === "comparison"
                Flickable {
                    id: metricFlick
                    anchors.fill: parent
                    contentWidth: metricRow.width; contentHeight: height
                    flickableDirection: Flickable.HorizontalFlick
                    clip: true
                    Row {
                        id: metricRow
                        height: parent.height
                        leftPadding: 62; rightPadding: 28
                        spacing: sc.sp2
                        Repeater {
                            model: Analysis.metrics
                            delegate: Rectangle {
                                required property var modelData
                                anchors.verticalCenter: parent.verticalCenter
                                readonly property bool on: Analysis.metric === modelData.id
                                implicitWidth: ml.implicitWidth + sc.sp10   // roomier side padding
                                implicitHeight: 28
                                radius: height / 2
                                // scroll-aware edge fade: a pill dims as it slides
                                // behind the METRIC label (left) or off the right
                                // edge. _vx is the pill's left edge in viewport
                                // coords; the resting first pill stays crisp.
                                readonly property real _vx: x - metricFlick.contentX
                                opacity: {
                                    // the active pill stays fully crisp — otherwise the
                                    // edge-fade dimmed its subtle border to nothing while
                                    // the bold text stayed readable, so it looked like the
                                    // selected pill had lost its outline.
                                    if (on) return 1.0;
                                    // left: stay crisp until the pill's RIGHT (trailing)
                                    // edge reaches the METRIC label, then fade as it
                                    // slides under — so the pill doesn't vanish while
                                    // still well right of the blur, and short labels
                                    // like "MSD" stay crisp at rest.
                                    var vR = _vx + width;
                                    var la = Math.max(0, Math.min(1, (vR - 32) / 36));
                                    // right: fade as the LEFT edge slides off the right.
                                    var ra = Math.max(0, Math.min(1, (metricFlick.width - _vx) / 72));
                                    return Math.min(la, ra);
                                }
                                color: on ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.16) : pal.PANEL
                                border.width: 1
                                border.color: on ? pal.ACC : pal.BORDER
                                RowLayout {
                                    id: ml
                                    anchors.centerIn: parent
                                    spacing: 4
                                    Text {
                                        text: modelData.label
                                        color: on ? pal.ACC : pal.TXT_MUTED
                                        font.pixelSize: sc.textSm; font.bold: on
                                    }
                                    Text {
                                        visible: modelData.approx === true
                                        text: "≈"; color: root.faint; font.pixelSize: sc.textSm
                                    }
                                }
                                TapHandler { onTapped: Analysis.setMetric(modelData.id) }
                                HoverHandler { cursorShape: Qt.PointingHandCursor }
                            }
                        }
                    }
                }
                // fixed METRIC label + left fade
                Rectangle {
                    anchors { left: parent.left; top: parent.top; bottom: parent.bottom }
                    width: 62
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: pal.BG }
                        GradientStop { position: 0.82; color: pal.BG }
                        GradientStop { position: 1.0; color: "transparent" }
                    }
                    Text {
                        anchors { left: parent.left; verticalCenter: parent.verticalCenter }
                        text: "METRIC"; color: root.faint
                        font.pixelSize: 11; font.bold: true; font.letterSpacing: 1.0
                    }
                }
                // right-edge fade — dissolves the hard clip boundary into the
                // background so a pill scrolling off the right doesn't hit a
                // harsh cut line.  Only shown while there's more content to the
                // right (fades out when scrolled fully to the end).
                Rectangle {
                    anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                    width: 64
                    opacity: Math.max(0, Math.min(1,
                        (metricFlick.contentWidth - metricFlick.width - metricFlick.contentX) / 40))
                    Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop { position: 0.0; color: "transparent" }
                        GradientStop { position: 1.0; color: pal.BG }
                    }
                }
            }
        }

        // ════ body (two columns, vertical scroll) ════
        Flickable {
            id: bodyFlick
            Layout.fillWidth: true; Layout.fillHeight: true
            contentWidth: width; contentHeight: bodyRow.implicitHeight
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

            GridLayout {
                id: bodyRow
                width: bodyFlick.width
                // Stack the two columns below ~880 px so neither gets squeezed —
                // mirrors the responsive pattern on the Process / Results tabs.
                columns: width < 880 ? 1 : 2
                readonly property bool stacked: columns === 1
                columnSpacing: sc.sp8
                rowSpacing: sc.sp8

                // ── LEFT: live readout ──
                ColumnLayout {
                    Layout.alignment: Qt.AlignTop
                    Layout.fillWidth: true
                    spacing: root.gGap

                    // empty state
                    WCard {
                        visible: !Analysis.enough
                        Layout.fillWidth: true
                        title: "Live results"
                        ColumnLayout {
                            width: parent.width
                            spacing: sc.sp3
                            Item { Layout.preferredHeight: sc.sp6 }
                            Icon { Layout.alignment: Qt.AlignHCenter; name: "git-compare"; size: 40; color: root.faint }
                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                horizontalAlignment: Text.AlignHCenter
                                text: "Add 2+ run folders to at least two conditions.\nResults update live as you add data — no Generate needed."
                                color: pal.TXT_MUTED; font.pixelSize: sc.textSm
                            }
                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: Analysis.readyCount + " of " + Analysis.conditionCount + " conditions ready"
                                color: root.faint; font.pixelSize: sc.textXs
                            }
                            Item { Layout.preferredHeight: sc.sp6 }
                        }
                    }

                    // comparison / panels view — SlideStack crossfades the two
                    // body views in-place (slide:0 = FIREFLY default; the views
                    // share this LEFT-column frame, side panels live on the right).
                    // active+visible reproduce the old `active: Analysis.enough`
                    // gate so it collapses (no ghost height) for the empty state.
                    SlideStack {
                        Layout.fillWidth: true
                        visible: Analysis.enough
                        active: Analysis.enough
                        index: Analysis.view === "panels" ? 1 : 0
                        slide: 0
                        views: [comparisonView, panelsView]
                    }
                }

                // ── RIGHT: inputs — ~35% of the width like the design's
                //    minmax(340px, 0.85fr), clamped so it's neither cramped nor
                //    able to swallow the whole row in the empty state.
                ColumnLayout {
                    Layout.alignment: Qt.AlignTop
                    // Full width when stacked; ~34% (clamped 360–580) side-by-side.
                    Layout.fillWidth: bodyRow.stacked
                    Layout.preferredWidth: bodyRow.stacked ? bodyRow.width
                                                           : Math.round(bodyRow.width * 0.34)
                    Layout.minimumWidth: bodyRow.stacked ? 0 : 360
                    Layout.maximumWidth: bodyRow.stacked ? bodyRow.width : 580
                    spacing: root.gGap
                    ConditionsCard {}
                    DesignCard {}
                    SettingsCard {}
                }
            }
        }
    }

    // ════ comparison view ════
    Component {
        id: comparisonView
        ColumnLayout {
            spacing: root.gGap

            // figure hero
            WCard {
                Layout.fillWidth: true
                title: Analysis.figureTitle
                pad: 0
                headerRight: RowLayout {
                    spacing: sc.sp2
                    RowLayout {
                        spacing: 5
                        StatusDot {                       // pulsing glow ring on the live readout
                            tone: Analysis.busy ? pal.WARN : pal.SUCCESS
                            pulsing: true
                            Layout.alignment: Qt.AlignVCenter
                        }
                        Text {
                            text: Analysis.busy ? "Recomputing" : "Live"
                            color: Analysis.busy ? pal.WARN : pal.SUCCESS
                            font.pixelSize: 10; font.bold: true; font.capitalization: Font.AllUppercase
                        }
                    }
                    Text { text: Analysis.metricLabel; color: root.faint; font.pixelSize: 10 }
                }
                ColumnLayout {
                    // inset 1px so the mat doesn't paint over the card's L/R border
                    x: 1
                    width: parent.width - 2
                    spacing: 0
                    // figure mat — the SURROUND (the area around the figure,
                    // outside its border) uses the darker mat colour.
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 360
                        color: root.mat
                        clip: true
                        // static metric figure: a rounded panel filled with the
                        // engine figureBg (so the padded image blends), sized to
                        // the figure aspect.  The image is inset by the corner
                        // radius so its square corners never reach the rounded
                        // corners — the surround shows through them (no masking).
                        Item {
                            id: figArea
                            anchors.fill: parent; anchors.margins: sc.sp4
                            visible: !Analysis.paired && Analysis.hasFigure
                            opacity: Analysis.busy ? 0.35 : 1.0
                            Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 180 } }
                            // hidden measure image → source aspect (uniform across
                            // panels, so every figFrame is the same size)
                            Image {
                                id: figMeasure
                                source: (Analysis.paired || !Analysis.hasFigure)
                                        ? "" : ("image://workspacefig/" + Analysis.figureToken)
                                visible: false; cache: false; asynchronous: true
                            }
                            readonly property real ar: figMeasure.sourceSize.height > 0
                                    ? (figMeasure.sourceSize.width / figMeasure.sourceSize.height) : 1.4
                            readonly property bool hbound: (width / height) > ar
                            readonly property real fw: hbound ? height * ar : width
                            readonly property real fh: hbound ? height : width / ar
                            Rectangle {
                                anchors.centerIn: parent
                                width: figArea.fw; height: figArea.fh
                                radius: root.gRad
                                color: Analysis.figureBg
                                border.width: 1; border.color: pal.BORDER
                                antialiasing: true
                                visible: figMeasure.sourceSize.width > 0
                                // Skeleton shimmer while it renders, then a blur-up
                                // reveal of the fresh figure (re-reveals each recompute).
                                FigureReveal {
                                    anchors.fill: parent; anchors.margins: root.gRad
                                    source: figMeasure.source
                                    cornerRadius: 0
                                }
                            }
                        }
                        // paired line plot — fades + zoom-settles in when the
                        // paired series arrive (matches the image branch's reveal)
                        Reveal {
                            anchors.fill: parent; anchors.margins: sc.sp4
                            visible: Analysis.paired
                            ready: Analysis.paired && Analysis.pairedSeries.length > 0
                            PairedPlot {
                                anchors.fill: parent
                            }
                        }
                        Text {
                            anchors.centerIn: parent
                            visible: !Analysis.paired && !Analysis.hasFigure
                            text: "Rendering…"; color: root.faint; font.pixelSize: sc.textSm
                        }
                    }
                    // legend + caption
                    Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
                    Flow {
                        Layout.fillWidth: true
                        Layout.leftMargin: sc.sp6; Layout.rightMargin: sc.sp6
                        Layout.topMargin: sc.sp5; Layout.bottomMargin: sc.sp5
                        spacing: sc.sp4
                        Repeater {
                            model: Analysis.legend
                            delegate: RowLayout {
                                required property var modelData
                                spacing: 5
                                Rectangle { width: 9; height: 9; radius: 2; color: modelData.color }
                                Text { text: modelData.name; color: pal.TXT_MUTED; font.pixelSize: 10 }
                            }
                        }
                        Item { width: sc.sp6 }
                        Text {
                            text: Analysis.caption; color: root.faint
                            font.pixelSize: 10; font.family: "Menlo"
                        }
                    }
                }
            }

            // headline metrics
            WCard {
                Layout.fillWidth: true
                visible: Analysis.hasStats
                title: "Headline metrics"
                enterDelay: 50
                GridLayout {
                    width: parent.width
                    columns: Math.max(2, Math.min(6, Math.floor(width / 150)))
                    columnSpacing: 20
                    rowSpacing: 16
                    Repeater {
                        model: Analysis.headline
                        delegate: ColumnLayout {
                            required property var modelData
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignTop
                            spacing: 2
                            readonly property bool jump: modelData.jump === true
                            RowLayout {
                                spacing: 4
                                Text {
                                    text: modelData.label; color: pal.TXT_MUTED; font.pixelSize: 11
                                }
                                Icon { visible: jump; name: "arrow-down-right"; size: 11; color: root.faint }
                            }
                            RowLayout {
                                spacing: 4
                                Text {
                                    text: modelData.value
                                    color: modelData.color !== "" ? modelData.color : pal.TXT
                                    font.pixelSize: 22; font.bold: true
                                }
                                Text {
                                    visible: modelData.unit !== ""
                                    text: modelData.unit; color: pal.TXT_MUTED
                                    font.pixelSize: 11; font.family: "Menlo"
                                }
                            }
                            TapHandler { enabled: jump; onTapped: sigCard.flash() }
                        }
                    }
                }
            }

            // plain-language verdict for the selected metric (results_format)
            Alert {
                Layout.fillWidth: true
                visible: (Analysis.metricVerdict.html || "") !== ""
                severity: Analysis.metricVerdict.severity || "info"
                text: Analysis.metricVerdict.html || ""
            }

            // stats + significance — side by side when there's room (design),
            // stacking only when the column is narrow.
            GridLayout {
                Layout.fillWidth: true
                visible: Analysis.hasStats
                columns: width > 660 ? 2 : 1
                columnSpacing: root.gGap
                rowSpacing: root.gGap
                StatsCard { Layout.alignment: Qt.AlignTop; enterDelay: 100 }
                SignificanceCard { id: sigCard; Layout.alignment: Qt.AlignTop; enterDelay: 100 }
            }

            // two-way mixed ANOVA (group × time) — only for factorial designs
            WCard {
                visible: Analysis.hasTwoway
                Layout.fillWidth: true
                title: "Two-way mixed ANOVA"
                enterDelay: 150
                ColumnLayout {
                    width: parent.width; spacing: sc.sp2
                    Text { text: "Group × time · per-replicate · Greenhouse–Geisser corrected"
                           color: root.faint; font.pixelSize: 10 }
                    Repeater {
                        model: Analysis.twowayRows
                        delegate: Rectangle {
                            Layout.fillWidth: true; implicitHeight: 42; radius: 8
                            color: modelData.sig ? Qt.rgba(0.337, 0.827, 0.392, 0.13) : pal.PANEL_ALT
                            border.width: 1
                            border.color: modelData.sig ? Qt.rgba(0.337, 0.827, 0.392, 0.28) : pal.BORDER
                            RowLayout {
                                anchors.fill: parent; anchors.leftMargin: sc.sp4
                                anchors.rightMargin: sc.sp4; spacing: sc.sp3
                                ColumnLayout {
                                    Layout.fillWidth: true; spacing: 1
                                    Text { text: modelData.effect; color: pal.TXT; font.pixelSize: 12; font.bold: true }
                                    Text { text: modelData.stat + (modelData.eta ? "  ·  " + modelData.eta : "")
                                           color: pal.TXT_MUTED; font.pixelSize: 10; font.family: "Menlo" }
                                }
                                Text { text: modelData.p; color: modelData.sig ? pal.STATUS_OK : pal.TXT_MUTED
                                       font.pixelSize: 11; font.family: "Menlo" }
                                Text { text: modelData.stars; color: modelData.sig ? pal.STATUS_OK : root.faint
                                       font.pixelSize: 12; font.bold: true; Layout.preferredWidth: 26
                                       horizontalAlignment: Text.AlignRight }
                            }
                        }
                    }
                    Text { visible: Analysis.twowayNote !== ""
                           text: Analysis.twowayNote; color: root.faint; font.pixelSize: 10
                           wrapMode: Text.WordWrap; Layout.fillWidth: true }
                }
            }

            // methods
            MethodsCard { visible: Analysis.hasStats }

            // report output destination (where Generate full report writes)
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp2
                Text { text: "Output"; color: pal.TXT_MUTED; font.pixelSize: 11 }
                Rectangle {
                    Layout.fillWidth: true; implicitHeight: 28; radius: 7
                    color: dirHov.hovered ? pal.PANEL_ALT : "transparent"
                    border.width: 1; border.color: pal.BORDER
                    RowLayout {
                        anchors.fill: parent; anchors.leftMargin: sc.sp3
                        anchors.rightMargin: sc.sp3; spacing: sc.sp2
                        Icon { name: "folder-open"; size: 13; color: pal.TXT_MUTED }
                        Text { Layout.fillWidth: true; text: Analysis.outputDir; color: pal.TXT_MUTED
                               font.pixelSize: 11; font.family: "Menlo"; elide: Text.ElideMiddle }
                        Text { text: "Change"; color: pal.ACC; font.pixelSize: 11 }
                    }
                    HoverHandler { id: dirHov; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: Analysis.chooseOutputDir() }
                }
                Rectangle {
                    Layout.preferredWidth: 150; implicitHeight: 28; radius: 7
                    color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                    TextInput {
                        anchors.fill: parent; anchors.leftMargin: sc.sp3; anchors.rightMargin: sc.sp3
                        verticalAlignment: TextInput.AlignVCenter
                        text: Analysis.cfg.outputStem; color: pal.TXT; font.pixelSize: 11; font.family: "Menlo"
                        onEditingFinished: Analysis.setCfg("outputStem", text)
                    }
                }
            }

            // action row
            RowLayout {
                Layout.fillWidth: true
                spacing: sc.sp2
                // Real engine: full multi-panel figure + Prism CSV + two-way CSV
                // + PDF report + results JSON, written to the output folder.
                Button {
                    variant: "primary"
                    icon: Analysis.reportBusy ? "loader-circle" : "git-merge"
                    spin: Analysis.reportBusy
                    text: Analysis.reportBusy ? "Generating report…" : "Generate full report"
                    enabled: !Analysis.reportBusy && Analysis.conditionCount >= 2
                    onClicked: Analysis.generateComparison()
                }
                Item { Layout.fillWidth: true }
                // compact icon toolbar — tooltips name each action on hover
                RowLayout {
                    spacing: sc.sp1
                    Button { variant: "secondary"; icon: "image"; tip: "Quick figure (PNG)"; onClicked: Analysis.exportFigure() }
                    Button { variant: "secondary"; icon: "table"; tip: "Quick stats (CSV)"; onClicked: Analysis.exportStats() }
                    Button { variant: "secondary"; icon: "clock"; tip: "Open previous comparison…"; onClicked: Analysis.openPreviousComparison() }
                    Button { variant: "secondary"; icon: "folder-open"; tip: "Open output folder"; onClicked: Analysis.openOutputFolder() }
                }
            }

            // report progress — real %, from compare_groups' progress_cb (loading
            // every replicate folder), then indeterminate while it renders + writes
            ColumnLayout {
                Layout.fillWidth: true
                visible: Analysis.reportBusy
                spacing: sc.sp1
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 6; radius: 3; clip: true
                    color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                    Rectangle {                       // determinate fill (load phase)
                        visible: Analysis.reportProgress >= 0
                        height: parent.height; radius: 3
                        width: Math.max(0, Math.min(1, Analysis.reportProgress)) * parent.width
                        gradient: Gradient {
                            orientation: Gradient.Horizontal
                            GradientStop { position: 0.0; color: pal.SUCCESS }
                            GradientStop { position: 1.0; color: pal.ACC }
                        }
                        Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                    }
                    IndeterminateShimmer { active: Analysis.reportBusy && Analysis.reportProgress < 0 }
                }
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp2
                    Text { Layout.fillWidth: true
                           text: Analysis.reportStatus || "Working…"
                           color: pal.TXT_MUTED; font.pixelSize: 11; elide: Text.ElideRight }
                    Text { visible: Analysis.reportProgress >= 0
                           text: Math.round(Analysis.reportProgress * 100) + "%"
                           color: pal.TXT_MUTED; font.pixelSize: 11; font.family: "Menlo" }
                }
            }
        }
    }

    // ════ panels view (per-condition pooled publication figure) ════
    Component {
        id: panelsView
        ColumnLayout {
            spacing: root.gGap
            WCard {
                Layout.fillWidth: true
                title: "Publication figure"
                count: Analysis.panelCount + " panels"
                pad: 0
                headerRight: RowLayout {
                    spacing: sc.sp2
                    Text { text: "group"; color: root.faint; font.pixelSize: 11 }
                    Repeater {
                        model: Analysis.panelConditions
                        delegate: Rectangle {
                            required property var modelData
                            required property int index
                            readonly property bool on: index === Analysis.panelCondIdx
                            implicitWidth: pcrow.implicitWidth + sc.sp3 * 2; implicitHeight: 24
                            radius: height / 2
                            color: on ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12) : pal.PANEL_ALT
                            border.width: 1; border.color: on ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.5) : pal.BORDER
                            RowLayout {
                                id: pcrow; anchors.centerIn: parent; spacing: 5
                                Rectangle { width: 8; height: 8; radius: 2; color: modelData.color }
                                Text { text: modelData.label.split(" · ").pop() + "  (" + modelData.n + ")"; color: on ? pal.ACC : pal.TXT_MUTED; font.pixelSize: 11; font.bold: on }
                            }
                            TapHandler { onTapped: Analysis.setPanelCond(index) }
                        }
                    }
                }
                ColumnLayout {
                    width: parent.width
                    spacing: 0
                    // hero
                    Rectangle {
                        Layout.fillWidth: true; Layout.preferredHeight: 340
                        color: root.mat; clip: true
                        FigureReveal {
                            anchors.fill: parent; anchors.margins: sc.sp4
                            cornerRadius: 0
                            source: Analysis.hasPanel ? ("image://workspacepanel/hero/" + Analysis.panelToken) : ""
                        }
                        Column {
                            anchors.centerIn: parent
                            visible: !Analysis.hasPanel
                            spacing: sc.sp2
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "Rendering…"; color: root.faint; font.pixelSize: sc.textSm
                            }
                            LoadingDots {
                                anchors.horizontalCenter: parent.horizontalCenter
                                tone: root.faint
                            }
                        }
                    }
                    Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
                    // hero caption
                    RowLayout {
                        Layout.fillWidth: true; Layout.margins: sc.sp3; spacing: sc.sp2
                        Text { text: Analysis.panelIndexLabel; color: root.faint; font.pixelSize: 11; font.family: "Menlo" }
                        Text { text: Analysis.panelHeroName; color: pal.TXT; font.pixelSize: 13; font.bold: true }
                        Rectangle {
                            implicitWidth: pcat.implicitWidth + sc.sp2 * 2; implicitHeight: 18; radius: 999
                            color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                            Text { id: pcat; anchors.centerIn: parent; text: Analysis.panelHeroCat; color: pal.ACC; font.pixelSize: 10; font.bold: true; font.capitalization: Font.AllUppercase }
                        }
                        Item { Layout.fillWidth: true }
                    }
                    // per-replicate selector — spatial/heat-map panels can't be
                    // averaged, so pick which folder's image to show.
                    RowLayout {
                        visible: Analysis.panelIsSpatial && Analysis.panelReplicates.length > 1
                        Layout.fillWidth: true
                        Layout.leftMargin: sc.sp3; Layout.rightMargin: sc.sp3; Layout.bottomMargin: sc.sp2
                        spacing: sc.sp2
                        Text { text: "replicate"; color: root.faint; font.pixelSize: 11; Layout.alignment: Qt.AlignVCenter }
                        Flow {
                            Layout.fillWidth: true; spacing: sc.sp2
                            Repeater {
                                model: Analysis.panelReplicates
                                delegate: Rectangle {
                                    required property var modelData
                                    required property int index
                                    readonly property bool on: index === Analysis.panelReplicateIdx
                                    implicitWidth: Math.min(150, rlbl.implicitWidth + sc.sp3 * 2); implicitHeight: 22
                                    radius: height / 2
                                    color: on ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12) : pal.PANEL_ALT
                                    border.width: 1; border.color: on ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.5) : pal.BORDER
                                    Text { id: rlbl; anchors.fill: parent; anchors.leftMargin: sc.sp3; anchors.rightMargin: sc.sp3
                                           verticalAlignment: Text.AlignVCenter; horizontalAlignment: Text.AlignHCenter
                                           text: (index + 1) + ". " + modelData.label
                                           color: on ? pal.ACC : pal.TXT_MUTED; font.pixelSize: 10; font.bold: on; elide: Text.ElideMiddle }
                                    TapHandler { onTapped: Analysis.setPanelReplicate(index) }
                                    HoverHandler { cursorShape: Qt.PointingHandCursor }
                                }
                            }
                        }
                    }
                    Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
                    // thumbnail grid grouped by category
                    ColumnLayout {
                        Layout.fillWidth: true; Layout.margins: sc.sp3; spacing: sc.sp3
                        Repeater {
                            model: Analysis.panelCategories
                            delegate: ColumnLayout {
                                required property var modelData
                                Layout.fillWidth: true; spacing: sc.sp2
                                Text {
                                    text: modelData.cat + " · " + modelData.count
                                    color: root.faint; font.pixelSize: 10; font.bold: true
                                    font.letterSpacing: 1.0; font.capitalization: Font.AllUppercase
                                }
                                Flow {
                                    Layout.fillWidth: true; spacing: sc.sp2
                                    Repeater {
                                        model: modelData.items
                                        delegate: ColumnLayout {
                                            required property var modelData
                                            readonly property bool on: modelData.idx === Analysis.panelSel
                                            width: 120; spacing: 3
                                            Rectangle {
                                                Layout.preferredWidth: 120; Layout.preferredHeight: 120
                                                radius: sc.radiusSm; color: root.mat; clip: true
                                                border.width: 1; border.color: on ? pal.ACC : pal.BORDER
                                                Image {                       // live thumbnail preview
                                                    anchors.fill: parent; anchors.margins: 2
                                                    fillMode: Image.PreserveAspectFit
                                                    cache: true; asynchronous: true; smooth: true
                                                    sourceSize.width: 232; sourceSize.height: 232
                                                    source: "image://workspacepanel/thumb/" + Analysis.panelCondIdx
                                                            + "/" + modelData.idx + "/" + Analysis.panelDataRev
                                                            + "_" + Analysis.panelGroupRev
                                                }
                                                Rectangle {                   // number badge
                                                    anchors { left: parent.left; top: parent.top; margins: 4 }
                                                    radius: 3; color: Qt.rgba(0, 0, 0, 0.6)
                                                    width: nlbl.implicitWidth + 7; height: nlbl.implicitHeight + 3
                                                    Text { id: nlbl; anchors.centerIn: parent
                                                           text: String(modelData.idx + 1).padStart(2, "0")
                                                           color: on ? pal.ACC : "#cdd5df"; font.pixelSize: 9
                                                           font.bold: true; font.family: "Menlo" }
                                                }
                                            }
                                            Text { Layout.preferredWidth: 120; text: modelData.name; color: on ? pal.TXT : pal.TXT_MUTED; font.pixelSize: 10; elide: Text.ElideRight }
                                            TapHandler { onTapped: Analysis.setPanelSel(modelData.idx) }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // action row
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp2
                Button { variant: "secondary"; icon: "download"; text: "Export panel (PDF)"; onClicked: Analysis.exportFigure() }
                Button { variant: "secondary"; icon: "images"; text: "Export all (PDF)"; onClicked: Analysis.exportFigure() }
                Item { Layout.fillWidth: true }
                Button { variant: "secondary"; icon: "folder-open"; text: "Open output folder"; onClicked: Analysis.openOutputFolder() }
            }
        }
    }

    // ════ paired line plot (Canvas) ════
    component PairedPlot: Canvas {
        id: pc
        readonly property var series: Analysis.pairedSeries
        readonly property var axis: Analysis.pairedAxis
        onSeriesChanged: requestPaint()
        onAxisChanged: requestPaint()
        Connections { target: Analysis; function onResultsChanged() { pc.requestPaint() } }
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var W = width, H = height, mL = 54, mR = 16, mT = 16, mB = 38;
            if (!series || series.length === 0 || !axis || axis.length === 0) return;
            // y-range
            var lo = Infinity, hi = -Infinity;
            for (var i = 0; i < series.length; i++)
                for (var j = 0; j < series[i].points.length; j++) {
                    var y = series[i].points[j].y;
                    if (y < lo) lo = y; if (y > hi) hi = y;
                }
            if (!isFinite(lo) || !isFinite(hi)) return;
            var padv = (hi - lo) * 0.18 || Math.abs(hi) * 0.18 || 1; lo -= padv; hi += padv;
            function xOf(k) { return mL + (axis.length <= 1 ? 0.5 : k / (axis.length - 1)) * (W - mL - mR); }
            function yOf(v) { return mT + (1 - (v - lo) / (hi - lo || 1)) * (H - mT - mB); }
            // gridlines + y labels
            ctx.strokeStyle = pal.BORDER; ctx.fillStyle = root.faint;   // theme-aware gridlines + labels
            ctx.font = "10px Menlo"; ctx.textAlign = "right";
            for (var t = 0; t <= 4; t++) {
                var v = lo + (t / 4) * (hi - lo), yy = yOf(v);
                ctx.beginPath(); ctx.moveTo(mL, yy); ctx.lineTo(W - mR, yy); ctx.stroke();
                ctx.fillText(v.toFixed(3), mL - 6, yy + 3);
            }
            // x labels
            ctx.textAlign = "center"; ctx.fillStyle = pal.TXT_MUTED; ctx.font = "11px sans-serif";
            for (var a = 0; a < axis.length; a++) ctx.fillText(axis[a], xOf(a), H - mB + 20);
            // subject lines
            for (var s = 0; s < series.length; s++) {
                var pts = series[s].points, col = series[s].color;
                ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2; ctx.globalAlpha = 0.75;
                ctx.beginPath();
                for (var p = 0; p < pts.length; p++) {
                    var k = axis.indexOf(pts[p].x);
                    var px = xOf(k), py = yOf(pts[p].y);
                    if (p === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                }
                ctx.stroke(); ctx.globalAlpha = 1;
                for (var p2 = 0; p2 < pts.length; p2++) {
                    var k2 = axis.indexOf(pts[p2].x);
                    ctx.beginPath(); ctx.arc(xOf(k2), yOf(pts[p2].y), 5, 0, 6.2832); ctx.fill();
                }
            }
        }
    }

    // ════ stats card ════
    component StatsCard: WCard {
        Layout.fillWidth: true
        title: "Group statistics"
        ColumnLayout {
            width: parent.width
            spacing: 0
            // header row
            RowLayout {
                Layout.fillWidth: true
                Layout.bottomMargin: sc.sp4
                spacing: sc.sp5
                Text { Layout.fillWidth: true; text: "Condition"; color: pal.TXT_MUTED; font.pixelSize: 10; font.bold: true; font.capitalization: Font.AllUppercase }
                Text { Layout.preferredWidth: 54; horizontalAlignment: Text.AlignRight; text: "n"; color: pal.TXT_MUTED; font.pixelSize: 10; font.bold: true; font.capitalization: Font.AllUppercase }
                Text { Layout.preferredWidth: 90; horizontalAlignment: Text.AlignRight; text: Analysis.metricLabel; color: pal.ACC; font.pixelSize: 10; font.bold: true; font.capitalization: Font.AllUppercase }
                Text { Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight; text: "compare"; color: pal.TXT_MUTED; font.pixelSize: 10; font.bold: true; font.capitalization: Font.AllUppercase }
            }
            Repeater {
                model: Analysis.statsRows
                delegate: ColumnLayout {
                    required property var modelData
                    Layout.fillWidth: true
                    spacing: 0
                    // per-row dividing line — segmented per column with gaps at
                    // the column boundaries, like the prototype (each cell has its
                    // own borderTop and the grid has a column gap).
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: sc.sp5
                        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: pal.BORDER }
                        Rectangle { Layout.preferredWidth: 54; Layout.preferredHeight: 1; color: pal.BORDER }
                        Rectangle { Layout.preferredWidth: 90; Layout.preferredHeight: 1; color: pal.BORDER }
                        Rectangle { Layout.preferredWidth: 80; Layout.preferredHeight: 1; color: pal.BORDER }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.topMargin: sc.sp8; Layout.bottomMargin: sc.sp8
                        spacing: sc.sp5
                        RowLayout {
                            Layout.fillWidth: true; spacing: 6
                            Rectangle { width: 9; height: 9; radius: 2; color: modelData.color }
                            Text { Layout.fillWidth: true; text: modelData.label; color: pal.TXT_MUTED; font.pixelSize: 12; elide: Text.ElideRight }
                        }
                        Text { Layout.preferredWidth: 54; horizontalAlignment: Text.AlignRight; text: modelData.n; color: pal.TXT_MUTED; font.pixelSize: 12; font.family: "Menlo" }
                        RowLayout {
                            Layout.preferredWidth: 90; spacing: 3; layoutDirection: Qt.RightToLeft
                            Text { text: modelData.err; visible: modelData.err !== ""; color: root.faint; font.pixelSize: 11; font.family: "Menlo" }
                            Text { text: modelData.value; color: pal.TXT; font.pixelSize: 12; font.family: "Menlo" }
                        }
                        Rectangle {
                            Layout.preferredWidth: 80; height: 7; radius: 4; color: pal.PANEL_ALT
                            Rectangle {
                                height: parent.height; radius: 4
                                width: Math.max(4, modelData.barFrac * parent.width)
                                color: modelData.color
                                Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 280 } }
                            }
                        }
                    }
                }
            }
        }
    }

    // ════ significance card ════
    component SignificanceCard: WCard {
        id: sc2
        function flash() { flashAnim.restart() }
        Layout.fillWidth: true
        title: "Pairwise significance"
        headerRight: Text {
            text: (Analysis.cfg.test || "") + " · " + (Analysis.cfg.correction === "None" ? "uncorrected" : (Analysis.cfg.correction || "").split(" ")[0]) + (Analysis.paired ? " · paired" : "")
            color: root.faint; font.pixelSize: 10
            width: Math.min(implicitWidth, 230); elide: Text.ElideRight
        }
        Rectangle {
            id: flashRect; anchors.fill: parent; radius: root.gRad
            color: pal.ACC; opacity: 0
            NumberAnimation { id: flashAnim; target: flashRect; property: "opacity"; from: 0.18; to: 0; duration: 700 }
        }
        ColumnLayout {
            width: parent.width
            spacing: sc.sp2
            Repeater {
                model: Analysis.significanceRows
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: sigRow.implicitHeight + sc.sp3
                    radius: sc.radiusSm
                    color: modelData.sig ? Qt.rgba(0.337, 0.827, 0.392, 0.13) : pal.PANEL_ALT
                    border.width: 1
                    border.color: modelData.sig ? Qt.rgba(0.337, 0.827, 0.392, 0.28) : pal.BORDER
                    RowLayout {
                        id: sigRow
                        anchors.fill: parent; anchors.margins: sc.sp2; anchors.leftMargin: sc.sp3; anchors.rightMargin: sc.sp3
                        spacing: 8
                        Rectangle { width: 8; height: 8; radius: 2; color: modelData.aColor }
                        Text { text: "vs"; color: pal.TXT_MUTED; font.pixelSize: 11 }
                        Rectangle { width: 8; height: 8; radius: 2; color: modelData.bColor }
                        ColumnLayout {
                            Layout.fillWidth: true; Layout.leftMargin: sc.sp2; spacing: 0
                            Text { Layout.fillWidth: true; text: modelData.label; color: pal.TXT; font.pixelSize: 11; elide: Text.ElideRight }
                            Text { text: modelData.delta + " · " + modelData.mag; color: modelData.magColor; font.pixelSize: 10; font.family: "Menlo" }
                        }
                        Text { text: modelData.p; color: modelData.sig ? pal.STATUS_OK : pal.TXT_MUTED; font.pixelSize: 11; font.family: "Menlo" }
                        Text { text: modelData.stars; color: modelData.sig ? pal.STATUS_OK : root.faint; font.pixelSize: 12; font.bold: true; Layout.preferredWidth: 26; horizontalAlignment: Text.AlignRight }
                    }
                }
            }
            Text {
                Layout.fillWidth: true
                Layout.topMargin: sc.sp4
                text: "α = " + Analysis.cfg.alpha + " · " + Analysis.significanceRows.length + " pair" + (Analysis.significanceRows.length === 1 ? "" : "s") + " · δ = Cliff's effect size"
                color: root.faint; font.pixelSize: 10
            }
        }
    }

    // ════ methods card ════
    component MethodsCard: Rectangle {
        Layout.fillWidth: true
        color: pal.PANEL; radius: root.gRad; border.width: 1; border.color: pal.BORDER
        implicitHeight: mrow.implicitHeight + root.gPad * 2
        RowLayout {
            id: mrow
            anchors.fill: parent; anchors.margins: root.gPad
            spacing: sc.sp3
            Icon { name: "quote"; size: 15; color: root.faint; Layout.alignment: Qt.AlignTop }
            ColumnLayout {
                Layout.fillWidth: true; spacing: 3
                Text { text: "METHODS"; color: root.faint; font.pixelSize: 10; font.bold: true; font.letterSpacing: 1.0 }
                Text { Layout.fillWidth: true; text: Analysis.methods; color: pal.TXT; font.pixelSize: 12; wrapMode: Text.WordWrap; lineHeight: 1.4 }
            }
            Button { variant: "secondary"; icon: "copy"; text: "Copy"; onClicked: Analysis.copyMethods() }
        }
    }

    // ════ conditions card ════
    component ConditionsCard: WCard {
        Layout.fillWidth: true
        title: "Conditions"
        count: Analysis.conditionCount + " / " + Analysis.maxConditions
        headerRight: RowLayout {
            visible: Analysis.hasTimepointsSet
            spacing: 5
            Icon { name: "git-merge"; size: 12; color: root.faint }
            Text {
                text: (Analysis.cfg.groupBy || "").indexOf("Timepoint") === 0 ? "paired by timepoint" : "timepoints set"
                color: root.faint; font.pixelSize: 10
            }
        }
        ColumnLayout {
            width: parent.width
            spacing: sc.sp5
            TimepointManager {}
            Repeater {
                model: Analysis.conditions
                delegate: GroupRow { required property var modelData; cond: modelData }
            }
            // Add condition — dashed, like the design
            Rectangle {
                visible: Analysis.conditionCount < Analysis.maxConditions
                Layout.fillWidth: true; Layout.preferredHeight: 40
                radius: 10; color: addCondHover.hovered ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.06) : "transparent"
                DashedRect { anchors.fill: parent; radius: 10; stroke: pal.BORDER_HI }
                RowLayout {
                    anchors.centerIn: parent; spacing: 7
                    Icon { name: "plus"; size: 14; color: pal.ACC }
                    Text { text: "Add condition"; color: pal.ACC; font.pixelSize: 12; font.bold: true }
                }
                HoverHandler { id: addCondHover; cursorShape: Qt.PointingHandCursor }
                TapHandler { onTapped: Analysis.addCondition() }
            }
        }
    }

    // ── timepoints manager ──
    component TimepointManager: Rectangle {
        id: tpManager
        Layout.fillWidth: true
        property bool adding: false
        color: root.cSunken; radius: 10; border.width: 1; border.color: pal.BORDER
        implicitHeight: tpc.implicitHeight + sc.sp4 * 2
        ColumnLayout {
            id: tpc
            anchors { left: parent.left; right: parent.right; top: parent.top; margins: sc.sp4 }
            spacing: sc.sp3
            RowLayout {
                Layout.fillWidth: true; spacing: 6
                Icon { name: "clock"; size: 12; color: root.faint }
                Text { text: "Timepoints"; color: pal.TXT_MUTED; font.pixelSize: 11; font.bold: true }
                Item { Layout.fillWidth: true }
                Text { text: "your own labels — assign per condition"; color: root.faint
                       font.pixelSize: 10; elide: Text.ElideRight }
            }
            Flow {
                Layout.fillWidth: true; spacing: 6
                Repeater {
                    model: Analysis.timepoints
                    delegate: Rectangle {
                        required property var modelData
                        implicitWidth: tprow.implicitWidth + sc.sp4 * 2
                        implicitHeight: 26; radius: 999
                        color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                        RowLayout {
                            id: tprow; anchors.centerIn: parent; spacing: 6
                            Rectangle { width: 8; height: 8; radius: 4; color: modelData.colorHex }
                            Text { text: modelData.name; color: pal.TXT; font.pixelSize: 11 }
                            Icon {
                                name: "x"; size: 12; color: root.faint
                                TapHandler {
                                    onTapped: {
                                        var nm = modelData.name
                                        confirmDelete.title = "Remove timepoint?"
                                        confirmDelete.message = "Remove the “" + nm + "” timepoint? Conditions assigned to it become unassigned."
                                        confirmDelete.confirmText = "Remove timepoint"
                                        confirmDelete.action = function() { Analysis.removeTimepoint(nm) }
                                        confirmDelete.open()
                                    }
                                }
                            }
                        }
                    }
                }
                // add control
                Rectangle {
                    visible: !tpManager.adding
                    implicitWidth: addrow.implicitWidth + sc.sp3 * 2
                    implicitHeight: 26; radius: 999
                    color: "transparent"
                    DashedRect { anchors.fill: parent; radius: height / 2; stroke: pal.BORDER_HI }
                    RowLayout {
                        id: addrow; anchors.centerIn: parent; spacing: 5
                        Icon { name: "plus"; size: 12; color: pal.ACC }
                        Text { text: "New timepoint"; color: pal.ACC; font.pixelSize: 11; font.bold: true }
                    }
                    TapHandler { onTapped: { tpManager.adding = true; tpInput.forceActiveFocus() } }
                }
                Rectangle {
                    visible: tpManager.adding
                    implicitWidth: 130; implicitHeight: 24; radius: 999
                    color: pal.PANEL; border.width: 1; border.color: pal.ACC
                    TextInput {
                        id: tpInput
                        anchors.fill: parent; anchors.leftMargin: sc.sp3; anchors.rightMargin: sc.sp3
                        verticalAlignment: TextInput.AlignVCenter
                        color: pal.TXT; font.pixelSize: 11; clip: true
                        onAccepted: { if (text.trim() !== "") Analysis.addTimepoint(text.trim()); text = ""; tpManager.adding = false }
                        Keys.onEscapePressed: { text = ""; tpManager.adding = false }
                    }
                }
            }
        }
    }

    // ── one condition row ──
    component GroupRow: Rectangle {
        id: gr
        property var cond
        Layout.fillWidth: true
        // DropTarget reaction: border→accent + 1% scale-up + faint fill while a
        // folder is dragged over; a SUCCESS flash when one lands.
        readonly property bool dragOver: cardDrop.containsDrag
        color: dragOver ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.06) : root.cSunken
        radius: 10
        border.width: dragOver ? 2 : 1
        border.color: dragOver ? pal.ACC : pal.BORDER
        scale: dragOver ? 1.01 : 1.0
        Behavior on color        { ColorAnimation  { duration: Theme.reducedMotion ? 0 : 120 } }
        Behavior on border.color { ColorAnimation  { duration: Theme.reducedMotion ? 0 : 120 } }
        Behavior on scale        { NumberAnimation { duration: Theme.reducedMotion ? 0 : 120; easing.type: Easing.OutCubic } }
        implicitHeight: grc.implicitHeight + sc.sp5 * 2
        Rectangle {                              // landed-flash overlay (keeps gr.color binding intact)
            id: landOverlay
            anchors.fill: parent; radius: parent.radius
            color: pal.SUCCESS; opacity: 0
            SequentialAnimation {
                id: landFlash
                NumberAnimation { target: landOverlay; property: "opacity"; to: 0.12; duration: Theme.reducedMotion ? 0 : 120 }
                PauseAnimation { duration: Theme.reducedMotion ? 0 : 260 }
                NumberAnimation { target: landOverlay; property: "opacity"; to: 0; duration: Theme.reducedMotion ? 0 : 240 }
            }
        }
        // the whole condition card is a drop target
        DropArea {
            id: cardDrop
            anchors.fill: parent
            onDropped: (drop) => {
                if (drop.hasUrls) { Analysis.addFolders(gr.cond.id, drop.urls); drop.accept(); if (!Theme.reducedMotion) landFlash.restart() }
            }
        }
        ColumnLayout {
            id: grc
            anchors { left: parent.left; right: parent.right; top: parent.top; margins: sc.sp5 }
            spacing: sc.sp4
            // header
            RowLayout {
                Layout.fillWidth: true; spacing: 8
                Rectangle { width: 14; height: 14; radius: 4; color: gr.cond.colorHex }
                TextInput {
                    Layout.fillWidth: true
                    text: gr.cond.name; color: pal.TXT; font.pixelSize: 12; font.bold: true
                    onEditingFinished: Analysis.setConditionName(gr.cond.id, text)
                }
                Rectangle {
                    implicitWidth: fcount.implicitWidth + sc.sp2 * 2; implicitHeight: 20; radius: 999
                    color: gr.cond.ready ? Qt.rgba(1,1,1,0.05) : "transparent"
                    border.width: 1; border.color: pal.BORDER
                    Text {
                        id: fcount; anchors.centerIn: parent
                        text: gr.cond.activeFolders + "/" + gr.cond.totalFolders + " folder" + (gr.cond.totalFolders === 1 ? "" : "s")
                        color: gr.cond.ready ? pal.TXT : root.faint; font.pixelSize: 10; font.bold: true
                    }
                }
                Icon {
                    name: "x"; size: 14; color: root.faint
                    opacity: Analysis.conditionCount > 2 ? 1 : 0.35
                    TapHandler {
                        enabled: Analysis.conditionCount > 2
                        onTapped: {
                            var cid = gr.cond.id
                            confirmDelete.title = "Remove condition?"
                            confirmDelete.message = "Remove “" + gr.cond.name + "” and unassign its run folders? This can't be undone."
                            confirmDelete.confirmText = "Remove condition"
                            confirmDelete.action = function() { Analysis.removeCondition(cid) }
                            confirmDelete.open()
                        }
                    }
                }
            }
            // timepoint selector
            RowLayout {
                Layout.fillWidth: true; spacing: 8
                Icon { name: "clock"; size: 12; color: root.faint }
                Text { text: "Timepoint"; color: root.faint; font.pixelSize: 10 }
                Select {
                    pill: true
                    fillColor: pal.PANEL_ALT
                    dotColor: gr.cond.phaseColor
                    dimText: gr.cond.phase === "—"
                    Layout.preferredHeight: 26
                    Layout.preferredWidth: 138
                    model: { var m = ["Unassigned"]; for (var i = 0; i < Analysis.timepoints.length; i++) m.push(Analysis.timepoints[i].name); return m }
                    currentIndex: { var p = gr.cond.phase; if (p === "—") return 0; for (var i = 0; i < Analysis.timepoints.length; i++) if (Analysis.timepoints[i].name === p) return i + 1; return 0 }
                    onPicked: (t) => Analysis.setConditionPhase(gr.cond.id, t === "Unassigned" ? "—" : t)
                }
                Item { Layout.fillWidth: true }
            }
            // folder chips — wrapping flow like the design (short run ids pack
            // several per row).  Each chip caps its width and elides the name in
            // the middle, so long real filenames never overflow the card.
            Flow {
                Layout.fillWidth: true; spacing: 6
                visible: gr.cond.folders.length > 0
                Repeater {
                    model: gr.cond.folders
                    delegate: Rectangle {
                        required property var modelData
                        readonly property color qcC: modelData.qc === "error" ? pal.DANGER : modelData.qc === "warn" ? pal.WARN : pal.SUCCESS
                        implicitWidth: Math.min(chrow.implicitWidth + sc.sp3 * 2, 230)
                        implicitHeight: 26; radius: sc.radiusSm
                        color: pal.PANEL_ALT; opacity: modelData.excluded ? 0.5 : 1
                        border.width: 1
                        border.color: modelData.qc === "error" ? Qt.rgba(0.878, 0.322, 0.322, 0.4) : pal.BORDER
                        RowLayout {
                            id: chrow
                            anchors.fill: parent
                            anchors.leftMargin: sc.sp3; anchors.rightMargin: sc.sp3
                            spacing: 6
                            Rectangle {
                                width: 6; height: 6; radius: 3; color: qcC; Layout.alignment: Qt.AlignVCenter
                                TapHandler { onTapped: Analysis.toggleFolder(gr.cond.id, modelData.id) }
                            }
                            Text {
                                Layout.fillWidth: true
                                text: modelData.id; color: pal.TXT_MUTED; font.pixelSize: 10; font.family: "Menlo"
                                font.strikeout: modelData.excluded; elide: Text.ElideMiddle
                                TapHandler { onTapped: Analysis.toggleFolder(gr.cond.id, modelData.id) }
                            }
                            Text { text: modelData.n; color: root.faint; font.pixelSize: 10; font.family: "Menlo" }
                            Icon {
                                name: "x"; size: 11; color: root.faint
                                TapHandler { onTapped: Analysis.removeFolder(gr.cond.id, modelData.id) }
                            }
                        }
                    }
                }
            }
            // drop zone — persistent dashed outline; highlights while a folder
            // is dragged over the card.
            Rectangle {
                Layout.fillWidth: true; Layout.preferredHeight: 40
                radius: 8
                color: cardDrop.containsDrag ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.10) : "transparent"
                DashedRect {
                    anchors.fill: parent; radius: 8
                    stroke: cardDrop.containsDrag ? pal.ACC : pal.BORDER_HI
                }
                RowLayout {
                    anchors.centerIn: parent; spacing: 6
                    Icon { name: "folder-plus"; size: 13; color: cardDrop.containsDrag ? pal.ACC : pal.TXT_MUTED }
                    Text { text: "Drop run folders, or"; color: pal.TXT_MUTED; font.pixelSize: 12 }
                    Text {
                        text: "Browse…"; color: pal.ACC; font.pixelSize: 12; font.bold: true
                        TapHandler { onTapped: Analysis.browseAddFolder(gr.cond.id) }
                    }
                }
            }
        }
    }

    // ════ settings card ════
    // ── experimental design + stats recommendations (ported from the Widgets
    //    "design & recommended settings" panel) ─────────────────────────────
    component DesignCard: WCard {
        Layout.fillWidth: true
        visible: Analysis.designSummary.ready
        title: "Experimental design"
        ColumnLayout {
            width: parent.width
            spacing: sc.sp5

            // ── 1 · your experimental design ──
            ColumnLayout {
                Layout.fillWidth: true; spacing: sc.sp3
                Flow {
                    Layout.fillWidth: true
                    spacing: sc.sp2
                    Repeater {
                        model: Analysis.designSummary.groups
                        delegate: Rectangle {
                            required property var modelData
                            radius: sc.radiusPill; height: 26
                            width: gRow.implicitWidth + sc.sp4 * 2
                            color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                            RowLayout {
                                id: gRow; anchors.centerIn: parent; spacing: sc.sp2
                                Rectangle { width: 8; height: 8; radius: 4
                                            color: modelData.color || pal.ACC
                                            Layout.alignment: Qt.AlignVCenter }
                                Text { text: modelData.name; color: pal.TXT
                                       font.pixelSize: sc.textSm; font.bold: true }
                                Text { text: "n=" + modelData.n; color: root.faint
                                       font.pixelSize: sc.textXs; font.family: "Menlo" }
                            }
                        }
                    }
                }
                Text {
                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                    text: Analysis.designSummary.description
                    color: pal.TXT_MUTED; font.pixelSize: sc.textSm; lineHeight: 1.3
                }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }

            // ── 2 · recommended for your data ──
            ColumnLayout {
                Layout.fillWidth: true; spacing: sc.sp3
                Text { text: "Recommended for your data"; color: pal.TXT_MUTED
                       font.pixelSize: 10; font.bold: true; font.letterSpacing: 0.8 }
                Repeater {
                    model: Analysis.recommendations
                    delegate: Rectangle {
                        required property var modelData
                        readonly property color tone: modelData.tone === "ok" ? pal.SUCCESS
                                                     : modelData.tone === "warn" ? pal.WARN : pal.ACC
                        Layout.fillWidth: true
                        radius: sc.radiusMd
                        color: pal.PANEL_ALT
                        // left accent stripe (full-side border → square corners on that edge)
                        Rectangle { anchors.left: parent.left; anchors.top: parent.top
                                    anchors.bottom: parent.bottom; width: 2; radius: 1; color: tone }
                        implicitHeight: recRow.implicitHeight + sc.sp4 * 2
                        RowLayout {
                            id: recRow
                            anchors.fill: parent
                            anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                            anchors.topMargin: sc.sp4; anchors.bottomMargin: sc.sp4
                            spacing: sc.sp4
                            Icon {
                                name: modelData.tone === "ok" ? "circle-check"
                                    : modelData.tone === "warn" ? "triangle-alert" : "info"
                                size: 15; color: tone; Layout.alignment: Qt.AlignTop
                            }
                            Text {
                                Layout.fillWidth: true; wrapMode: Text.WordWrap
                                text: modelData.text; color: pal.TXT
                                font.pixelSize: sc.textSm; lineHeight: 1.3
                            }
                        }
                    }
                }
            }

            Button {
                Layout.alignment: Qt.AlignRight
                variant: "primary"; text: "Apply recommended settings"; icon: "sparkles"
                onClicked: Analysis.applyRecommended()
            }
        }
    }

    component SettingsCard: WCard {
        Layout.fillWidth: true
        title: "Comparison settings"
        headerRight: Icon { name: "sliders-horizontal"; size: 13; color: pal.TXT_MUTED }
        ColumnLayout {
            width: parent.width
            spacing: sc.sp8
            // (the "Recommend settings" affordance now lives in the richer
            //  DesignCard above — kept DRY to avoid two Apply buttons)
            // presets
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp4
                Select {
                    id: presetSel
                    Layout.fillWidth: true
                    // index 0 is the placeholder; >0 is a real saved preset.
                    readonly property bool hasSel: currentIndex > 0 && Analysis.presets.length > 0
                    model: { var m = [Analysis.presets.length ? "Load preset…" : "No saved presets"]; for (var i = 0; i < Analysis.presets.length; i++) m.push(Analysis.presets[i].name); return m }
                    onPicked: (t) => { if (t !== "Load preset…" && t !== "No saved presets") Analysis.loadPreset(t) }
                }
                Button { variant: "secondary"; icon: "bookmark-plus"; text: "Save"; onClicked: Analysis.savePreset() }
                IconButton {
                    icon: "x"; tip: "Delete selected preset"; danger: true
                    enabled: presetSel.hasSel; opacity: enabled ? 1 : 0.35
                    onClicked: { Analysis.deletePreset(presetSel.currentText); presetSel.currentIndex = 0 }
                }
            }
            Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
            // group by / normalisation
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp6
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 4
                    Text { text: "Group by"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    Select {
                        Layout.fillWidth: true
                        model: ["Condition", "Timepoint (pre/post)", "Cell line", "Replicate"]
                        currentIndex: Math.max(0, model.indexOf(Analysis.cfg.groupBy))
                        onPicked: (t) => Analysis.setCfg("groupBy", t)
                    }
                }
            }
            // test
            ColumnLayout {
                Layout.fillWidth: true; spacing: 4
                Text { text: "Statistical test"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                Select {
                    Layout.fillWidth: true
                    model: ["Auto", "Mann–Whitney U", "Brunner–Munzel", "Permutation", "Welch's t-test", "Kruskal–Wallis", "Wilcoxon signed-rank", "Paired t-test"]
                    currentIndex: Math.max(0, model.indexOf(Analysis.cfg.test))
                    onPicked: (t) => Analysis.setCfg("test", t)
                }
            }
            // correction / alpha
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp6
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 4
                    Text { text: "Correction"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    Select {
                        Layout.fillWidth: true
                        model: ["None", "Bonferroni", "Holm", "Šidák", "Hochberg", "Benjamini–Hochberg (FDR)"]
                        currentIndex: Math.max(0, model.indexOf(Analysis.cfg.correction))
                        onPicked: (t) => Analysis.setCfg("correction", t)
                    }
                }
                ColumnLayout {
                    Layout.preferredWidth: 70; spacing: 4
                    Text { text: "α"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    Rectangle {
                        Layout.fillWidth: true; implicitHeight: 28; radius: 7
                        color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                        TextInput {
                            anchors.fill: parent; anchors.leftMargin: sc.sp3
                            verticalAlignment: TextInput.AlignVCenter
                            text: Analysis.cfg.alpha; color: pal.TXT; font.pixelSize: 12; font.family: "Menlo"
                            onEditingFinished: Analysis.setCfg("alpha", text)
                        }
                    }
                }
            }
            // ── advanced statistics (engine: post-hoc / Dunnett / CI / TOST) ──
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp6
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 4
                    Text { text: "Post-hoc (3+ groups)"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    Select {
                        Layout.fillWidth: true
                        readonly property var vals: ["auto", "games_howell", "dunn", "tukey"]
                        model: ["Automatic", "Games–Howell", "Dunn", "Tukey HSD"]
                        currentIndex: Math.max(0, vals.indexOf(Analysis.cfg.posthoc))
                        onPicked: (t) => Analysis.setCfg("posthoc", vals[model.indexOf(t)])
                    }
                }
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 4
                    Text { text: "Omnibus (3+ groups)"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    Select {
                        Layout.fillWidth: true
                        readonly property var vals: ["welch", "oneway", "auto"]
                        model: ["Welch ANOVA", "One-way ANOVA", "Auto"]
                        currentIndex: Math.max(0, vals.indexOf(Analysis.cfg.anova3plus))
                        onPicked: (t) => Analysis.setCfg("anova3plus", vals[model.indexOf(t)])
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp6
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 4
                    Text { text: "Control group"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    Select {
                        Layout.fillWidth: true
                        readonly property var vals: ["—"].concat(Analysis.groupLabels)
                        model: vals
                        currentIndex: Math.max(0, vals.indexOf(Analysis.cfg.control_group || "—"))
                        onPicked: (t) => Analysis.setCfg("control_group", t === "—" ? "" : t)
                    }
                }
                ColumnLayout {
                    Layout.preferredWidth: 120; spacing: 4
                    Text { text: "Dunnett vs control"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    RowLayout {
                        Layout.fillWidth: true; Layout.preferredHeight: 28
                        Switch {
                            checked: Analysis.cfg.dunnett === true
                            enabled: (Analysis.cfg.control_group || "") !== ""
                            onToggled: (c) => Analysis.setCfg("dunnett", c)
                        }
                        Item { Layout.fillWidth: true }
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp6
                ColumnLayout {
                    Layout.preferredWidth: 90; spacing: 4
                    Text { text: "CI level"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                    Rectangle {
                        Layout.fillWidth: true; implicitHeight: 28; radius: 7
                        color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                        TextInput {
                            anchors.fill: parent; anchors.leftMargin: sc.sp3
                            verticalAlignment: TextInput.AlignVCenter
                            text: Analysis.cfg.ci_level; color: pal.TXT; font.pixelSize: 12; font.family: "Menlo"
                            onEditingFinished: Analysis.setCfg("ci_level", text)
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true; Layout.alignment: Qt.AlignBottom; spacing: sc.sp2
                    Switch { checked: Analysis.cfg.across_metric_correction === true
                             onToggled: (c) => Analysis.setCfg("across_metric_correction", c) }
                    Text { text: "Correct across all metrics"; color: pal.TXT; font.pixelSize: 12 }
                }
            }
            RowLayout {
                Layout.fillWidth: true; spacing: sc.sp2
                Switch { checked: Analysis.cfg.equivalence_tost === true
                         onToggled: (c) => Analysis.setCfg("equivalence_tost", c) }
                Text { text: "Equivalence test (TOST)"; color: pal.TXT; font.pixelSize: 12 }
                Item { Layout.fillWidth: true }
                Text { text: "margin (SD)"; color: pal.TXT_MUTED; font.pixelSize: 11
                       visible: Analysis.cfg.equivalence_tost === true }
                Rectangle {
                    visible: Analysis.cfg.equivalence_tost === true
                    Layout.preferredWidth: 52; implicitHeight: 26; radius: 7
                    color: pal.PANEL_ALT; border.width: 1; border.color: pal.BORDER
                    TextInput {
                        anchors.fill: parent; anchors.leftMargin: sc.sp2
                        verticalAlignment: TextInput.AlignVCenter; horizontalAlignment: TextInput.AlignHCenter
                        text: Analysis.cfg.tost_margin; color: pal.TXT; font.pixelSize: 11; font.family: "Menlo"
                        onEditingFinished: Analysis.setCfg("tost_margin", text)
                    }
                }
            }
            // circular statistics (turning angles) — adds κ/R̄/μ between-group
            // tests + per-replicate CSVs / circular PDF to the full report.
            ColumnLayout {
                Layout.fillWidth: true; spacing: sc.sp2
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp2
                    Switch { checked: Analysis.cfg.include_circular_outputs === true
                             onToggled: (c) => Analysis.setCfg("include_circular_outputs", c) }
                    Text { text: "Circular statistics (turning angles)"; color: pal.TXT; font.pixelSize: 12 }
                }
                Flow {
                    Layout.fillWidth: true; Layout.leftMargin: sc.sp6; Layout.topMargin: sc.sp4
                    spacing: sc.sp3
                    visible: Analysis.cfg.include_circular_outputs === true
                    Repeater {
                        model: [{ k: "circ_test_kappa", t: "κ concentration" },
                                { k: "circ_test_rbar", t: "R̄ resultant" },
                                { k: "circ_test_mu", t: "μ direction" },
                                { k: "circ_test_circlin", t: "circular–linear" }]
                        delegate: Rectangle {
                            readonly property bool on_: Analysis.cfg[modelData.k] === true
                            implicitHeight: 26; implicitWidth: ccTxt.implicitWidth + sc.sp5 * 2
                            radius: height / 2                     // pill
                            color: on_ ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14) : pal.PANEL_ALT
                            border.width: 1; border.color: on_ ? pal.ACC : pal.BORDER
                            Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                            Text { id: ccTxt; anchors.centerIn: parent; text: modelData.t
                                   color: on_ ? pal.ACC : pal.TXT_MUTED; font.pixelSize: 11 }
                            HoverHandler { cursorShape: Qt.PointingHandCursor }
                            TapHandler { onTapped: Analysis.setCfg(modelData.k, !on_) }
                        }
                    }
                }
            }
            Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
            // error bars
            ColumnLayout {
                Layout.fillWidth: true; spacing: 4
                Text { text: "Error bars"; color: pal.TXT_MUTED; font.pixelSize: 12; font.bold: true }
                Segmented { Layout.fillWidth: true; options: ["SD", "SEM", "95% CI"]; value: Analysis.cfg.err; onPicked: (v) => Analysis.setCfg("err", v) }
            }
        }
    }

    // ════ toast ════
    Rectangle {
        id: toast
        anchors { bottom: parent.bottom; horizontalCenter: parent.horizontalCenter; bottomMargin: sc.sp5 }
        visible: opacity > 0; opacity: 0
        color: pal.PANEL_ALT; radius: sc.radiusMd; border.width: 1; border.color: pal.BORDER
        implicitWidth: tlabel.implicitWidth + sc.sp5 * 2; implicitHeight: 34
        Text { id: tlabel; anchors.centerIn: parent; color: pal.TXT; font.pixelSize: sc.textSm }
        Behavior on opacity { NumberAnimation { duration: 200 } }
        Timer { id: toastTimer; interval: 1600; onTriggered: toast.opacity = 0 }
        Connections { target: Analysis; function onToast(msg) { tlabel.text = msg; toast.opacity = 1; toastTimer.restart() } }
    }

    // ════ destructive-action confirmation ════
    ConfirmModal { id: confirmDelete }
}
