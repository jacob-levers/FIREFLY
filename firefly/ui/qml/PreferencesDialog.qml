import QtQuick
import QtQuick.Layouts
import "components"

// Application Preferences — a 940×680 floating window over a dimmed backdrop,
// with a left section rail (Appearance / Figures / Updates), a scrolling content
// pane, and a footer action bar. Theme + accent + reduce-motion repaint the app
// live via ThemeController; figure/update settings persist through
// SettingsController; the GitHub updater is UpdatesController. Opened from the
// header gear / ⌘,. Recreated from the design handoff (high-fidelity).
Item {
    id: root
    anchors.fill: parent
    // Stay mounted while the exit tween plays, then go invisible.
    visible: opened || backdrop.opacity > 0.001 || panel.opacity > 0.001
    property bool opened: false
    property string section: "appearance"
    property int rev: 0                         // bump → re-read Settings (Restore defaults + any live write)
    // Any settings write bumps rev so the live preview + field bindings re-read
    // immediately (this is what makes the Figures preview actually live).
    Connections { target: Settings; function onChanged(key) { root.rev++ } }
    // Entrance 220ms / exit 160ms (reduce-motion → 0). §7.2
    readonly property int durIn:  Theme.reducedMotion ? 0 : 220
    readonly property int durOut: Theme.reducedMotion ? 0 : 160

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    function open(sec) { if (sec !== undefined && sec !== "") section = sec; opened = true }
    function close() { opened = false }
    // Hide the always-on-top native viewer/ROI island so it can't occlude this
    // dialog (e.g. over the Visualise tab's viewer).
    onOpenedChanged: Embed.setModalOpen(opened)

    // ── data ──────────────────────────────────────────────────────────────
    readonly property var nav: [
        { id: "appearance", icon: "palette",        label: "Appearance", sub: "Theme & accent" },
        { id: "figures",    icon: "chart-spline",   label: "Figures",    sub: "Plots & export" },
        { id: "analysis",   icon: "scan-search",    label: "Analysis",   sub: "ROI & file matching" },
        { id: "glossary",   icon: "quote",          label: "Glossary",   sub: "Stats & analysis terms" },
        { id: "gpu",        icon: "zap",            label: "GPU",        sub: "CUDA acceleration" },
        { id: "updates",    icon: "download-cloud", label: "Updates",    sub: "Version & channel" }
    ]
    readonly property var sectionTitle: ({ appearance: "Appearance", figures: "Figures", analysis: "Analysis", glossary: "Glossary", gpu: "GPU acceleration", updates: "Updates" })
    readonly property var sectionSub: ({
        appearance: "Choose how FIREFLY looks — its theme and accent colour.",
        figures: "Control the graphs FIREFLY renders and how they are exported.",
        analysis: "How FIREFLY finds companion files and builds regions of interest.",
        glossary: "Plain-language definitions of the statistics and analysis terms.",
        gpu: "Install or update the CUDA backend so analysis runs on your NVIDIA GPU.",
        updates: "Manage releases from GitHub and the update channel." })
    readonly property var themeTiles: [
        { name: "Dark",   sub: "GitHub-dark · default", bg: "#0d1117", panel: "#161b22", line: "#30363d", text: "#e6edf3", muted: "#8b949e" },
        { name: "Light",  sub: "High-contrast daytime", bg: "#ffffff", panel: "#f6f8fa", line: "#d0d7de", text: "#24292f", muted: "#57606a" },
        { name: "AMOLED", sub: "Pure black · OLED",     bg: "#000000", panel: "#0a0a0a", line: "#1c1c1c", text: "#e6edf3", muted: "#8b949e" }
    ]
    readonly property var motionStd: [
        { n: "Immobile", c: "#e05252" }, { n: "Confined", c: "#f5a623" },
        { n: "Brownian", c: "#4a90d9" }, { n: "Directed", c: "#7ed321" }]
    readonly property var motionCb: [
        { n: "Immobile", c: "#d55e00" }, { n: "Confined", c: "#e69f00" },
        { n: "Brownian", c: "#0072b2" }, { n: "Directed", c: "#009e73" }]
    readonly property var cmapStops: ({
        "Inferno": ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
        "Hot":     ["#0b0000", "#e60000", "#ffcc00", "#ffffff"],
        "Viridis": ["#440154", "#3b528b", "#21908d", "#5dc863", "#fde725"],
        "Plasma":  ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"],
        "Greys":   ["#000000", "#ffffff"] })

    readonly property var cmapOpts: ["Inferno", "Hot", "Viridis", "Plasma", "Greys"]
    readonly property var figThemeOpts: ["Dark", "Light", "Publication"]
    readonly property var dpiOpts: ["150", "300", "440", "600"]
    readonly property var fontOpts: ["DejaVu Sans", "Helvetica", "Arial", "JetBrains Mono"]
    readonly property var formatOpts: ["PDF (vector)", "PNG", "SVG", "TIFF"]
    readonly property var densityOpts: ["Compact", "Comfortable"]
    readonly property var fontSizeOpts: ["Small — 11px", "Medium — 12px", "Large — 14px"]
    readonly property var channelOpts: ["Stable", "Pre-release"]
    readonly property var logdLabels: ["Faceted (per-replicate)", "Ridgeline", "Overlaid KDEs", "Violins + points"]
    readonly property var logdValues: ["faceted", "ridgeline", "overlaid", "violin"]
    readonly property var msdLabels: ["Mean ± error (faceted)", "Individual cells + mean", "Group overlaid"]
    readonly property var msdValues: ["mean_faceted", "individual", "overlaid"]
    readonly property var groupLabels: ["Box + points", "Grouped by timepoint", "Violin + points", "Bar"]
    readonly property var groupValues: ["box_points", "grouped", "violin", "bar"]
    // Per-graph mark for the scalar comparison panels (one control each, below).
    readonly property var markLabels: ["Box + points", "Violin + points", "Bar"]
    readonly property var markValues: ["box_points", "violin", "bar"]
    readonly property var lengthLabels: ["Density", "Box"]
    readonly property var lengthValues: ["density", "box"]
    readonly property var aucLabels: ["Box + points", "Violin + points", "Bar", "Paired lines (timepoints)", "Δ box (timepoints)"]
    readonly property var aucValues: ["box_points", "violin", "bar", "paired", "delta"]

    function restoreDefaults() {
        Theme.setTheme("Dark"); Theme.setAccent("Luminous blue"); Theme.reducedMotion = false
        Settings.setValue("ui/density", "Compact")
        Settings.setValue("ui/font_size", "Medium — 12px")
        Settings.setValue("ui/header_pill", true)
        Settings.setValue("figures/theme", "Dark")
        Settings.setValue("figures/proj_cmap", "Inferno")
        Settings.setValue("figures/dpi", 150)
        Settings.setValue("figures/line_width", "0.8")
        Settings.setValue("figures/font", "DejaVu Sans")
        Settings.setValue("figures/antialias", true)
        Settings.setValue("figures/transparent", false)
        Settings.setValue("figures/traj_bg", true)
        Settings.setValue("figures/format", "PDF (vector)")
        Settings.setValue("figures/save_pdf", true)
        Settings.setValue("figures/per_panel", false)
        Settings.setValue("figures/logd_style", "overlaid")
        Settings.setValue("updates/auto_check", true)
        Settings.setValue("updates/channel", "Stable")
        Settings.setValue("updates/auto_download", false)
        Settings.setValue("updates/notify_pre", false)
        root.rev++
        // Controls sever their value bindings once touched, so rebuild the
        // current section view to re-read every reset value.
        contentLoader.active = false
        contentLoader.active = true
    }

    // ── reusable primitives ────────────────────────────────────────────────
    component Group: Rectangle {
        id: grp
        property string title: ""
        property string desc: ""
        default property alias body: grpBody.data
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        Layout.fillWidth: true
        radius: sc.radiusXl
        color: pal.PANEL
        border.width: 1; border.color: pal.BORDER
        clip: true
        implicitHeight: grpCol.implicitHeight
        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
        ColumnLayout {
            id: grpCol
            width: parent.width
            spacing: 0
            ColumnLayout {                       // header
                Layout.fillWidth: true
                Layout.margins: sc.sp5
                Layout.bottomMargin: sc.sp3
                spacing: 2
                Text { text: grp.title; color: pal.TXT; font.pixelSize: sc.textMd
                       font.weight: Font.DemiBold }
                Text { visible: grp.desc !== ""; text: grp.desc; color: pal.TXT_MUTED
                       font.pixelSize: sc.textXs; Layout.fillWidth: true; wrapMode: Text.WordWrap }
            }
            ColumnLayout { id: grpBody; Layout.fillWidth: true; spacing: 0 }
        }
    }

    component PrefRow: Rectangle {
        id: pr
        property string label: ""
        property string desc: ""
        default property alias control: ctlHolder.data
        readonly property var pal: Theme.palette
        readonly property var sc: Theme.scale
        Layout.fillWidth: true
        color: "transparent"
        implicitHeight: Math.max(rowLbl.implicitHeight, 30) + sc.sp5 * 2
        Rectangle { anchors { left: parent.left; right: parent.right; top: parent.top }
                    height: 1; color: pal.BORDER }
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
            spacing: sc.sp8
            ColumnLayout {
                id: rowLbl
                Layout.fillWidth: true; Layout.preferredWidth: 0; spacing: 1
                Text { text: pr.label; color: pal.TXT; font.pixelSize: sc.textSm
                       Layout.fillWidth: true; elide: Text.ElideRight }
                Text { visible: pr.desc !== ""; text: pr.desc; color: pal.TXT_MUTED
                       font.pixelSize: sc.textXs; Layout.fillWidth: true; wrapMode: Text.WordWrap }
            }
            Item { id: ctlHolder; Layout.preferredWidth: childrenRect.width
                   Layout.preferredHeight: childrenRect.height
                   Layout.alignment: Qt.AlignVCenter }
        }
    }

    // A per-graph "how is this scalar comparison drawn" row: box+points / violin /
    // bar, bound to its OWN setting (figures/style_<panelKey>).
    component MarkRow: PrefRow {
        id: mrow
        property string panelKey: ""
        Select { implicitWidth: 170; model: root.markLabels
                 currentIndex: (root.rev, Math.max(0, root.markValues.indexOf(
                     Settings.getStr("figures/style_" + mrow.panelKey, "box_points"))))
                 onPicked: (t) => { var i = root.markLabels.indexOf(t)
                                    if (i >= 0) Settings.setValue("figures/style_" + mrow.panelKey, root.markValues[i]) } }
    }

    // ── dimmed backdrop ─────────────────────────────────────────────────────
    Rectangle {
        id: backdrop
        anchors.fill: parent; color: "#000000"
        opacity: root.opened ? 0.55 : 0
        Behavior on opacity { NumberAnimation { duration: root.opened ? root.durIn : root.durOut } }
        MouseArea {                              // block + dismiss; nothing leaks behind
            anchors.fill: parent
            hoverEnabled: true
            acceptedButtons: Qt.AllButtons
            onClicked: root.close()
            onWheel: (wheel) => { wheel.accepted = true }   // don't scroll the UI behind
        }
    }

    // ── window panel ────────────────────────────────────────────────────────
    Rectangle {
        id: panel
        anchors.centerIn: parent
        width: Math.min(940, parent.width - 48)
        height: Math.min(680, parent.height - 48)
        radius: sc.radius2xl
        color: pal.BG
        border.width: 1; border.color: pal.BORDER
        clip: true
        // Entrance: fade + scale 0.96→1 + rise 8px (OutCubic); reverse on close.
        opacity: root.opened ? 1 : 0
        scale:   root.opened ? 1 : 0.96
        Behavior on opacity { NumberAnimation { duration: root.opened ? root.durIn : root.durOut; easing.type: Easing.OutCubic } }
        Behavior on scale   { NumberAnimation { duration: root.opened ? root.durIn : root.durOut; easing.type: Easing.OutCubic } }
        transform: Translate {
            y: root.opened ? 0 : 8
            Behavior on y { NumberAnimation { duration: root.opened ? root.durIn : root.durOut; easing.type: Easing.OutCubic } }
        }
        MouseArea {                              // swallow clicks/hover/wheel over the panel
            anchors.fill: parent
            hoverEnabled: true
            acceptedButtons: Qt.AllButtons
            onWheel: (wheel) => { wheel.accepted = true }   // catch wheel the Flickable doesn't use
        }

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            // ── title bar ────────────────────────────────────────────────
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 46
                color: pal.PANEL
                // round the top corners to match the panel; square where it
                // meets the body below
                radius: sc.radius2xl; bottomLeftRadius: 0; bottomRightRadius: 0
                Rectangle { anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
                            height: 1; color: pal.BORDER }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp4
                    spacing: sc.sp3
                    Rectangle {
                        width: 24; height: 24; radius: 7
                        color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.16)
                        Icon { anchors.centerIn: parent; name: "settings"; size: 13; color: pal.ACC }
                    }
                    Text { text: "Preferences"; color: pal.TXT; font.pixelSize: sc.textLg
                           font.weight: Font.ExtraBold }
                    Text { text: "FIREFLY · v" + appVersion; color: pal.TXT_MUTED
                           font.pixelSize: 11; font.family: "Menlo"; Layout.leftMargin: sc.sp1 }
                    Item { Layout.fillWidth: true }
                    Rectangle {                  // close ✕ (red on hover)
                        width: 26; height: 26; radius: sc.radiusSm
                        color: "transparent"
                        border.width: 1
                        border.color: closeHov.hovered ? pal.DANGER : "transparent"
                        Icon { anchors.centerIn: parent; name: "x"; size: 14
                               color: closeHov.hovered ? pal.DANGER : pal.TXT_MUTED }
                        HoverHandler { id: closeHov; cursorShape: Qt.PointingHandCursor }
                        TapHandler { onTapped: root.close() }
                    }
                }
            }

            // ── body: rail + content ─────────────────────────────────────
            RowLayout {
                Layout.fillWidth: true; Layout.fillHeight: true
                spacing: 0

                // nav rail
                Rectangle {
                    Layout.preferredWidth: 198; Layout.fillHeight: true
                    color: pal.PANEL
                    Rectangle { anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                                width: 1; color: pal.BORDER }
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: sc.sp5
                        spacing: sc.sp1
                        Text { text: "SETTINGS"; color: pal.TXT_MUTED; font.pixelSize: 10
                               font.bold: true; font.letterSpacing: 1.0; Layout.bottomMargin: sc.sp2 }
                        Repeater {
                            model: root.nav
                            delegate: Rectangle {
                                required property var modelData
                                readonly property bool active: root.section === modelData.id
                                Layout.fillWidth: true
                                Layout.preferredHeight: 44     // fixed → uniform row pitch
                                radius: sc.radiusLg
                                color: active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                                              : (navHov.hovered ? pal.PANEL_ALT : "transparent")
                                border.width: 1
                                border.color: active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.40)
                                                     : "transparent"
                                Rectangle {       // accent left-bar (active)
                                    visible: active
                                    anchors { left: parent.left; top: parent.top; bottom: parent.bottom
                                              topMargin: sc.sp3; bottomMargin: sc.sp3 }
                                    width: 3; radius: 1.5; color: pal.ACC
                                }
                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: sc.sp4; anchors.rightMargin: sc.sp3
                                    spacing: sc.sp3
                                    Rectangle {
                                        Layout.preferredWidth: 26; Layout.preferredHeight: 26
                                        Layout.alignment: Qt.AlignVCenter
                                        radius: 7
                                        color: active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.16)
                                                      : pal.PANEL_ALT
                                        Icon { anchors.centerIn: parent; name: modelData.icon; size: 14
                                               color: active ? pal.ACC : pal.TXT_MUTED }
                                    }
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        Layout.alignment: Qt.AlignVCenter
                                        spacing: 0
                                        // fillWidth + explicit left-align so the text
                                        // never falls back to a centred default (which
                                        // shifted some rows' labels right of others)
                                        Text { text: modelData.label
                                               Layout.fillWidth: true
                                               horizontalAlignment: Text.AlignLeft
                                               elide: Text.ElideRight
                                               color: active ? pal.TXT : pal.TXT_MUTED
                                               font.pixelSize: sc.textMd; font.weight: Font.DemiBold }
                                        Text { text: modelData.sub
                                               Layout.fillWidth: true
                                               horizontalAlignment: Text.AlignLeft
                                               elide: Text.ElideRight
                                               color: pal.TXT_MUTED
                                               font.pixelSize: 11 }
                                    }
                                }
                                HoverHandler { id: navHov; cursorShape: Qt.PointingHandCursor }
                                TapHandler { onTapped: root.section = modelData.id }
                            }
                        }
                        Item { Layout.fillHeight: true }
                        RowLayout {               // rail footer: config path
                            Layout.fillWidth: true; spacing: sc.sp2
                            Icon { name: "folder-cog"; size: 13; color: pal.TXT_MUTED }
                            Text { text: "~/.firefly/config.toml"; color: pal.TXT_MUTED
                                   font.pixelSize: 11; font.family: "Menlo"
                                   Layout.fillWidth: true; elide: Text.ElideMiddle }
                        }
                    }
                }

                // content pane (scroll)
                Flickable {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    contentWidth: width
                    contentHeight: contentCol.implicitHeight + sc.sp10
                    clip: true
                    boundsBehavior: Flickable.StopAtBounds
                    ColumnLayout {
                        id: contentCol
                        x: sc.sp10; y: sc.sp8
                        width: Math.min(760, parent.width - sc.sp10 * 2)
                        spacing: sc.sp8
                        ColumnLayout {            // section header
                            Layout.fillWidth: true; spacing: 2
                            Text { text: root.sectionTitle[root.section]; color: pal.TXT
                                   font.pixelSize: 19; font.weight: Font.ExtraBold }
                            Text { text: root.sectionSub[root.section]; color: pal.TXT_MUTED
                                   font.pixelSize: sc.textMd }
                        }
                        Loader {
                            id: contentLoader
                            Layout.fillWidth: true
                            sourceComponent: root.section === "appearance" ? appearanceView
                                           : root.section === "figures" ? figuresView
                                           : root.section === "analysis" ? analysisView
                                           : root.section === "glossary" ? glossaryView
                                           : root.section === "gpu" ? gpuView
                                           : updatesView
                        }
                    }
                }
            }

            // ── footer ───────────────────────────────────────────────────
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 56
                color: pal.PANEL
                // round the bottom corners to match the panel; square at the top
                // where it meets the body above
                radius: sc.radius2xl; topLeftRadius: 0; topRightRadius: 0
                Rectangle { anchors { left: parent.left; right: parent.right; top: parent.top }
                            height: 1; color: pal.BORDER }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                    spacing: sc.sp4
                    Button { variant: "ghost"; text: "Restore defaults"; icon: "rotate-ccw"
                             onClicked: root.restoreDefaults() }
                    Item { Layout.fillWidth: true }
                    Button { variant: "secondary"; text: "Cancel"; onClicked: root.close() }
                    Button { variant: "primary"; text: "Done"; onClicked: root.close() }
                }
            }
        }
    }

    Shortcut { sequence: "Escape"; enabled: root.opened; onActivated: root.close() }

    // ════════════════════════ APPEARANCE ════════════════════════════════════
    Component {
        id: appearanceView
        ColumnLayout {
            spacing: sc.sp8
            Group {
                title: "Theme"
                desc: "Re-skins the entire application. Changes apply live — the window repaints as you pick."
                RowLayout {
                    Layout.fillWidth: true; Layout.margins: sc.sp5; Layout.topMargin: 0
                    spacing: sc.sp5
                    Repeater {
                        model: root.themeTiles
                        delegate: Rectangle {
                            required property var modelData
                            readonly property bool sel: Theme.name === modelData.name
                            Layout.fillWidth: true
                            implicitHeight: 126
                            radius: 9
                            color: pal.PANEL_ALT
                            border.width: sel ? 2 : 1
                            border.color: sel ? pal.ACC : pal.BORDER
                            ColumnLayout {
                                anchors.fill: parent; anchors.margins: sc.sp3; spacing: sc.sp3
                                Rectangle {       // mini app-window preview in the tile's palette
                                    Layout.fillWidth: true; Layout.preferredHeight: 70
                                    radius: sc.radiusSm; color: modelData.bg
                                    border.width: 1; border.color: modelData.line
                                    clip: true
                                    RowLayout {
                                        anchors.fill: parent; anchors.margins: 4; spacing: 4
                                        Rectangle {        // mini sidebar
                                            Layout.preferredWidth: 22; Layout.fillHeight: true
                                            radius: 2; color: modelData.panel
                                            ColumnLayout {
                                                anchors.fill: parent; anchors.margins: 3; spacing: 3
                                                Rectangle { Layout.preferredWidth: 10; height: 3; radius: 1.5; color: pal.ACC }
                                                Rectangle { Layout.fillWidth: true; height: 2; radius: 1; color: modelData.line }
                                                Rectangle { Layout.fillWidth: true; height: 2; radius: 1; color: modelData.line }
                                                Item { Layout.fillHeight: true }
                                            }
                                        }
                                        ColumnLayout {     // mini content
                                            Layout.fillWidth: true; Layout.fillHeight: true; spacing: 3
                                            Rectangle { Layout.preferredWidth: 24; height: 3; radius: 1.5; color: modelData.text }
                                            Rectangle { Layout.fillWidth: true; height: 2; radius: 1; color: modelData.muted }
                                            Rectangle { Layout.preferredWidth: 30; height: 2; radius: 1; color: modelData.muted }
                                            Item { Layout.fillHeight: true }
                                            Rectangle { Layout.preferredWidth: 18; height: 7; radius: 2; color: pal.ACC }
                                        }
                                    }
                                }
                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp2
                                    Rectangle { width: 11; height: 11; radius: 5.5
                                                color: "transparent"; border.width: 1.5
                                                border.color: sel ? pal.ACC : pal.BORDER_HI
                                                Rectangle { anchors.centerIn: parent; visible: sel
                                                            width: 5; height: 5; radius: 2.5; color: pal.ACC } }
                                    ColumnLayout {
                                        Layout.fillWidth: true; spacing: 0
                                        Text { text: modelData.name; color: pal.TXT; font.pixelSize: sc.textSm
                                               font.weight: Font.DemiBold }
                                        Text { text: modelData.sub; color: pal.TXT_MUTED; font.pixelSize: 10
                                               elide: Text.ElideRight; Layout.fillWidth: true }
                                    }
                                }
                            }
                            TapHandler { onTapped: Theme.setTheme(modelData.name) }
                            HoverHandler { cursorShape: Qt.PointingHandCursor }
                        }
                    }
                }
            }
            Group {
                title: "Accent colour"
                desc: "The single highlight reserved for the primary action, focus rings and the active state."
                PrefRow {
                    label: "Accent"
                    desc: "Most of FIREFLY stays grey — colour means data or action."
                    RowLayout {
                        spacing: sc.sp3
                        Repeater {
                            model: Theme.accents
                            delegate: Rectangle {
                                required property var modelData
                                readonly property bool sel: Theme.accentName === modelData.name
                                width: 26; height: 26; radius: 7
                                color: modelData.v
                                border.width: sel ? 2 : 0
                                border.color: pal.TXT
                                Rectangle { anchors.fill: parent; anchors.margins: -3; radius: 9
                                            visible: sel; color: "transparent"
                                            border.width: 1; border.color: pal.PANEL }
                                TapHandler { onTapped: Theme.setAccent(modelData.name) }
                                HoverHandler { cursorShape: Qt.PointingHandCursor }
                            }
                        }
                    }
                }
                PrefRow {
                    label: "Interface density"
                    desc: "Compact mirrors the shipping desktop build; comfortable adds breathing room."
                    Select {
                        implicitWidth: 170
                        model: root.densityOpts
                        currentIndex: (root.rev, Math.max(0, root.densityOpts.indexOf(Settings.getStr("ui/density", "Compact"))))
                        onPicked: (t) => { Settings.setValue("ui/density", t); Theme.setDensity(t) }
                    }
                }
                PrefRow {
                    label: "UI font size"
                    Select {
                        implicitWidth: 170
                        model: root.fontSizeOpts
                        currentIndex: (root.rev, Math.max(0, root.fontSizeOpts.indexOf(Settings.getStr("ui/font_size", "Medium — 12px"))))
                        onPicked: (t) => { Settings.setValue("ui/font_size", t); Theme.setFontSize(t) }
                    }
                }
            }
            Group {
                title: "Motion & chrome"
                PrefRow {
                    label: "Reduce motion"
                    desc: "Disables live-meter easing and panel fade transitions."
                    Switch { checked: Theme.reducedMotion; onToggled: (c) => Theme.reducedMotion = c }
                }
                PrefRow {
                    label: "Show “Update available” pill in header"
                    desc: "The amber pill beside the wordmark when a release is waiting."
                    Switch {
                        checked: (root.rev, Settings.getBool("ui/header_pill", true))
                        onToggled: (c) => Settings.setValue("ui/header_pill", c)
                    }
                }
            }
        }
    }

    // ════════════════════════ FIGURES ═══════════════════════════════════════
    Component {
        id: figuresView
        RowLayout {
            spacing: sc.sp6
            ColumnLayout {                       // settings column
                Layout.fillWidth: true; Layout.preferredWidth: 0
                Layout.alignment: Qt.AlignTop; spacing: sc.sp8
                Group {
                    title: "Figure theme & palette"
                    desc: "How plots are coloured. The motion-class palette is the canonical pipeline output — switch to colour-blind-safe for publication."
                    PrefRow {
                        label: "Figure theme"; desc: "Background and axis styling for rendered plots."
                        Select { implicitWidth: 170; model: root.figThemeOpts
                                 currentIndex: (root.rev, Math.max(0, root.figThemeOpts.indexOf(Settings.getStr("figures/theme", "Dark"))))
                                 onPicked: (t) => Settings.setValue("figures/theme", t) }
                    }
                    PrefRow {
                        label: "Projection colormap"; desc: "Applied to max-projection and intensity panels."
                        Select { implicitWidth: 170; model: root.cmapOpts
                                 currentIndex: (root.rev, Math.max(0, root.cmapOpts.indexOf(Settings.getStr("figures/proj_cmap", "Inferno"))))
                                 onPicked: (t) => Settings.setValue("figures/proj_cmap", t) }
                    }
                    PrefRow {
                        label: "Motion-class palette"
                        desc: "Colour-blind safe uses the Okabe–Ito set in the Visualise viewer + legend. (Exported figures follow the figure theme — pick Publication there for colour-blind.)"
                        Select { implicitWidth: 170
                                 model: ["Default", "Colour-blind safe"]
                                 currentIndex: (root.rev, Math.max(0, ["Default", "Colour-blind safe"].indexOf(Settings.getStr("visualise/motion_colours", "Default"))))
                                 onPicked: (t) => Settings.setValue("visualise/motion_colours", t) }
                    }
                }
                Group {
                    title: "Graph styles"
                    desc: "How individual graphs are drawn, in both the live Analysis tab and the exported report. More graphs will become customisable here over time."
                    PrefRow {
                        label: "Log-D distribution"
                        desc: "How the per-condition log₁₀(D) distributions are plotted."
                        Select { implicitWidth: 170; model: root.logdLabels
                                 currentIndex: (root.rev, Math.max(0, root.logdValues.indexOf(Settings.getStr("figures/logd_style", "overlaid"))))
                                 onPicked: (t) => { var i = root.logdLabels.indexOf(t)
                                                    if (i >= 0) Settings.setValue("figures/logd_style", root.logdValues[i]) } }
                    }
                    PrefRow {
                        label: "MSD curves"
                        desc: "How the ensemble-MSD vs. time-lag curves are plotted, faceted by group (timepoints from your setup; error type from the Analysis tab)."
                        Select { implicitWidth: 170; model: root.msdLabels
                                 currentIndex: (root.rev, Math.max(0, root.msdValues.indexOf(Settings.getStr("figures/msd_style", "mean_faceted"))))
                                 onPicked: (t) => { var i = root.msdLabels.indexOf(t)
                                                    if (i >= 0) Settings.setValue("figures/msd_style", root.msdValues[i]) } }
                    }
                    MarkRow { label: "Fluorescence"; panelKey: "fluor"
                              desc: "How the fluorescence (spot-intensity) comparison is drawn." }
                    MarkRow { label: "Radius of gyration"; panelKey: "rg"
                              desc: "How the radius-of-gyration comparison is drawn." }
                    MarkRow { label: "Net displacement"; panelKey: "netdisp"
                              desc: "How the net-displacement (first→last) comparison is drawn." }
                    MarkRow { label: "Path length"; panelKey: "path"
                              desc: "How the path-length comparison is drawn." }
                    MarkRow { label: "Step distance"; panelKey: "step"
                              desc: "How the (measured) step-distance comparison is drawn." }
                    MarkRow { label: "Step speed"; panelKey: "speed"
                              desc: "How the (measured) step-speed comparison is drawn." }
                    MarkRow { label: "Observed-link distance"; panelKey: "linkstep"
                              desc: "How displacement between every adjacent observed localisation (including gap-spanning links) is drawn." }
                    MarkRow { label: "Observed-link speed"; panelKey: "linkspeed"
                              desc: "How each observed-link displacement divided by its actual elapsed frame time is drawn." }
                    MarkRow { label: "Directionality ratio"; panelKey: "dir"
                              desc: "How the directionality-ratio (net÷path) comparison is drawn." }
                    MarkRow { label: "Track duration"; panelKey: "dur"
                              desc: "How the track-duration comparison is drawn." }
                    MarkRow { label: "Localisations"; panelKey: "nlocs"
                              desc: "How the localisation-count comparison is drawn." }
                    MarkRow { label: "Mobile fraction"; panelKey: "mob_immob"
                              desc: "How the mobile/immobile-ratio comparison is drawn." }
                    MarkRow { label: "Track count"; panelKey: "track_count"
                              desc: "How the tracks-per-dish comparison is drawn." }
                    MarkRow { label: "Non-Gaussian α₂"; panelKey: "van_hove"
                              desc: "How the van-Hove non-Gaussian parameter comparison is drawn." }
                    MarkRow { label: "Persistence (VACF)"; panelKey: "vacf"
                              desc: "How the VACF directional-persistence comparison is drawn." }
                    PrefRow {
                        label: "Track-length distribution"
                        desc: "Overlaid density (with the filter-threshold line) or a per-group box."
                        Select { implicitWidth: 170; model: root.lengthLabels
                                 currentIndex: (root.rev, Math.max(0, root.lengthValues.indexOf(Settings.getStr("figures/length_style", "density"))))
                                 onPicked: (t) => { var i = root.lengthLabels.indexOf(t)
                                                    if (i >= 0) Settings.setValue("figures/length_style", root.lengthValues[i]) } }
                    }
                    PrefRow {
                        label: "MSD-AUC"
                        desc: "Area under each MSD curve. Box + points / violin / bar draw it per condition; 'Paired lines' and 'Δ box' show the change across timepoints and apply only to group × pre/post designs (otherwise they fall back to box + points)."
                        Select { implicitWidth: 170; model: root.aucLabels
                                 currentIndex: (root.rev, Math.max(0, root.aucValues.indexOf(Settings.getStr("figures/auc_style", "box_points"))))
                                 onPicked: (t) => { var i = root.aucLabels.indexOf(t)
                                                    if (i >= 0) Settings.setValue("figures/auc_style", root.aucValues[i]) } }
                    }
                }
                Group {
                    title: "Rendering & quality"
                    desc: "Resolution and line styling for everything FIREFLY draws."
                    PrefRow {
                        label: "Export resolution"
                        Select { implicitWidth: 170; model: root.dpiOpts
                                 currentIndex: (root.rev, Math.max(0, root.dpiOpts.indexOf("" + Settings.get("figures/dpi", 150))))
                                 onPicked: (t) => Settings.setValue("figures/dpi", parseInt(t)) }
                    }
                    PrefRow {
                        label: "Trajectory line width"; desc: "Stroke weight for track overlays, in points."
                        FieldInput {
                            implicitWidth: 170; horizontalAlignment: TextInput.AlignRight
                            font.family: "Menlo"
                            text: (root.rev, Settings.getStr("figures/line_width", "0.8"))
                            onEditingFinished: Settings.setValue("figures/line_width", text)
                        }
                    }
                    PrefRow {
                        label: "Trajectory background image"
                        desc: "Show the raw microscope image behind the trajectory panels (Trajectories / Trajectories by D). Off draws the tracks on a plain background. Applies to figures generated from now on — re-run the analysis to update existing ones."
                        Switch { checked: (root.rev, Settings.getBool("figures/traj_bg", true))
                                 onToggled: (c) => Settings.setValue("figures/traj_bg", c) }
                    }
                    PrefRow {
                        label: "Plot font"
                        Select { implicitWidth: 170; model: root.fontOpts
                                 currentIndex: (root.rev, Math.max(0, root.fontOpts.indexOf(Settings.getStr("figures/font", "DejaVu Sans"))))
                                 onPicked: (t) => Settings.setValue("figures/font", t) }
                    }
                    PrefRow {
                        label: "Anti-aliasing"; desc: "Smooth strokes; disable for crisp pixel-exact masks."
                        Switch { checked: (root.rev, Settings.getBool("figures/antialias", true))
                                 onToggled: (c) => Settings.setValue("figures/antialias", c) }
                    }
                    PrefRow {
                        label: "Transparent background"; desc: "Render panels with an alpha channel for overlays."
                        Switch { checked: (root.rev, Settings.getBool("figures/transparent", false))
                                 onToggled: (c) => Settings.setValue("figures/transparent", c) }
                    }
                }
                Group {
                    title: "Export"
                    desc: "What FIREFLY writes alongside each run."
                    PrefRow {
                        label: "Default format"
                        Select { implicitWidth: 170; model: root.formatOpts
                                 currentIndex: (root.rev, Math.max(0, root.formatOpts.indexOf(Settings.getStr("figures/format", "PDF (vector)"))))
                                 onPicked: (t) => Settings.setValue("figures/format", t) }
                    }
                    PrefRow {
                        label: "Vector PDF export"; desc: "Fully editable, infinite-resolution figure."
                        Checkbox { checked: (root.rev, Settings.getBool("figures/save_pdf", true))
                                 onToggled: (c) => Settings.setValue("figures/save_pdf", c) }
                    }
                    PrefRow {
                        label: "Per-panel PNGs"; desc: "Also write each of the 17 panels as a separate raster."
                        Checkbox { checked: (root.rev, Settings.getBool("figures/per_panel", false))
                                 onToggled: (c) => Settings.setValue("figures/per_panel", c) }
                    }
                }
                Group {
                    title: "Comparison report"
                    desc: "Defaults for the Analysis tab’s ‘Generate full report’."
                    PrefRow {
                        label: "Mobile-D threshold"; desc: "Diffusion cutoff (µm²/s) splitting mobile vs immobile."
                        FieldInput {
                            implicitWidth: 170; horizontalAlignment: TextInput.AlignRight
                            text: (root.rev, Settings.getStr("analysis/mobile_d", "0.02"))
                            onEditingFinished: Settings.setValue("analysis/mobile_d", text)
                        }
                    }
                    // full-width (a PrefRow's eliding label-column squishes this)
                    Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.margins: sc.sp5; Layout.topMargin: sc.sp4
                        spacing: sc.sp3
                        ColumnLayout {
                            Layout.fillWidth: true; spacing: 1
                            Text { text: "Figure panels"; color: pal.TXT; font.pixelSize: sc.textSm }
                            Text { text: "Which panels the multi-panel comparison figure includes."
                                   color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                   Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        }
                        RowLayout {
                            Layout.fillWidth: true; spacing: sc.sp3
                            Repeater {
                                model: Analysis.comparePanelPresets
                                delegate: Rectangle {
                                    implicitHeight: 26; implicitWidth: pTxt.implicitWidth + sc.sp5 * 2
                                    radius: height / 2                 // pill
                                    // named presets read as filled buttons; All/None
                                    // are lighter utility actions; active = accent.
                                    color: modelData.active ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.18)
                                           : pHov.hovered ? pal.PANEL_ALT
                                           : modelData.util ? "transparent" : pal.PANEL_ALT
                                    border.width: 1
                                    border.color: modelData.active ? pal.ACC : pal.BORDER
                                    Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                                    Text { id: pTxt; anchors.centerIn: parent; text: modelData.name
                                           color: modelData.active ? pal.ACC : pal.TXT_MUTED
                                           font.pixelSize: 11; font.bold: modelData.active }
                                    HoverHandler { id: pHov; cursorShape: Qt.PointingHandCursor }
                                    TapHandler { onTapped: Analysis.setPanelPreset(modelData.name) }
                                }
                            }
                            Item { Layout.fillWidth: true }
                        }
                        Flow {
                            Layout.fillWidth: true; spacing: sc.sp4
                            Layout.topMargin: sc.sp5            // separate presets from the individual figures
                            Repeater {
                                model: Analysis.comparePanels
                                delegate: Rectangle {
                                    implicitHeight: 24; implicitWidth: cTxt.implicitWidth + sc.sp4
                                    radius: 6
                                    color: modelData.on ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14) : pal.PANEL_ALT
                                    border.width: 1; border.color: modelData.on ? pal.ACC : pal.BORDER
                                    Text { id: cTxt; anchors.centerIn: parent; text: modelData.label
                                           color: modelData.on ? pal.ACC : pal.TXT_MUTED; font.pixelSize: 10 }
                                    TapHandler { onTapped: Analysis.togglePanel(modelData.key, !modelData.on) }
                                }
                            }
                        }
                    }
                }
            }

            // sticky live-preview column
            ColumnLayout {
                id: previewCol
                Layout.preferredWidth: 248; Layout.maximumWidth: 248
                Layout.alignment: Qt.AlignTop; spacing: sc.sp4
                readonly property string figTheme: (root.rev, Settings.getStr("figures/theme", "Dark"))
                readonly property string cmap: (root.rev, Settings.getStr("figures/proj_cmap", "Inferno"))
                readonly property var motion: (figTheme === "Publication"
                                               || (root.rev, Settings.getStr("visualise/motion_colours", "Default")) === "Colour-blind safe")
                                              ? root.motionCb : root.motionStd
                Rectangle {
                    Layout.fillWidth: true
                    radius: sc.radiusXl; color: pal.PANEL
                    border.width: 1; border.color: pal.BORDER
                    implicitHeight: prevCol.implicitHeight + sc.sp5 * 2
                    ColumnLayout {
                        id: prevCol
                        x: sc.sp4; y: sc.sp5; width: parent.width - sc.sp4 * 2
                        spacing: sc.sp3
                        Text { text: "LIVE PREVIEW"; color: pal.TXT_MUTED; font.pixelSize: 10
                               font.bold: true; font.letterSpacing: 1.0 }
                        Rectangle {                 // framed sample figure (themed)
                            Layout.fillWidth: true; Layout.preferredHeight: 150
                            radius: sc.radiusSm; clip: true
                            // match the panel's own figure background so the
                            // letterbox border blends with the render
                            color: previewCol.figTheme === "Dark" ? "#0d1117" : "#ffffff"
                            border.width: 1; border.color: pal.BORDER
                            Image {
                                anchors.fill: parent; anchors.margins: 4
                                fillMode: Image.PreserveAspectFit; smooth: true
                                sourceSize: Qt.size(300, 300)
                                // a real render per theme AND projection colormap —
                                // Light/Publication get a white-background figure,
                                // Dark a dark one; the cmap recolours the projection
                                source: "assets/figures/panel_A_"
                                        + (previewCol.figTheme === "Light" ? "light"
                                           : previewCol.figTheme === "Publication" ? "publication"
                                           : "dark")
                                        + "_" + (root.cmapStops[previewCol.cmap]
                                                 ? previewCol.cmap.toLowerCase() : "inferno")
                                        + ".png"
                            }
                        }
                        Canvas {                    // colormap gradient strip
                            Layout.fillWidth: true; Layout.preferredHeight: 12
                            property var stops: root.cmapStops[previewCol.cmap] || ["#000000", "#ffffff"]
                            onStopsChanged: requestPaint()
                            onWidthChanged: requestPaint()
                            onPaint: {
                                var ctx = getContext("2d"); ctx.reset()
                                var g = ctx.createLinearGradient(0, 0, width, 0)
                                for (var i = 0; i < stops.length; ++i)
                                    g.addColorStop(stops.length > 1 ? i / (stops.length - 1) : 0, stops[i])
                                ctx.fillStyle = g; ctx.fillRect(0, 0, width, height)
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Text { text: previewCol.cmap; color: pal.TXT_MUTED
                                   font.pixelSize: sc.textXs; font.family: "Menlo" }
                            Item { Layout.fillWidth: true }
                            Text { text: (root.rev, Settings.get("figures/dpi", 150)) + " DPI"
                                   color: pal.TXT_MUTED
                                   font.pixelSize: sc.textXs; font.family: "Menlo" }
                        }
                        Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
                        GridLayout {                // motion-class legend
                            Layout.fillWidth: true; columns: 2; columnSpacing: sc.sp4; rowSpacing: sc.sp2
                            Repeater {
                                model: previewCol.motion
                                delegate: RowLayout {
                                    required property var modelData
                                    spacing: sc.sp2
                                    Rectangle { width: 10; height: 10; radius: 2; color: modelData.c }
                                    Text { text: modelData.n; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                }
                            }
                        }
                        Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }
                        Text {
                            Layout.fillWidth: true
                            text: {
                                var fmt = (root.rev, Settings.getStr("figures/format", "PDF (vector)"))
                                var ext = fmt.indexOf("PDF") === 0 ? "pdf" : fmt.toLowerCase()
                                var vec = (fmt.indexOf("PDF") === 0 || fmt === "SVG") ? "vector" : "raster"
                                return "figure." + ext + " · " + previewCol.figTheme + " · " + vec
                            }
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo"
                            elide: Text.ElideRight
                        }
                    }
                }
                Text {
                    Layout.fillWidth: true; wrapMode: Text.WordWrap
                    text: "Settings drive every exported panel of the 17-panel publication figure and the in-app plots alike."
                    color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                }
            }
        }
    }

    // ════════════════════════ ANALYSIS ══════════════════════════════════════
    Component {
        id: analysisView
        ColumnLayout {
            spacing: sc.sp8
            Group {
                title: "Companion ROI image"
                desc: "FIREFLY can read a second image saved beside a recording — a "
                    + "widefield or marker channel — and use it to build the region of "
                    + "interest, or just to look at. It is matched by the text that "
                    + "follows the recording's own name."
                PrefRow {
                    label: "Filename suffix"
                    desc: "For a recording named 'Cell1', a suffix of '_green' matches "
                        + "'Cell1_green'. A microscope export might instead use "
                        + "'-Green Image'. .tif, .tiff and .czi are all accepted, and "
                        + "capitalisation is ignored. Leave blank to switch companion "
                        + "matching off."
                    FieldInput {
                        id: sisterSuffixField
                        implicitWidth: 200
                        placeholderText: "_green"
                        // (root.rev, …) re-reads after a Reset, matching the other rows.
                        text: (root.rev, Settings.getStr("analysis/roi_sister_suffix", "_green"))
                        onEditingFinished:
                            Settings.setValue("analysis/roi_sister_suffix", text.trim())
                    }
                }
                PrefRow {
                    label: ""
                    desc: ""
                    Button {
                        variant: "secondary"
                        text: "Reset to _green"
                        onClicked: {
                            Settings.setValue("analysis/roi_sister_suffix", "_green")
                            sisterSuffixField.text = "_green"
                        }
                    }
                }
            }
        }
    }

    // ════════════════════════ GLOSSARY ══════════════════════════════════════
    Component {
        id: glossaryView
        ColumnLayout {
            spacing: sc.sp8
            Group {
                title: "Statistics"
                desc: "Plain-language definitions of the comparison-statistics terms."
                Repeater {
                    model: Analysis.statsGlossary
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: sc.sp5; Layout.rightMargin: sc.sp5
                        Layout.topMargin: sc.sp4; spacing: 1
                        Text { text: modelData.term; color: pal.TXT; font.pixelSize: sc.textSm
                               font.bold: true; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Text { text: modelData.definition; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                               Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    }
                }
                Item { Layout.fillWidth: true; Layout.preferredHeight: sc.sp8 }
            }
            Group {
                title: "Analysis parameters"
                desc: "Terms used across the Import / Process pipeline."
                Repeater {
                    model: Analysis.analysisGlossary
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        Layout.leftMargin: sc.sp5; Layout.rightMargin: sc.sp5
                        Layout.topMargin: sc.sp4; spacing: 1
                        Text { text: modelData.term; color: pal.TXT; font.pixelSize: sc.textSm
                               font.bold: true; Layout.fillWidth: true; wrapMode: Text.WordWrap }
                        Text { text: modelData.definition; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                               Layout.fillWidth: true; wrapMode: Text.WordWrap }
                    }
                }
                Item { Layout.fillWidth: true; Layout.preferredHeight: sc.sp8 }
            }
        }
    }

    // ════════════════════════ GPU ACCELERATION (CUDA) ═══════════════════════
    Component {
        id: gpuView
        ColumnLayout {
            spacing: sc.sp5
            Component.onCompleted: Cuda.refresh()    // probe GPU + sidecar (off-thread)

            Group {
                title: "GPU backend"
                RowLayout {
                    Layout.fillWidth: true; spacing: sc.sp4
                    Layout.leftMargin: sc.sp5; Layout.rightMargin: sc.sp5
                    Layout.bottomMargin: sc.sp5
                    Rectangle {
                        Layout.preferredWidth: 44; Layout.preferredHeight: 44; radius: sc.radius2xl
                        color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                        Icon { anchors.centerIn: parent; name: "zap"; size: 22; color: pal.ACC }
                    }
                    ColumnLayout {
                        Layout.fillWidth: true; Layout.preferredWidth: 0; spacing: 2
                        Text {
                            Layout.fillWidth: true; elide: Text.ElideRight
                            text: !Cuda.checked ? "Checking for a GPU…"
                                : Cuda.gpuName !== "" ? Cuda.gpuName
                                : Cuda.supported ? "No NVIDIA GPU detected"
                                : "Apple GPU — Metal (MPS)"
                            color: pal.TXT; font.pixelSize: sc.textMd; font.weight: Font.DemiBold
                        }
                        Text {
                            Layout.fillWidth: true; wrapMode: Text.WordWrap
                            text: !Cuda.checked ? "Probing nvidia-smi + the torch sidecar…"
                                : !Cuda.supported ? "macOS uses the built-in Metal (MPS) backend — no CUDA install is needed."
                                : Cuda.installed ? ("CUDA torch " + Cuda.installedVersion + " installed — analysis runs on the GPU.")
                                : ("Using the bundled CPU torch " + (Cuda.bundledVersion || "—") + " — install CUDA below to use the GPU.")
                            color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                        }
                    }
                    Badge {
                        visible: Cuda.checked && Cuda.supported
                        text: Cuda.installed ? "GPU ready" : (Cuda.gpuName !== "" ? "CPU only" : "no GPU")
                        tone: Cuda.installed ? pal.SUCCESS : pal.TXT_MUTED; dot: true
                    }
                }
            }

            Group {
                visible: Cuda.checked && Cuda.supported && Cuda.gpuName !== ""
                title: Cuda.installed ? "Update or remove" : "Install CUDA acceleration"
                ColumnLayout {
                    Layout.fillWidth: true; spacing: sc.sp4
                    Layout.leftMargin: sc.sp5; Layout.rightMargin: sc.sp5
                    Layout.bottomMargin: sc.sp5
                    Text {
                        Layout.fillWidth: true; wrapMode: Text.WordWrap
                        text: Cuda.installed
                            ? "A CUDA torch build is installed. Reinstall to pull the newest matching CUDA wheel, or remove it to fall back to the bundled CPU build."
                            : "Downloads the CUDA build of PyTorch matched to your GPU + interpreter so detection runs on the GPU. ~2–3 GB; a restart is needed afterwards."
                        color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                    }

                    // download / extract progress
                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: Cuda.busy
                        spacing: sc.sp3
                        Rectangle {
                            Layout.fillWidth: true; implicitHeight: 8; radius: 4; clip: true
                            color: pal.PANEL; border.width: 1; border.color: pal.BORDER
                            Rectangle {
                                height: parent.height; radius: 4
                                visible: Cuda.progress >= 0
                                width: Math.max(0, Math.min(1, Cuda.progress)) * parent.width
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: pal.SUCCESS }
                                    GradientStop { position: 1.0; color: pal.ACC }
                                }
                                Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                            }
                            IndeterminateShimmer { active: Cuda.busy && Cuda.progress < 0 }
                        }
                        RowLayout {
                            Layout.fillWidth: true; spacing: sc.sp3
                            Text { Layout.fillWidth: true; text: Cuda.status || "Working…"
                                   color: pal.TXT_MUTED; font.pixelSize: sc.textXs; elide: Text.ElideRight }
                            Text { visible: Cuda.progress >= 0
                                   text: Math.round(Cuda.progress * 100) + "%"
                                   color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo" }
                        }
                    }

                    Alert {
                        Layout.fillWidth: true
                        visible: Cuda.error !== ""
                        severity: "warn"
                        text: Cuda.error
                    }
                    Alert {
                        Layout.fillWidth: true
                        visible: !Cuda.busy && Cuda.status.indexOf("restart") >= 0
                        severity: "success"
                        text: Cuda.status
                    }

                    RowLayout {
                        spacing: sc.sp3
                        visible: !Cuda.busy
                        Button { variant: "primary"
                                 text: Cuda.installed ? "Reinstall / update" : "Install CUDA"
                                 icon: "download"; onClicked: Cuda.install() }
                        Button { variant: "secondary"; text: "Remove"; icon: "trash-2"
                                 visible: Cuda.installed; onClicked: Cuda.uninstall() }
                        Button { variant: "ghost"; text: "Re-check"; icon: "refresh-cw"
                                 onClicked: Cuda.refresh() }
                    }
                    RowLayout {
                        visible: Cuda.busy
                        Button { variant: "secondary"; text: "Cancel"; icon: "x"
                                 onClicked: Cuda.cancel() }
                    }
                }
            }
        }
    }

    // ════════════════════════ UPDATES ═══════════════════════════════════════
    Component {
        id: updatesView
        ColumnLayout {
            spacing: sc.sp8
            Group {
                title: "Software updates"
                desc: "FIREFLY checks GitHub for new releases on launch. You decide when to install."
                Rectangle {                        // version row (full-width)
                    Layout.fillWidth: true; color: "transparent"
                    implicitHeight: verRow.implicitHeight + sc.sp5 * 2
                    Rectangle { anchors { left: parent.left; right: parent.right; top: parent.top }
                                height: 1; color: pal.BORDER }
                    RowLayout {
                        id: verRow
                        anchors.fill: parent
                        anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                        anchors.topMargin: sc.sp5; anchors.bottomMargin: sc.sp5
                        spacing: sc.sp4
                        Rectangle {
                            Layout.preferredWidth: 44; Layout.preferredHeight: 44
                            radius: sc.radius2xl
                            color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                            Icon { anchors.centerIn: parent; name: "microscope"; size: 22; color: pal.ACC }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true; Layout.preferredWidth: 0; spacing: 2
                            RowLayout {
                                spacing: sc.sp3
                                Text { text: "FIREFLY"; color: pal.TXT; font.pixelSize: sc.textMd
                                       font.weight: Font.DemiBold }
                                Text { text: "v" + Updates.version; color: pal.TXT_MUTED
                                       font.pixelSize: sc.textSm; font.family: "Menlo" }
                                Badge { text: Updates.checkError !== "" ? "Couldn't check"
                                              : Updates.updateAvailable ? "Update available" : "Up to date"
                                        tone: Updates.checkError !== "" ? pal.WARN
                                              : Updates.updateAvailable ? pal.ACC : pal.SUCCESS; dot: true }
                            }
                            Text {
                                text: Updates.checkError !== "" ? Updates.checkError
                                      : ("Last checked " + Updates.lastChecked + " · channel "
                                         + (root.rev, Settings.getStr("updates/channel", "Stable")).toLowerCase())
                                color: Updates.checkError !== "" ? pal.WARN : pal.TXT_MUTED
                                font.pixelSize: sc.textXs
                            }
                        }
                        Button {
                            variant: "secondary"
                            text: Updates.checking ? "Checking…" : "Check now"
                            icon: Updates.checking ? "loader-circle" : "refresh-cw"
                            spin: Updates.checking
                            enabled: !Updates.checking
                            onClicked: Updates.checkNow()
                        }
                    }
                }
                Rectangle {                        // update card (only when available)
                    Layout.fillWidth: true; color: "transparent"
                    visible: Updates.updateAvailable
                    implicitHeight: visible ? upCard.implicitHeight + sc.sp5 + sc.sp3 : 0
                    Rectangle {
                        id: upCard
                        anchors.fill: parent
                        anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                        anchors.topMargin: 0; anchors.bottomMargin: sc.sp3
                        implicitHeight: upCol.implicitHeight + sc.sp4 * 2
                        radius: sc.radiusLg; color: pal.PANEL_ALT
                        border.width: 1; border.color: pal.BORDER
                        Rectangle { anchors { left: parent.left; top: parent.top; bottom: parent.bottom }
                                    width: 3; radius: 1.5; color: pal.ACC }
                        ColumnLayout {
                            id: upCol
                            x: sc.sp5; y: sc.sp4; width: parent.width - sc.sp5 - sc.sp4
                            spacing: sc.sp3
                            Text {
                                visible: Updates.returningToStable
                                text: "Return to the stable release"
                                color: pal.ACC; font.pixelSize: sc.textXs; font.weight: Font.DemiBold
                            }
                            Text { text: Updates.latestTag; color: pal.TXT; font.pixelSize: sc.textMd
                                   font.family: "Menlo"; font.weight: Font.DemiBold }
                            Text {
                                Layout.fillWidth: true; wrapMode: Text.WordWrap
                                // render the GitHub notes as Markdown (bold
                                // phrases, code spans, bullet lists) instead of
                                // dumping the raw ## / ** / `` source.
                                textFormat: Text.MarkdownText
                                text: Updates.releaseBody !== "" ? Updates.releaseBody
                                      : Updates.returningToStable
                                          ? "You're on a pre-release build. Switch back to the stable release."
                                          : "A new release is available on GitHub."
                                color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                                lineHeight: 1.25
                                maximumLineCount: 12; elide: Text.ElideRight
                                onLinkActivated: (link) => Qt.openUrlExternally(link)
                            }
                            // download progress (real in-app update)
                            ColumnLayout {
                                Layout.fillWidth: true
                                visible: Updates.installing
                                spacing: sc.sp2
                                Rectangle {
                                    Layout.fillWidth: true
                                    implicitHeight: 8; radius: 4; clip: true
                                    color: pal.PANEL; border.width: 1; border.color: pal.BORDER
                                    Rectangle {                       // determinate fill
                                        height: parent.height; radius: 4
                                        visible: Updates.installProgress >= 0
                                        width: Math.max(0, Math.min(1, Updates.installProgress)) * parent.width
                                        gradient: Gradient {
                                            orientation: Gradient.Horizontal
                                            GradientStop { position: 0.0; color: pal.SUCCESS }
                                            GradientStop { position: 1.0; color: pal.ACC }
                                        }
                                        Behavior on width { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160 } }
                                    }
                                    IndeterminateShimmer { active: Updates.installing && Updates.installProgress < 0 }
                                }
                                RowLayout {
                                    Layout.fillWidth: true; spacing: sc.sp3
                                    Text { Layout.fillWidth: true; text: Updates.installStatus || "Working…"
                                           color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                                    Text { visible: Updates.installProgress >= 0
                                           text: Math.round(Updates.installProgress * 100) + "%"
                                           color: pal.TXT_MUTED; font.pixelSize: sc.textXs; font.family: "Menlo" }
                                }
                            }
                            // error (unverifiable download / no installer / from-source)
                            Alert {
                                Layout.fillWidth: true
                                visible: Updates.installError !== ""
                                severity: "warn"
                                text: Updates.installError
                            }
                            RowLayout {
                                spacing: sc.sp3
                                Layout.topMargin: sc.sp4      // breathing room below the notes
                                visible: !Updates.installing
                                Button {
                                    variant: "primary"
                                    // when the background pre-fetch has already staged
                                    // a verified installer, install is instant
                                    text: Updates.updateDownloaded ? "Restart & install" : "Download & install"
                                    icon: Updates.updateDownloaded ? "refresh-cw" : "download"
                                    onClicked: Updates.downloadAndInstall()
                                }
                                Button { variant: "ghost"; text: "Release notes"
                                         onClicked: Updates.openReleasePage() }
                                Item { Layout.fillWidth: true }
                                RowLayout {
                                    visible: Updates.updateDownloaded; spacing: sc.sp1
                                    Icon { name: "circle-check"; size: 13; color: pal.SUCCESS }
                                    Text { text: "Downloaded"; color: pal.SUCCESS
                                           font.pixelSize: sc.textXs }
                                }
                            }
                        }
                    }
                }
                Rectangle {                        // pre-release notice (notify-only)
                    Layout.fillWidth: true; color: "transparent"
                    visible: Updates.prereleaseAvailable
                    implicitHeight: visible ? preRow.implicitHeight + sc.sp4 * 2 : 0
                    Rectangle { anchors { left: parent.left; right: parent.right; top: parent.top }
                                height: 1; color: pal.BORDER }
                    RowLayout {
                        id: preRow
                        anchors.fill: parent
                        anchors.leftMargin: sc.sp5; anchors.rightMargin: sc.sp5
                        anchors.topMargin: sc.sp4; anchors.bottomMargin: sc.sp4
                        spacing: sc.sp3
                        Icon { name: "sparkles"; size: 16; color: pal.WARN }
                        ColumnLayout {
                            Layout.fillWidth: true; Layout.preferredWidth: 0; spacing: 0
                            Text { text: "Pre-release " + Updates.prereleaseTag + " is available"
                                   color: pal.TXT; font.pixelSize: sc.textSm; font.weight: Font.DemiBold }
                            Text { Layout.fillWidth: true; wrapMode: Text.WordWrap
                                   text: "You're on the Stable channel, so this beta won't be installed automatically."
                                   color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
                        }
                        Button { variant: "ghost"; text: "Release notes"
                                 onClicked: Updates.openPrereleasePage() }
                    }
                }
            }
            Group {
                title: "Update behaviour"
                PrefRow {
                    label: "Check for updates on launch"
                    desc: "A quiet background request to the GitHub releases API."
                    Switch { checked: (root.rev, Settings.getBool("updates/auto_check", true))
                             onToggled: (c) => Settings.setValue("updates/auto_check", c) }
                }
                PrefRow {
                    label: "Update channel"
                    desc: "Pre-release offers beta builds before they're promoted to stable."
                    Select { implicitWidth: 170; model: root.channelOpts
                             currentIndex: (root.rev, Math.max(0, root.channelOpts.indexOf(Settings.getStr("updates/channel", "Stable"))))
                             onPicked: (t) => Settings.setValue("updates/channel", t) }
                }
                PrefRow {
                    label: "Download in the background"
                    desc: "Fetch the installer automatically, then prompt to restart."
                    Switch { checked: (root.rev, Settings.getBool("updates/auto_download", false))
                             onToggled: (c) => Settings.setValue("updates/auto_download", c) }
                }
                PrefRow {
                    label: "Notify about pre-releases"; desc: "Even while on the stable channel."
                    Checkbox { checked: (root.rev, Settings.getBool("updates/notify_pre", false))
                             onToggled: (c) => Settings.setValue("updates/notify_pre", c) }
                }
            }
        }
    }
}
