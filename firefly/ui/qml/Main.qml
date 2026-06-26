import QtQuick
import QtQuick.Layouts
import QtQuick.Controls as QQC
import "components"
import "tabs"

// FIREFLY QML shell (Phase 1 foundation): themed header + landing/main pages.
// Hosted by a QQuickWidget (root is an Item, the QMainWindow owns the window).
Item {
    id: root
    implicitWidth: 1100
    implicitHeight: 760

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    // Quietly check GitHub for a newer release shortly after launch (honours the
    // Preferences ▸ "Check for updates on launch" toggle).  This is what surfaces
    // the header update pill + the Preferences banner without the user asking.
    Timer {
        running: true; interval: 1500; repeat: false
        onTriggered: if (Settings.getBool("updates/auto_check", true)) Updates.checkNow()
    }

    Rectangle {            // app canvas
        anchors.fill: parent
        color: pal.BG
        // While a full-window modal (Preferences / ROI editor) is open, make the
        // whole app behind it non-interactive.  A backdrop MouseArea alone does
        // NOT reliably swallow the app's TapHandler-based controls (a Qt 6
        // event-delivery gap), so disable the subtree outright — this stops every
        // MouseArea AND pointer handler under it.
        enabled: !(prefs.opened || Roi.editing)

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            // ── header ───────────────────────────────────────────────
            Rectangle {
                // The landing carries its own branding in the left rail + a floating
                // gear, so the app chrome header only shows in the main UI.
                visible: App.page === "main"
                Layout.fillWidth: true
                Layout.preferredHeight: 52
                color: pal.PANEL
                Rectangle {            // hairline base
                    anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
                    height: 1; color: pal.BORDER
                }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: sc.sp8
                    anchors.rightMargin: sc.sp8
                    spacing: sc.sp4
                    Text {
                        text: "FIRE"; color: pal.TXT
                        font.pixelSize: 17; font.bold: true
                        font.letterSpacing: 0.5
                    }
                    Text {
                        text: "FLY"; color: pal.ACC
                        font.pixelSize: 17; font.bold: true
                        font.letterSpacing: 0.5
                        Layout.leftMargin: -sc.sp4
                    }
                    Text {
                        text: "v" + appVersion; color: pal.TXT_MUTED
                        font.pixelSize: sc.textXs
                        Layout.leftMargin: sc.sp2
                    }
                    Item { Layout.fillWidth: true }
                    // update-available pill → opens Preferences ▸ Updates
                    Rectangle {
                        id: updatePill
                        visible: Updates.updateAvailable
                        Layout.alignment: Qt.AlignVCenter
                        implicitHeight: 24
                        implicitWidth: upPillRow.implicitWidth + sc.sp3 * 2
                        radius: height / 2
                        color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b,
                                       upPillHov.hovered ? 0.22 : 0.14)
                        border.width: 1
                        border.color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.40)
                        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                        RowLayout {
                            id: upPillRow
                            anchors.centerIn: parent
                            spacing: sc.sp2
                            Rectangle {                         // pulsing status dot
                                width: 7; height: 7; radius: 3.5; color: pal.ACC
                                Layout.alignment: Qt.AlignVCenter
                                SequentialAnimation on opacity {
                                    running: Updates.updateAvailable && !Theme.reducedMotion
                                    loops: Animation.Infinite
                                    NumberAnimation { from: 1.0; to: 0.35; duration: 900; easing.type: Easing.InOutSine }
                                    NumberAnimation { from: 0.35; to: 1.0; duration: 900; easing.type: Easing.InOutSine }
                                }
                            }
                            Text {
                                text: Updates.installing ? "Updating…" : "Update available"
                                color: pal.ACC; font.pixelSize: sc.textXs; font.weight: Font.DemiBold
                            }
                        }
                        HoverHandler { id: upPillHov; cursorShape: Qt.PointingHandCursor }
                        TapHandler { onTapped: prefs.open("updates") }   // jump to Updates
                    }
                    IconButton { icon: "settings"; tip: "Preferences (⌘,)"; size: 28
                                 onClicked: prefs.open() }
                    // Home affordance (only in the main UI)
                    Text {
                        visible: App.page === "main"
                        text: "‹ Home"; color: pal.TXT_MUTED
                        font.pixelSize: sc.textSm
                        Layout.leftMargin: sc.sp2
                        MouseArea {
                            anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                            onClicked: App.goLanding()
                        }
                    }
                }
            }

            // ── body: landing or main ────────────────────────────────
            // Incoming page fades + rises 8px (280ms) — the workflow "settles
            // into place" rather than hard-cutting.
            Loader {
                id: pageLoader
                Layout.fillWidth: true
                Layout.fillHeight: true
                sourceComponent: App.page === "landing" ? landingPage : mainPage
                opacity: 0
                transform: Translate { id: pageT; y: 8 }
                onLoaded: { pageT.y = 8; pageLoader.opacity = 0; pageShow.restart() }
                ParallelAnimation {
                    id: pageShow
                    NumberAnimation { target: pageLoader; property: "opacity"; to: 1
                                      duration: Theme.reducedMotion ? 0 : 280; easing.type: Easing.OutCubic }
                    NumberAnimation { target: pageT; property: "y"; to: 0
                                      duration: Theme.reducedMotion ? 0 : 280; easing.type: Easing.OutCubic }
                }
            }
        }
    }

    // Manual-polygon ROI editor — a full-window modal shown while drawing.
    Loader {
        anchors.fill: parent
        active: Roi.editing
        sourceComponent: Component { RoiOverlay {} }
    }

    // Preferences modal + ⌘, shortcut.
    PreferencesDialog { id: prefs }
    Shortcut { sequence: StandardKey.Preferences; onActivated: prefs.open() }

    // ── global keyboard shortcuts ────────────────────────────────────────
    // ⌘1…⌘5 jump to a tab; ⌘↵ starts/stops a run (Qt maps "Ctrl" → ⌘ on macOS).
    // Gated to the main UI with no full-window modal (Preferences / ROI editor) up.
    readonly property bool shortcutsLive: App.page === "main" && !prefs.opened && !Roi.editing
    Shortcut { sequence: "Ctrl+1"; enabled: root.shortcutsLive; onActivated: App.setTab(0) }
    Shortcut { sequence: "Ctrl+2"; enabled: root.shortcutsLive; onActivated: App.setTab(1) }
    Shortcut { sequence: "Ctrl+3"; enabled: root.shortcutsLive; onActivated: App.setTab(2) }
    Shortcut { sequence: "Ctrl+4"; enabled: root.shortcutsLive; onActivated: App.setTab(3) }
    Shortcut { sequence: "Ctrl+5"; enabled: root.shortcutsLive; onActivated: App.setTab(4) }
    Shortcut {
        sequences: ["Ctrl+Return", "Ctrl+Enter"]
        enabled: root.shortcutsLive
        onActivated: Process.running ? Process.stop() : Process.start()
    }

    // Restart prompt once a CUDA install finishes (the GPU torch only loads on a
    // fresh process). Declared after Preferences so it layers above it.
    Modal {
        id: cudaRestart
        title: "Restart to use the GPU"
        Connections { target: Cuda; function onInstallCompleted() { cudaRestart.open() } }
        Text {
            Layout.fillWidth: true; wrapMode: Text.WordWrap
            text: "CUDA acceleration is installed. FIREFLY needs to restart so detection "
                + "runs on your GPU."
            color: pal.TXT_MUTED; font.pixelSize: sc.textSm; lineHeight: 1.25
        }
        RowLayout {
            Layout.fillWidth: true; Layout.topMargin: sc.sp2; spacing: sc.sp3
            Item { Layout.fillWidth: true }
            Button { variant: "secondary"; text: "Later"; onClicked: cudaRestart.close() }
            Button { variant: "primary"; text: "Restart now"; icon: "refresh-cw"
                     onClicked: { cudaRestart.close(); Cuda.restartNow() } }
        }
    }

    // ── landing page (two-column "Localiser" design) ─────────────────
    Component {
        id: landingPage
        Item {
            id: landing
            // Curated recent highlights (real versions / dates / summaries).
            readonly property var rupdates: [
                { v: "v2.76.25", d: "26 Jun 2026", s: "Live progress while a full report is generated." },
                { v: "v2.76.24", d: "26 Jun 2026", s: "Single analysis shows the combined multi-file series." },
                { v: "v2.76.22", d: "26 Jun 2026", s: "Auto-minmass runs now replay exactly." }
            ]
            readonly property var actions: [
                { icon: "scan-search", t: "Analyse a sample", d: "Run the full pipeline on one .czi / .tif file.", tab: 0, batch: false },
                { icon: "layers",      t: "Batch a folder",   d: "Process every file in a folder — in parallel on capable machines.", tab: 0, batch: true },
                { icon: "git-compare", t: "Compare & analyse", d: "Drop 2–12 conditions into one live comparison — figure, stats, significance.", tab: 2, batch: undefined },
                { icon: "waypoints",   t: "Visualise tracks", d: "Open a previous run in the interactive viewer.", tab: 3, batch: undefined }
            ]
            function _go(a) {
                if (a.batch !== undefined) Import.setBatchMode(a.batch)
                App.enterMain(a.tab)
            }

            RowLayout {
                anchors.fill: parent
                spacing: 0

                // ── left rail: branding + recent updates over the field ──
                Rectangle {
                    Layout.preferredWidth: 444
                    Layout.fillHeight: true
                    color: pal.BG
                    clip: true
                    MoleculeField { anchors.fill: parent; active: App.page === "landing" }
                    Rectangle { anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                                width: 1; color: pal.BORDER }

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 32
                        spacing: 0

                        RowLayout {                          // eyebrow
                            spacing: 7
                            Icon { name: "microscope"; size: 12; color: pal.WARN }
                            Text { text: "SINGLE-PARTICLE TRACKING · PALM"; color: pal.WARN
                                   font.pixelSize: 10; font.bold: true; font.letterSpacing: 1.8 }
                        }
                        RowLayout {                          // wordmark
                            Layout.topMargin: 16; spacing: 0
                            Text { text: "FIRE"; color: pal.TXT; font.pixelSize: 52
                                   font.weight: Font.ExtraBold; font.letterSpacing: 0.5 }
                            Text { text: "FLY"; color: pal.ACC; font.pixelSize: 52
                                   font.weight: Font.ExtraBold; font.letterSpacing: 0.5 }
                        }
                        Text {                               // tagline
                            Layout.topMargin: 16; Layout.maximumWidth: 330
                            wrapMode: Text.WordWrap; lineHeight: 1.5; textFormat: Text.StyledText
                            text: "Localise, link and analyse single fluorescent molecules in "
                                  + "<font face='Menlo'>.czi</font> / <font face='Menlo'>.tif</font> stacks."
                            color: pal.TXT_MUTED; font.pixelSize: 14
                        }

                        Item { Layout.fillHeight: true }     // flexible spacer

                        ColumnLayout {                       // recent updates log
                            Layout.fillWidth: true; spacing: 0
                            RowLayout {
                                Layout.bottomMargin: 13; spacing: 7
                                Icon { name: "history"; size: 13; color: pal.WARN }
                                Text { text: "RECENT UPDATES"; color: pal.WARN
                                       font.pixelSize: 10; font.bold: true; font.letterSpacing: 1.0 }
                            }
                            Repeater {
                                model: landing.rupdates
                                delegate: RowLayout {
                                    required property var modelData
                                    required property int index
                                    readonly property bool last: index === landing.rupdates.length - 1
                                    Layout.fillWidth: true
                                    spacing: 11
                                    Item {                   // timeline gutter (height tracks the body)
                                        Layout.preferredWidth: 9
                                        Layout.alignment: Qt.AlignTop
                                        Layout.preferredHeight: rbody.implicitHeight + (last ? 0 : 14)
                                        Rectangle {          // connector down to the next dot
                                            visible: !last
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            y: 13; width: 1
                                            height: Math.max(0, parent.height - 13); color: pal.BORDER
                                        }
                                        Rectangle {          // ring chip (active items)
                                            visible: !last
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            y: 2; width: 13; height: 13; radius: 6.5
                                            color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                                        }
                                        Rectangle {          // dot
                                            anchors.horizontalCenter: parent.horizontalCenter
                                            y: 5; width: 7; height: 7; radius: 3.5
                                            color: last ? pal.TXT_MUTED : pal.ACC
                                        }
                                    }
                                    ColumnLayout {           // item body
                                        id: rbody
                                        Layout.fillWidth: true
                                        Layout.alignment: Qt.AlignTop
                                        Layout.bottomMargin: last ? 0 : 14
                                        spacing: 3
                                        RowLayout {
                                            spacing: 6
                                            Text { text: modelData.v; color: pal.TXT; font.family: "Menlo"
                                                   font.pixelSize: 12; font.weight: Font.DemiBold }
                                            Text { text: modelData.d; color: pal.TXT_MUTED; font.pixelSize: 10 }
                                        }
                                        Text { text: modelData.s; color: pal.TXT_MUTED
                                               Layout.fillWidth: true; wrapMode: Text.WordWrap
                                               font.pixelSize: 12; lineHeight: 1.4 }
                                    }
                                }
                            }
                        }
                        Text { Layout.topMargin: 14; text: "By Jacob Levers"
                               color: pal.TXT_MUTED; font.pixelSize: 11 }
                    }
                }

                // ── right column: the four workflows ─────────────────
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    ColumnLayout {
                        anchors.left: parent.left; anchors.leftMargin: 48
                        anchors.verticalCenter: parent.verticalCenter
                        width: Math.min(600, parent.width - 96)
                        spacing: 0
                        Text { text: "START SOMETHING"; color: pal.TXT_MUTED
                               font.pixelSize: 11; font.bold: true; font.letterSpacing: 1.76 }
                        Text { Layout.topMargin: 6; Layout.bottomMargin: 22
                               text: "What would you like to do?"; color: pal.TXT
                               font.pixelSize: 30; font.bold: true }
                        ColumnLayout {
                            Layout.fillWidth: true; spacing: 10
                            Repeater {
                                model: landing.actions
                                delegate: Rectangle {
                                    required property var modelData
                                    Layout.fillWidth: true
                                    implicitHeight: arRow.implicitHeight + 16 * 2
                                    radius: 10
                                    color: arHov.hovered ? pal.PANEL_ALT : pal.PANEL
                                    border.width: 1
                                    border.color: arHov.hovered ? pal.ACC : pal.BORDER
                                    Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 140 } }
                                    Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 140 } }
                                    transform: Translate {
                                        y: arHov.hovered ? -2 : 0
                                        Behavior on y { NumberAnimation { duration: Theme.reducedMotion ? 0 : 140; easing.type: Easing.OutCubic } }
                                    }
                                    RowLayout {
                                        id: arRow
                                        anchors.fill: parent
                                        anchors.leftMargin: 18; anchors.rightMargin: 18
                                        spacing: 15
                                        Rectangle {              // icon chip
                                            Layout.alignment: Qt.AlignVCenter
                                            width: 46; height: 46; radius: 10
                                            color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.12)
                                            border.width: 1
                                            border.color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.22)
                                            Icon { anchors.centerIn: parent; name: modelData.icon; size: 21; color: pal.ACC }
                                        }
                                        ColumnLayout {
                                            Layout.fillWidth: true; spacing: 2
                                            Text { text: modelData.t; color: pal.TXT
                                                   font.pixelSize: 15; font.bold: true }
                                            Text { text: modelData.d; color: pal.TXT_MUTED
                                                   Layout.fillWidth: true; wrapMode: Text.WordWrap
                                                   font.pixelSize: 12 }
                                        }
                                        Icon { name: "chevron-right"; size: 18; color: pal.TXT_MUTED }
                                    }
                                    HoverHandler { id: arHov; cursorShape: Qt.PointingHandCursor }
                                    TapHandler { onTapped: landing._go(modelData) }
                                }
                            }
                        }
                    }
                }
            }

            // ── floating top-right: update pill + Preferences gear ───
            RowLayout {
                z: 11
                anchors { top: parent.top; right: parent.right; topMargin: 14; rightMargin: 14 }
                spacing: sc.sp2
                Rectangle {
                    visible: Updates.updateAvailable
                    Layout.alignment: Qt.AlignVCenter
                    implicitHeight: 26; implicitWidth: lupRow.implicitWidth + sc.sp3 * 2
                    radius: height / 2
                    color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, lupHov.hovered ? 0.22 : 0.14)
                    border.width: 1; border.color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.40)
                    RowLayout {
                        id: lupRow; anchors.centerIn: parent; spacing: sc.sp2
                        Rectangle { width: 7; height: 7; radius: 3.5; color: pal.ACC }
                        Text { text: Updates.installing ? "Updating…" : "Update available"
                               color: pal.ACC; font.pixelSize: sc.textXs; font.weight: Font.DemiBold }
                    }
                    HoverHandler { id: lupHov; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: prefs.open("updates") }
                }
                IconButton { icon: "settings"; tip: "Preferences (⌘,)"; size: 28
                             onClicked: prefs.open() }
            }
        }
    }

    // ── main page (left parameter dock + tab bar + content) ──────────
    Component {
        id: mainPage
        RowLayout {
            spacing: 0

            // ── persistent analysis-parameter dock (Import + Analysis) ───
            Rectangle {
                id: dock
                readonly property bool shown: App.currentTab === 0 || App.currentTab === 1
                Layout.preferredWidth: shown ? 312 : 0
                Layout.fillHeight: true
                visible: Layout.preferredWidth > 0
                clip: true
                color: pal.PANEL
                Behavior on Layout.preferredWidth {
                    NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic }
                }
                Rectangle {     // right hairline
                    anchors { right: parent.right; top: parent.top; bottom: parent.bottom }
                    width: 1; color: pal.BORDER
                }
                Flickable {
                    anchors.fill: parent
                    anchors.rightMargin: 1
                    contentWidth: width
                    contentHeight: paramCol.implicitHeight + sc.sp8
                    clip: true
                    ParameterSidebar {
                        id: paramCol
                        x: sc.sp4; y: sc.sp5
                        width: parent.width - sc.sp4 * 2
                    }
                }
            }

            // ── tab bar + per-tab content ────────────────────────────────
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 0
                RowLayout {                 // tab pills
                    Layout.fillWidth: true
                    Layout.margins: sc.sp4
                    spacing: sc.sp2
                    // One accent pill SLIDES to the active tab (instead of each
                    // tab popping its own fill). `activeItem` tracks the live
                    // delegate so the highlight binds without fragile child
                    // indexing over the Repeater.
                    Item {
                        id: tabbar
                        property Item activeItem: null
                        implicitWidth: tabRow.implicitWidth
                        implicitHeight: tabRow.implicitHeight
                        TabHighlight { target: tabbar.activeItem }
                        Row {
                            id: tabRow
                            spacing: sc.sp2
                            Repeater {
                                model: App.tabs
                                delegate: Item {
                                    required property int index
                                    required property string modelData
                                    readonly property bool active: App.currentTab === index
                                    implicitWidth: lbl.implicitWidth + sc.sp8 * 2
                                    implicitHeight: 30
                                    onActiveChanged: if (active) tabbar.activeItem = this
                                    Component.onCompleted: if (active) tabbar.activeItem = this
                                    Text {
                                        id: lbl; anchors.centerIn: parent; text: modelData
                                        color: active ? pal.ACC : pal.TXT_MUTED
                                        font.pixelSize: sc.textSm
                                        Behavior on color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                                    }
                                    TapHandler { onTapped: App.setTab(index) }
                                    HoverHandler { id: tabHov; cursorShape: Qt.PointingHandCursor }
                                    QQC.ToolTip.text: "⌘" + (index + 1)
                                    QQC.ToolTip.delay: 600
                                    QQC.ToolTip.visible: tabHov.hovered
                                }
                            }
                        }
                    }
                    Item { Layout.fillWidth: true }
                }
                Item {                       // per-tab content stage
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    // Tabs 0–2 (+ coming-soon) swap through one Loader.  The
                    // Visualise tab is *not* loaded here — it lives below as an
                    // always-present layer so its floating panels can animate
                    // out when you leave it (a switching Loader would destroy
                    // the tab the instant currentTab changes, killing the exit).
                    Loader {
                        id: tabLoader
                        anchors.fill: parent
                        active: App.currentTab !== 3
                        sourceComponent: App.currentTab === 0 ? importTab
                                       : App.currentTab === 1 ? processTab
                                       : App.currentTab === 2 ? analysisTab
                                       : App.currentTab === 4 ? hyperflyTab
                                       : comingSoon
                        // Fast fade between tabs (no horizontal slide — the app
                        // has no spatial tab order). 180ms.
                        opacity: 0
                        onLoaded: { tabLoader.opacity = 0; tabFade.restart() }
                        NumberAnimation { id: tabFade; target: tabLoader; property: "opacity"; to: 1
                                          duration: Theme.reducedMotion ? 0 : 180; easing.type: Easing.OutCubic }
                    }

                    // Always-loaded Visualise chrome (panels + viewer anchor).
                    // Transparent + disabled when off-tab, so it never blocks
                    // the tab below; its panels slide/fade on App.currentTab.
                    VisualiseTab { anchors.fill: parent }
                }
            }
        }
    }

    // ── tab content components ───────────────────────────────────────
    Component { id: importTab; ImportTab {} }
    Component { id: processTab; ProcessTab {} }
    Component { id: analysisTab; AnalysisTab {} }
    Component { id: hyperflyTab; HyperflyTab {} }
    Component {
        id: comingSoon
        Item {
            Text {
                anchors.centerIn: parent
                text: App.tabs[App.currentTab] + " — coming soon"
                color: pal.TXT_MUTED; font.pixelSize: sc.textLg
            }
        }
    }

    // ── global toast (run-manifest replay confirmation, …) ───────────
    Rectangle {
        id: toast
        z: 9999
        anchors { bottom: parent.bottom; horizontalCenter: parent.horizontalCenter; bottomMargin: sc.sp5 }
        visible: opacity > 0; opacity: 0
        color: pal.PANEL_ALT; radius: sc.radiusMd; border.width: 1; border.color: pal.BORDER
        implicitWidth: tlabel.implicitWidth + sc.sp5 * 2; implicitHeight: 34
        Text { id: tlabel; anchors.centerIn: parent; color: pal.TXT; font.pixelSize: sc.textSm }
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200 } }
        transform: Translate {                       // slide up 12px on enter
            y: toast.opacity > 0 ? 0 : 12
            Behavior on y { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200; easing.type: Easing.OutCubic } }
        }
        Timer { id: toastTimer; interval: 2600; onTriggered: toast.opacity = 0 }
        Connections {
            target: Sidebar
            function onManifestLoaded(msg) {
                tlabel.text = msg; toast.opacity = 1; toastTimer.restart()
            }
        }
    }

    // ── Visualise warning toast (load / export failures, etc.) ──────────
    // Vis.warn(title, message) used to be emitted into the void — wire it to a
    // dismissible, error-toned toast so a broken run folder / corrupt cluster
    // map / failed export no longer fails silently.
    Rectangle {
        id: visWarn
        z: 9999
        anchors { bottom: parent.bottom; horizontalCenter: parent.horizontalCenter; bottomMargin: sc.sp5 }
        visible: opacity > 0; opacity: 0
        width: Math.min(root.width - sc.sp16, 460)
        implicitHeight: vwRow.implicitHeight + sc.sp4 * 2
        color: pal.PANEL_ALT; radius: sc.radiusMd; border.width: 1; border.color: pal.DANGER
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200 } }
        transform: Translate {
            y: visWarn.opacity > 0 ? 0 : 12
            Behavior on y { NumberAnimation { duration: Theme.reducedMotion ? 0 : 200; easing.type: Easing.OutCubic } }
        }
        RowLayout {
            id: vwRow
            x: sc.sp5; y: sc.sp4; width: parent.width - sc.sp5 * 2; spacing: sc.sp3
            Icon { name: "triangle-alert"; size: 16; color: pal.DANGER; Layout.alignment: Qt.AlignTop }
            ColumnLayout {
                Layout.fillWidth: true; spacing: 1
                Text { id: vwTitle; color: pal.TXT; font.pixelSize: sc.textSm; font.weight: Font.DemiBold
                       Layout.fillWidth: true; wrapMode: Text.WordWrap }
                Text { id: vwMsg; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
                       Layout.fillWidth: true; wrapMode: Text.WordWrap }
            }
        }
        Timer { id: vwTimer; interval: 6000; onTriggered: visWarn.opacity = 0 }
        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: visWarn.opacity = 0 }
        Connections {
            target: Vis
            function onWarn(title, msg) {
                vwTitle.text = title; vwMsg.text = msg
                visWarn.opacity = 1; vwTimer.restart()
            }
        }
    }
}
