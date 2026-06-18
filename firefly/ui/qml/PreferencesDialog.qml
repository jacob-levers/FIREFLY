import QtQuick
import QtQuick.Layouts
import "components"

// Preferences modal (Phase 6d): Appearance (theme · reduce motion · log-D plot
// style) + Updates (auto-check · version). Theme/reduce-motion persist through
// ThemeController; the rest through SettingsController. Opened from the header
// gear / ⌘,.
Modal {
    id: root
    title: "Preferences"

    readonly property var logdLabels: ["Faceted (per-replicate)", "Ridgeline",
                                       "Overlaid KDEs", "Violins + points"]
    readonly property var logdValues: ["faceted", "ridgeline", "overlaid", "violin"]

    function _row(label, ctl) {}    // (doc) each setting is a label + control row

    // ── Appearance ───────────────────────────────────────────────────────
    Text { text: "APPEARANCE"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
           font.bold: true; font.letterSpacing: 1.5 }

    RowLayout {
        Layout.fillWidth: true; spacing: sc.sp4
        Text { text: "Theme"; color: pal.TXT; font.pixelSize: sc.textSm
               Layout.fillWidth: true; Layout.preferredWidth: 0 }
        Select {
            Layout.preferredWidth: 180
            model: Theme.themes
            currentIndex: Math.max(0, Theme.themes.indexOf(Theme.name))
            onPicked: (t) => Theme.setTheme(t)
        }
    }
    RowLayout {
        Layout.fillWidth: true; spacing: sc.sp4
        Text { text: "Reduce motion"; color: pal.TXT; font.pixelSize: sc.textSm
               Layout.fillWidth: true; Layout.preferredWidth: 0 }
        Switch { checked: Theme.reducedMotion; onToggled: (c) => Theme.reducedMotion = c }
    }
    RowLayout {
        Layout.fillWidth: true; spacing: sc.sp4
        Text { text: "log-D plot style"; color: pal.TXT; font.pixelSize: sc.textSm
               Layout.fillWidth: true; Layout.preferredWidth: 0 }
        Select {
            Layout.preferredWidth: 180
            model: root.logdLabels
            currentIndex: Math.max(0, root.logdValues.indexOf(
                              Settings.getStr("figures/logd_style", "overlaid")))
            onPicked: (t) => {
                var i = root.logdLabels.indexOf(t)
                if (i >= 0) Settings.setValue("figures/logd_style", root.logdValues[i])
            }
        }
    }

    Rectangle { Layout.fillWidth: true; height: 1; color: pal.BORDER }

    // ── Updates ──────────────────────────────────────────────────────────
    Text { text: "UPDATES"; color: pal.TXT_MUTED; font.pixelSize: sc.textXs
           font.bold: true; font.letterSpacing: 1.5 }
    RowLayout {
        Layout.fillWidth: true; spacing: sc.sp4
        Text { text: "Check for updates on startup"; color: pal.TXT; font.pixelSize: sc.textSm
               Layout.fillWidth: true; Layout.preferredWidth: 0 }
        Switch {
            checked: Settings.getBool("updates/auto_check", true)
            onToggled: (c) => Settings.setValue("updates/auto_check", c)
        }
    }
    RowLayout {
        Layout.fillWidth: true; spacing: sc.sp4
        Text { text: "Version"; color: pal.TXT_MUTED; font.pixelSize: sc.textSm
               Layout.fillWidth: true; Layout.preferredWidth: 0 }
        Text { text: "v" + appVersion; color: pal.TXT; font.pixelSize: sc.textSm
               font.family: "Menlo" }
    }
}
