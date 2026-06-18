import QtQuick
import QtQuick.Layouts

// One sidebar parameter row: label + the right control for its kind (Switch /
// Select / SpinBox), bound to the SidebarController. Value + enabled are read
// reactively off Sidebar.revision so a single notify refreshes the sidebar.
RowLayout {
    id: root
    required property var field            // a Sidebar.fields(section) entry
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    // re-evaluated whenever any sidebar value changes
    readonly property bool en: (Sidebar.revision, Sidebar.isEnabled(field.key))

    Layout.fillWidth: true
    spacing: sc.sp3
    opacity: en ? 1.0 : 0.45

    Text {
        text: field.label
        color: pal.TXT_MUTED
        font.pixelSize: sc.textXs
        Layout.fillWidth: true
        Layout.preferredWidth: 0
        elide: Text.ElideRight
    }

    Loader {
        id: ctl
        Layout.preferredWidth: field.kind === "bool" ? -1 : 132
        sourceComponent: field.kind === "bool" ? switchC
                       : field.kind === "combo" ? selectC
                       : spinC
    }

    // ── bool ─────────────────────────────────────────────────────────────
    Component {
        id: switchC
        Switch {
            enabled: root.en
            checked: (Sidebar.revision, Sidebar.get(root.field.key) === true)
            onToggled: (c) => Sidebar.setValue(root.field.key, c)
        }
    }

    // ── combo ────────────────────────────────────────────────────────────
    Component {
        id: selectC
        Select {
            enabled: root.en
            model: root.field.items
            currentIndex: {
                Sidebar.revision
                return Math.max(0, root.field.items.indexOf(Sidebar.get(root.field.key)))
            }
            onPicked: (t) => Sidebar.setValue(root.field.key, t)
        }
    }

    // ── numeric ──────────────────────────────────────────────────────────
    Component {
        id: spinC
        SpinBox {
            enabled: root.en
            from: root.field.min !== null ? root.field.min : 0
            to: root.field.max !== null ? root.field.max : 1e9
            step: root.field.step !== null ? root.field.step : 1
            decimals: root.field.decimals !== null ? root.field.decimals : 0
            suffix: root.field.suffix
            special: root.field.special
            value: (Sidebar.revision, Sidebar.get(root.field.key))
            onCommitted: (v) => Sidebar.setValue(root.field.key, v)
        }
    }
}
