import QtQuick
import QtQuick.Layouts
import QtQuick.Controls as QQC      // namespaced so it doesn't shadow the local
                                    // SpinBox/Switch/Slider/Select components

// One sidebar parameter, stacked label-over-control: bool fields keep the label
// inline with a trailing Switch; combo / numeric fields put a small label above
// a full-width control. Value + enabled are read reactively off Sidebar.revision
// so a single notify refreshes the whole sidebar. Bound to SidebarController.
Column {
    id: root
    required property var field            // a Sidebar.fields(section) entry
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property bool en: (Sidebar.revision, Sidebar.isEnabled(field.key))
    readonly property bool isBool: field.kind === "bool"

    width: parent ? parent.width : implicitWidth
    spacing: sc.sp2                        // 4: label → control
    opacity: en ? 1.0 : 0.45

    // hover tooltip from the schema (the field's help text)
    readonly property string tip: (root.field && root.field.tooltip) ? root.field.tooltip : ""
    HoverHandler { id: fieldHov }
    QQC.ToolTip.text: root.tip
    QQC.ToolTip.delay: 450
    QQC.ToolTip.visible: root.tip !== "" && fieldHov.hovered

    // ── label (+ inline Switch for bool) ──────────────────────────────────
    RowLayout {
        width: parent.width
        spacing: sc.sp3
        Text {
            text: root.field.label
            color: pal.TXT_MUTED
            font.pixelSize: sc.textXs
            font.weight: Font.DemiBold
            Layout.fillWidth: true
            Layout.preferredWidth: 0
            elide: Text.ElideRight
        }
        // live real-units readout (e.g. a lag-time in frames shown as seconds) so
        // the user sees what the analysis / MSD curve actually uses
        Text {
            readonly property string hint: (Sidebar.revision,
                                            Sidebar.derivedHint(root.field.key))
            visible: hint !== ""
            text: hint
            color: pal.ACC
            font.pixelSize: sc.textXs
            font.weight: Font.DemiBold
            Layout.alignment: Qt.AlignVCenter
        }
        Loader {
            active: root.isBool; visible: root.isBool
            sourceComponent: switchC
        }
    }

    // ── control (combo / slider / numeric), full width ────────────────────
    Loader {
        width: parent.width
        active: !root.isBool; visible: !root.isBool
        sourceComponent: root.field.kind === "combo" ? selectC
                       : root.field.slider ? sliderC
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

    // ── slider (numeric, flagged slider:true) ─────────────────────────────
    Component {
        id: sliderC
        Slider {
            enabled: root.en
            from: root.field.min != null ? root.field.min : 0
            to: root.field.max != null ? root.field.max : 1
            step: root.field.step != null ? root.field.step : 0
            decimals: root.field.decimals != null ? root.field.decimals : 0
            suffix: root.field.suffix
            value: (Sidebar.revision, Sidebar.get(root.field.key))
            onMoved: (v) => Sidebar.setValue(root.field.key, v)
        }
    }

    // ── numeric ──────────────────────────────────────────────────────────
    Component {
        id: spinC
        SpinBox {
            enabled: root.en
            from: root.field.min != null ? root.field.min : 0
            to: root.field.max != null ? root.field.max : 1e9
            step: root.field.step != null ? root.field.step : 1
            decimals: root.field.decimals != null ? root.field.decimals : 0
            suffix: root.field.suffix
            special: root.field.special
            value: (Sidebar.revision, Sidebar.get(root.field.key))
            onCommitted: (v) => Sidebar.setValue(root.field.key, v)
        }
    }
}
