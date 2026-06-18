import QtQuick
import QtQuick.Layouts

// A compact motion-class legend: colour swatch + label per class. Fed a model of
// {label, colorHex} dicts (Results.motionClasses / Compare.motionClasses). Used
// both as a figure-card header chip and as a footer under the figure.
RowLayout {
    id: root
    property var model: []
    readonly property var sc: Theme.scale
    readonly property var pal: Theme.palette
    spacing: sc.sp4
    Repeater {
        model: root.model
        delegate: RowLayout {
            required property var modelData
            spacing: sc.sp2
            Rectangle { width: 8; height: 8; radius: 2; color: modelData.colorHex }
            Text { text: modelData.label; color: pal.TXT_MUTED; font.pixelSize: sc.textXs }
        }
    }
}
