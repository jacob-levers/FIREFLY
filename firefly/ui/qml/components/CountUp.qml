import QtQuick

// CountUp — an animated numeric value for metrics / counts. Bind `value` to the
// real number; the displayed figure tweens to it (OutCubic, ~520 ms). It is a
// Text, so set color / font on it as usual (metrics use the mono face).
//
//   CountUp { value: Results.trackCount; decimals: 0
//             color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true
//             font.family: "Menlo" }
//   CountUp { value: Results.medianD; decimals: 2; suffix: "" ; ... }
//
// Reduce-motion: the Behavior is disabled, so it snaps to the value.
Text {
    id: root
    property real   value: 0
    property int    decimals: 0
    property string prefix: ""
    property string suffix: ""
    property real   _shown: 0

    text: prefix + _shown.toLocaleString(Qt.locale(), "f", decimals) + suffix
    onValueChanged: _shown = value
    Component.onCompleted: _shown = value

    Behavior on _shown {
        enabled: !Theme.reducedMotion
        NumberAnimation { duration: 520; easing.type: Easing.OutCubic }
    }
}
