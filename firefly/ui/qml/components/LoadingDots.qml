import QtQuick
import QtQuick.Layouts

// LoadingDots — a three-dot "working…" indicator for indeterminate inline waits
// (loading a run, scanning a folder, computing stats). Each dot lifts + brightens
// in sequence. Reduce-motion → a static row of dots.
//
//   RowLayout { Text { text: "Processing" } LoadingDots {} }
Row {
    id: root
    property color tone: Theme.palette.TXT_MUTED
    spacing: 4
    Repeater {
        model: 3
        delegate: Rectangle {
            required property int index
            width: 5; height: 5; radius: 2.5; color: root.tone
            opacity: 0.3
            SequentialAnimation on opacity {
                running: !Theme.reducedMotion
                loops: Animation.Infinite
                PauseAnimation { duration: index * 160 }
                NumberAnimation { to: 1.0; duration: 300; easing.type: Easing.OutCubic }
                NumberAnimation { to: 0.3; duration: 300; easing.type: Easing.InCubic }
                PauseAnimation { duration: (2 - index) * 160 + 300 }
            }
        }
    }
}
