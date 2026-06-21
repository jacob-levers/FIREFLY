import QtQuick

// Odometer — rolling-digit counter for a LIVE, fast-changing integer (locs this
// frame, frames processed). Unlike CountUp (one settle to a target), Odometer
// rolls each digit as the number updates continuously. Mono, right-aligned.
//
//   Odometer { value: Process.locsThisFrame; digits: 5 }
Row {
    id: root
    property int value: 0
    property int digits: 5
    property int pixelSize: 16
    readonly property var pal: Theme.palette

    Repeater {
        model: root.digits
        delegate: Item {
            required property int index
            // most-significant first: place value of this column
            readonly property int place: Math.pow(10, root.digits - 1 - index)
            readonly property int d: Math.floor(root.value / place) % 10
            width: root.pixelSize * 0.62; height: root.pixelSize * 1.2; clip: true
            Column {
                y: -d * parent.height
                Behavior on y { NumberAnimation { duration: Theme.reducedMotion ? 0 : 220; easing.type: Easing.OutCubic } }
                Repeater {
                    model: 10
                    delegate: Text {
                        required property int index
                        width: root.pixelSize * 0.62; height: root.pixelSize * 1.2
                        horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                        text: index
                        font.pixelSize: root.pixelSize; font.bold: true; font.family: "Menlo"
                        color: pal.TXT
                    }
                }
            }
        }
    }
}
