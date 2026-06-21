import QtQuick

// FadeRise — the universal FIREFLY entrance: fade + 8 px rise, OutCubic.
// Wrap any content; it animates in once laid out. Re-play by toggling
// `shown` false→true. Reduce-motion safe.
//
//   FadeRise { Layout.fillWidth: true
//       Alert { width: parent.width; severity: "success"; text: "Run finished." }
//   }
//
// Size it via a Layout (parent sets the size) or let it take its child's size.
Item {
    id: root
    property bool shown: false
    property int  duration: 220          // dur.slow
    property int  rise: 8                 // entrance-rise
    property int  delay: 0                // stagger: pass index * 35
    default property alias content: holder.data

    // take the child's size when not driven by a Layout
    implicitWidth:  holder.childrenRect.width
    implicitHeight: holder.childrenRect.height

    Component.onCompleted: delayTimer.start()   // play once, after layout
    Timer { id: delayTimer; interval: Theme.reducedMotion ? 0 : root.delay; onTriggered: root.shown = true }

    Item {
        id: holder
        width:  root.width
        height: root.height
        opacity: root.shown ? 1 : 0
        Behavior on opacity {
            NumberAnimation { duration: Theme.reducedMotion ? 0 : root.duration
                              easing.type: Easing.OutCubic }
        }
        transform: Translate {
            y: root.shown ? 0 : root.rise
            Behavior on y {
                NumberAnimation { duration: Theme.reducedMotion ? 0 : root.duration
                                  easing.type: Easing.OutCubic }
            }
        }
    }
}
