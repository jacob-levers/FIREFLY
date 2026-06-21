import QtQuick

// Reveal — a content reveal for figures / images / plots that land after a
// render: fade in + a barely-there zoom-settle (1.02→1.0). Wrap the Image or
// figure container; flip `ready` when the render arrives.
//
//   Reveal { ready: Figure.ready
//       Image { source: Figure.url; ... }
//   }
Item {
    id: root
    property bool ready: false
    default property alias content: holder.data

    implicitWidth:  holder.childrenRect.width
    implicitHeight: holder.childrenRect.height

    Item {
        id: holder
        width: root.width; height: root.height
        opacity: root.ready ? 1 : 0
        scale:   root.ready ? 1 : 1.02
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 260; easing.type: Easing.OutCubic } }
        Behavior on scale   { NumberAnimation { duration: Theme.reducedMotion ? 0 : 260; easing.type: Easing.OutCubic } }
    }
}
