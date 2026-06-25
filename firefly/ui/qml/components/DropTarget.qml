import QtQuick

// DropTarget — a card that responds to drag-and-drop (Compare group cards, the
// Import drop zone). Animates its border/fill on hover-drag, and gives a single
// confirming flash when something lands. Wrap your card content as children.
//
//   DropTarget { id: groupCard; onDropped: Compare.addFolder(group, url)
//       /* card content */ }
Rectangle {
    id: root
    property bool dragActive: false        // bind to your DropArea.containsDrag
    signal landed()                        // call this when a drop is accepted
    default property alias content: holder.data

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    radius: sc.radiusLg
    color: dragActive ? Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.06) : pal.PANEL
    border.width: dragActive ? 2 : 1
    border.color: dragActive ? pal.ACC : pal.BORDER
    scale: dragActive ? 1.01 : 1.0

    Behavior on color        { ColorAnimation  { duration: Theme.reducedMotion ? 0 : 120 } }
    Behavior on border.color { ColorAnimation  { duration: Theme.reducedMotion ? 0 : 120 } }
    Behavior on scale        { NumberAnimation { duration: Theme.reducedMotion ? 0 : 120; easing.type: Easing.OutCubic } }

    // confirming flash on landed() (reduce-motion → no flash)
    function flash() { if (!Theme.reducedMotion) flashAnim.restart() }
    onLanded: flash()
    SequentialAnimation {
        id: flashAnim
        ColorAnimation { target: root; property: "color"; to: Qt.rgba(0.337, 0.827, 0.392, 0.12); duration: 120 }  // SUCCESS @ 12%
        PauseAnimation { duration: 260 }
        ColorAnimation { target: root; property: "color"; to: pal.PANEL; duration: 240 }
    }

    Item { id: holder; anchors.fill: parent }
}
