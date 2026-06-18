import QtQuick

// Surface card: panel fill + 1px hairline + lg radius (the design system's
// flat-with-borders hierarchy). `raised` adds the subtle elevation token.
Rectangle {
    id: root
    property bool raised: false
    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    color: pal.PANEL
    radius: sc.radiusLg
    border.width: 1
    border.color: pal.BORDER

    // optional depth (degrades to nothing in flat themes / reduced contexts)
    Rectangle {
        visible: root.raised
        z: -1
        anchors.fill: parent
        anchors.topMargin: 2
        anchors.bottomMargin: -3
        radius: root.radius
        color: "#000000"
        opacity: 0.28
    }
}
