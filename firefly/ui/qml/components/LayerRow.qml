import QtQuick
import QtQuick.Layouts

// One row of the Visualise layer rail: eye toggle · type icon · name · opacity
// bar.  Display-only of a layer model dict; emits intent the tab wires to the
// VisualiseController.  Tracks colour the dot/name by motion class.
RowLayout {
    id: root
    property string name: ""
    property string kind: "tracks"      // tracks | maxproj | superres | clusters | points
    property color tone: "#8b949e"
    property bool layerVisible: true
    property real opacity_: 1.0
    property int count: 0
    signal toggled(bool on)
    signal opacitySet(real v)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property string glyph:
        kind === "tracks"   ? "waypoints" :
        kind === "maxproj"  ? "image" :
        kind === "superres" ? "zap" :
        kind === "clusters" ? "circle-dot" : "circle-dot"

    spacing: sc.sp3
    Layout.fillWidth: true

    // eye toggle
    Item {
        implicitWidth: 18; implicitHeight: 18
        Icon {
            anchors.centerIn: parent
            name: root.layerVisible ? "eye" : "eye-off"
            size: 15
            color: root.layerVisible ? pal.ACC : pal.TXT_MUTED
        }
        MouseArea {
            anchors.fill: parent; cursorShape: Qt.PointingHandCursor
            onClicked: root.toggled(!root.layerVisible)
        }
    }

    // type icon, tinted to the layer colour
    Icon { name: root.glyph; size: 14; color: root.tone }

    // name (+ count for tracks)
    Text {
        Layout.fillWidth: true
        text: root.count > 0 ? (root.name + "  ·  " + root.count) : root.name
        color: root.layerVisible ? pal.TXT : pal.TXT_MUTED
        font.pixelSize: sc.textSm
        elide: Text.ElideRight
    }

    // opacity bar (click/drag to set)
    Rectangle {
        Layout.preferredWidth: 46
        implicitHeight: 6; radius: 3
        color: pal.PANEL_ALT
        border.width: 1; border.color: pal.BORDER
        Rectangle {
            height: parent.height; radius: 3
            width: Math.max(0, Math.min(1, root.opacity_)) * parent.width
            color: root.tone
            opacity: root.layerVisible ? 1.0 : 0.4
        }
        MouseArea {
            anchors.fill: parent
            onPressed: (m) => root.opacitySet(Math.max(0, Math.min(1, m.x / width)))
            onPositionChanged: (m) => root.opacitySet(Math.max(0, Math.min(1, m.x / width)))
        }
    }
}
