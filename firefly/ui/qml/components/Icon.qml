import QtQuick

// A themeable Lucide stroke icon, rendered + tinted by the Python "icon" image
// provider (backend-independent — works under the software scene-graph too).
Item {
    id: root
    property string name: ""
    property color color: "#e6edf3"
    property int size: 18

    implicitWidth: size
    implicitHeight: size

    Image {
        anchors.fill: parent
        sourceSize.width: root.size * 2      // crisp on HiDPI
        sourceSize.height: root.size * 2
        fillMode: Image.PreserveAspectFit
        smooth: true
        cache: true
        source: root.name
                ? ("image://icon/" + root.name + "/" + _hex(root.color))
                : ""
    }

    function _hex(c) {
        var s = "" + c;                      // "#rrggbb" or "#aarrggbb"
        return s.charAt(0) === "#" ? s.substring(s.length - 6) : s;
    }
}
