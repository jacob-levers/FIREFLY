import QtQuick
import QtQuick.Layouts

// A centered modal dialog: dimmed backdrop (swallows clicks) + a titled Card.
// `open`/`close()` toggle it; bridges Embed.setModalOpen so the native viewer
// island hides while it's up. Children go in the body column.
Item {
    id: root
    anchors.fill: parent
    property string title: ""
    property bool opened: false
    default property alias body: bodyCol.data
    signal closed()

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    visible: opened
    z: 1000

    function open()  { opened = true;  try { Embed.setModalOpen(true) } catch (e) {} }
    function close() { opened = false; try { Embed.setModalOpen(false) } catch (e) {}; root.closed() }

    Rectangle {
        anchors.fill: parent
        color: "#000000"; opacity: 0.55
        MouseArea { anchors.fill: parent; onClicked: root.close() }
    }

    Card {
        anchors.centerIn: parent
        width: Math.min(560, parent.width - sc.sp16)
        implicitHeight: col.implicitHeight + sc.sp6 * 2
        raised: true
        ColumnLayout {
            id: col
            x: sc.sp6; y: sc.sp6
            width: parent.width - sc.sp6 * 2
            spacing: sc.sp4
            RowLayout {
                Layout.fillWidth: true
                Text { text: root.title; color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true
                       Layout.fillWidth: true }
                IconButton { icon: "x"; tip: "Close"; onClicked: root.close() }
            }
            ColumnLayout { id: bodyCol; Layout.fillWidth: true; spacing: sc.sp4 }
        }
    }
    // swallow clicks on the card so the backdrop MouseArea doesn't close it
    Keys.onEscapePressed: root.close()
}
