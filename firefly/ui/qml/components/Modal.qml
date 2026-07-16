import QtQuick
import QtQuick.Layouts

// Modal — a centered modal dialog: dimmed backdrop (swallows clicks) + a titled
// Card. `open`/`close()` toggle it; bridges Embed.setModalOpen so the native
// viewer island hides while it's up. Children go in the body column. Animated
// entrance/exit: backdrop fades to 0.55, card fades + scales 0.96→1 + rises
// 8 px (OutCubic); exit is the reverse, slightly faster. Reduce-motion safe.
Item {
    id: root
    anchors.fill: parent
    property string title: ""
    property bool opened: false
    // dismissable=false → no backdrop-click / Esc / × close (used for a
    // blocking "loading…" popup that only its own action can dismiss).
    property bool dismissable: true
    default property alias body: bodyCol.data
    signal closed()

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property int durIn:  Theme.reducedMotion ? 0 : 220
    readonly property int durOut: Theme.reducedMotion ? 0 : 160

    // Stay mounted while the exit tween plays, then go invisible.
    visible: opened || backdrop.opacity > 0.001 || card.opacity > 0.001
    z: 1000

    function open()  { opened = true }
    function close() { opened = false; root.closed() }
    // Toggle the native viewer island's hide reactively, so `opened` works
    // whether set via open()/close() or bound directly to a controller flag.
    onOpenedChanged: { try { Embed.setModalOpen(opened) } catch (e) {} }

    Rectangle {
        id: backdrop
        anchors.fill: parent
        color: "#000000"
        opacity: root.opened ? 0.55 : 0
        Behavior on opacity { NumberAnimation { duration: root.opened ? root.durIn : root.durOut } }
        MouseArea { anchors.fill: parent; onClicked: if (root.dismissable) root.close() }
    }

    Card {
        id: card
        anchors.centerIn: parent
        width: Math.min(560, parent.width - sc.sp16)
        implicitHeight: col.implicitHeight + sc.sp6 * 2
        raised: true

        opacity: root.opened ? 1 : 0
        scale:   root.opened ? 1 : 0.96
        Behavior on opacity { NumberAnimation { duration: root.opened ? root.durIn : root.durOut; easing.type: Easing.OutCubic } }
        Behavior on scale   { NumberAnimation { duration: root.opened ? root.durIn : root.durOut; easing.type: Easing.OutCubic } }
        transform: Translate {
            y: root.opened ? 0 : 8
            Behavior on y { NumberAnimation { duration: root.opened ? root.durIn : root.durOut; easing.type: Easing.OutCubic } }
        }

        // swallow clicks on the card so the backdrop MouseArea doesn't close it
        MouseArea { anchors.fill: parent }

        ColumnLayout {
            id: col
            x: sc.sp6; y: sc.sp6
            width: parent.width - sc.sp6 * 2
            spacing: sc.sp4
            RowLayout {
                Layout.fillWidth: true
                // no title + not dismissable → hide the header entirely, so a
                // blocking "loading…" modal can present a clean centred body.
                visible: root.title !== "" || root.dismissable
                Text { text: root.title; color: pal.TXT; font.pixelSize: sc.textXl; font.bold: true
                       Layout.fillWidth: true }
                IconButton { visible: root.dismissable; icon: "x"; tip: "Close"; onClicked: root.close() }
            }
            ColumnLayout { id: bodyCol; Layout.fillWidth: true; spacing: sc.sp4 }
        }
    }

    Keys.onEscapePressed: if (root.dismissable) root.close()
}
