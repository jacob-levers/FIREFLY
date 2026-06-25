import QtQuick
import QtQuick.Layouts

// Destructive-action confirmation built on Modal. Set `title`, `message`,
// `confirmText`, and `action` (a function run when confirmed), then call open().
// Reduce-motion / backdrop / Esc behaviour all come from Modal.
Modal {
    id: root
    property string message: ""
    property string confirmText: "Remove"
    property var action: null            // function invoked on confirm

    Text {
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
        text: root.message
        color: pal.TXT_MUTED
        font.pixelSize: sc.textSm
        lineHeight: 1.3
    }
    RowLayout {
        Layout.fillWidth: true
        Layout.topMargin: sc.sp2
        spacing: sc.sp3
        Item { Layout.fillWidth: true }
        Button { variant: "secondary"; text: "Cancel"; onClicked: root.close() }
        Button {
            variant: "danger"; text: root.confirmText
            onClicked: { if (root.action) root.action(); root.close() }
        }
    }
}
