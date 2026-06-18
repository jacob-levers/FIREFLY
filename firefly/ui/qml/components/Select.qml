import QtQuick
import QtQuick.Controls

// Themed dropdown (Basic ComboBox restyled with design tokens). Binds combos by
// LABEL — `model` is a QStringList of labels, `currentText` the selected label,
// and `activated(text)` fires on a user pick. Used by the parameter sidebar.
ComboBox {
    id: root
    property color tone: Theme.palette.ACC
    signal picked(string text)

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    implicitHeight: 28
    font.pixelSize: sc.textSm
    onActivated: (i) => root.picked(textAt(i))

    contentItem: Text {
        leftPadding: sc.sp3
        rightPadding: root.indicator.width + sc.sp2
        text: root.displayText
        color: pal.TXT
        font: root.font
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        radius: sc.radiusSm
        color: pal.PANEL_ALT
        border.width: 1
        border.color: root.activeFocus || root.hovered ? pal.BORDER_HI : pal.BORDER
        Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
    }

    indicator: Icon {
        x: root.width - width - sc.sp2
        y: (root.height - height) / 2
        name: "chevron-down"; size: 14; color: pal.TXT_MUTED
    }

    delegate: ItemDelegate {
        width: root.width
        required property int index
        required property var modelData
        height: 28
        highlighted: root.highlightedIndex === index
        contentItem: Text {
            text: modelData; color: highlighted ? pal.ACC : pal.TXT
            font.pixelSize: sc.textSm; verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        background: Rectangle { color: highlighted ? pal.PANEL_ALT : "transparent" }
    }

    popup: Popup {
        y: root.height + 2
        width: root.width
        implicitHeight: Math.min(contentItem.implicitHeight + 2, 260)
        padding: 1
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: root.delegateModel
            currentIndex: root.highlightedIndex
            ScrollIndicator.vertical: ScrollIndicator {}
        }
        background: Rectangle {
            radius: sc.radiusSm; color: pal.PANEL
            border.width: 1; border.color: pal.BORDER
        }
    }
}
