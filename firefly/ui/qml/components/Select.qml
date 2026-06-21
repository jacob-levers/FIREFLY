import QtQuick
import QtQuick.Controls

// Themed dropdown (Basic ComboBox restyled with design tokens). Binds combos by
// LABEL — `model` is a QStringList of labels, `currentText` the selected label,
// and `activated(text)` fires on a user pick. Used by the parameter sidebar.
ComboBox {
    id: root
    property color tone: Theme.palette.ACC
    signal picked(string text)

    // pill mode: rounded, compact, with an optional leading colour dot — used
    // by the timepoint selector. Defaults keep the rectangular sidebar look.
    property bool pill: false
    property color dotColor: "transparent"
    property bool dimText: false
    property color fillColor: Theme.palette.PANEL_ALT

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    implicitHeight: pill ? 26 : 28
    font.pixelSize: pill ? 11 : sc.textSm
    onActivated: (i) => root.picked(textAt(i))

    contentItem: Item {
        Rectangle {
            id: cdot
            visible: root.pill
            width: 7; height: 7; radius: 4
            anchors { left: parent.left; leftMargin: sc.sp3; verticalCenter: parent.verticalCenter }
            color: root.dotColor
        }
        Text {
            anchors {
                left: root.pill ? cdot.right : parent.left; leftMargin: sc.sp3
                right: parent.right; rightMargin: root.indicator.width + sc.sp2
                verticalCenter: parent.verticalCenter
            }
            text: root.displayText
            color: root.dimText ? pal.TXT_MUTED : pal.TXT
            font: root.font
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
    }

    background: Rectangle {
        radius: root.pill ? height / 2 : 7    // design: inputs are 6–7px
        color: root.fillColor
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
        // fade + a subtle grow on open (reduce-motion → instant)
        enter: Transition {
            ParallelAnimation {
                NumberAnimation { property: "opacity"; from: 0; to: 1
                                  duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic }
                NumberAnimation { property: "scale"; from: 0.96; to: 1
                                  duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic }
            }
        }
        exit: Transition {
            NumberAnimation { property: "opacity"; from: 1; to: 0
                              duration: Theme.reducedMotion ? 0 : 120; easing.type: Easing.OutCubic }
        }
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
